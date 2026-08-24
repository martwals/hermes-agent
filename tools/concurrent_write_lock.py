"""Concurrent-write locking for Hermes core write primitives.

Implements the flock(2) sidecar-lock + audit-journal contract from Dilbert's
"Concurrent-Write Locking" spec (Boole-adopted), sections 3-5.

The problem (spec sec. 1): Hermes spawns independent agent *sessions*
(interactive chat + email/chat webhooks) that operate against the same shared
filesystem state -- primarily skill directories under
``~/.hermes/profiles/<agent>/skills/``.  These sessions are separate OS
processes (or threads in a long-lived gateway process) and share no
coordination primitive.  ``tools.file_state`` already serializes *threads
within one process* (``threading.Lock``) and warns on stale reads, but a
``threading.Lock`` does not coordinate across processes, and a stale-read
*warning* does not prevent a lost update.  This module is the cross-process
floor: a kernel ``flock(2)`` held across the whole read -> modify -> write
critical section, plus an append-only audit journal.

Section-by-section:

  sec. 3 -- Lock mechanism.
      ``flock(2) LOCK_EX`` on a *stable sidecar lock file* --
      ``~/.hermes/locks/<sha256(abs-path)>.lock`` -- NOT the target inode.
      (write-temp-then-rename drops an inode lock; the sidecar survives it.)
      The flock is held across the entire read -> modify -> write section.
      Release is implicit: the kernel drops the lock when the fd closes, so a
      crashed session self-heals.  On acquisition we touch the sidecar to
      record an mtime "lease"; that lease is *advisory* defense-in-depth for
      the hung-but-alive holder only -- a caller must never delete a sidecar
      on age alone without confirming the holder is gone.

  sec. 4 -- Two write classes.
      patch-class (``patch``, ``skill_manage patch``) re-reads the target
      inside the lock, applies a diff, and verifies the result; the critical
      section is read -> modify -> write -> verify.  Holding flock across that
      cycle closes the race fully, and a fuzzy-match miss is a *safe* failure.
      overwrite-class (``write_file``, ``skill_manage edit``) writes complete
      content and verifies the result; the critical section is write -> verify.
      flock serializes but cannot recover a lost update the agent already baked
      in upstream -- the stat-before-patch convention (``file_state.check_stale``)
      is the permanent protection for that class.  Verification is the terminal
      read of the same cycle and lives *inside* the lock (Sec.4 amendment):
      "did MY write land?" is only answerable while the lock is held -- the
      observed state must be post-my-write and pre-any-other-writer; after
      release a concurrent lock-holder can rewrite in the gap.  The
      ``write_class`` tag recorded in the journal is what makes the distinction
      auditable.

  sec. 5 -- Audit journal.
      Append-only ``~/.hermes/locks/write.log``, one JSON line per write:
      timestamp, agent, session id, absolute path, write-class
      (``patch`` | ``overwrite``), before-hash, after-hash.  Near-zero cost;
      covers the "who wrote what" failure mode without a coordinator.

Fail-open semantics (mirrors ``cron/jobs.py``): if the lock directory cannot
be created or the flock cannot be taken, writes still proceed -- a broken lock
mechanism must not brick the agent -- but we log loudly so the degraded mode
is diagnosable.  The journal write is likewise best-effort.

Scope and limitations
---------------------
* The lock serializes writers of the *local* shared filesystem (skill and
  state directories under ``~/.hermes/``).  For remote/container backends
  (docker/ssh/modal) the sidecar lives on the host and does not coordinate
  the remote FS — those backends are single-session by nature and out of
  scope for this feature.
* The sidecar identity is ``os.path.abspath`` (local cwd), which is correct
  for the local-paths this feature targets; it does not resolve symlinks, so
  two symlinked spellings of one real file map to different sidecars.  That
  is an accepted limitation — the observed failure mode (§1) is same-agent
  sessions patching the same skill by its canonical profile path.
* The journal's ``O_APPEND`` single-``os.write`` append is atomic on local
  filesystems but NOT guaranteed on NFS; a network-mounted ``~/.hermes``
  should guard the journal with its own flock if interleaved lines matter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Bounded acquisition: a plain blocking flock has no timeout, and this lock is
# taken on the write hot path.  If a sibling process wedges while holding the
# sidecar (hung-but-alive), an unbounded block would freeze every write in the
# process silently and forever.  The shared flock primitive (reused below from
# hermes_cli.auth._file_lock) polls LOCK_NB against a deadline; on timeout we
# fall through to degraded (unlocked) mode with a loud error -- a briefly-torn
# cross-process write is strictly better than a permanently wedged write path.
# (See cron/jobs.py #60703 for the same decision.)
_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_TIMEOUT_MESSAGE = (
    "Timed out waiting for concurrent-write lock; proceeding unlocked"
)


# ── Identity resolution (best-effort) ───────────────────────────────────────
def _resolve_agent() -> str:
    """Return the active agent (profile) name, best-effort.

    Prefers an explicit env override, then derives the profile name from a
    profile-shaped ``HERMES_HOME`` (``<root>/profiles/<name>``).  Falls back
    to ``"default"`` -- the journal is an audit trail, not a security gate, so
    a conservative best-effort name is acceptable.
    """
    for var in ("HERMES_AGENT", "HERMES_PROFILE"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    home = os.environ.get("HERMES_HOME", "").strip()
    if home:
        p = Path(home)
        if p.name and p.parent.name == "profiles":
            return p.name
    return "default"


def _resolve_session_id() -> str:
    """Return the current session id, best-effort.

    Spawners set ``HERMES_SESSION_ID`` (see ``acp_adapter/server.py``); when
    absent we fall back to a per-process id so the journal still distinguishes
    writers.
    """
    val = os.environ.get("HERMES_SESSION_ID", "").strip()
    if val:
        return val
    return f"pid-{os.getpid()}"


# ── Lock directory / sidecar path ───────────────────────────────────────────
def locks_dir() -> Path:
    """Return the shared lock directory (``~/.hermes/locks`` in the standard
    layout).

    ``HERMES_LOCKS_DIR`` overrides the location.  It is deliberately an env
    var rather than an in-process hook: the two-writer verification test
    (spec §6) spawns separate OS processes that must agree on the lock dir
    without sharing monkeypatched state across the process boundary.  The
    default path is resolved lazily so this module never forces an import
    cycle at load time."""
    override = os.environ.get("HERMES_LOCKS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "locks"


def sidecar_path(abs_path: str) -> Path:
    """Return the sidecar lock file for ``abs_path``.

    The sidecar is keyed by the SHA-256 of the *absolute* path so two
    sessions addressing the same file (via different relative spellings)
    collide on the same lock, while unrelated files never contend.
    """
    digest = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()
    return locks_dir() / f"{digest}.lock"


def journal_path() -> Path:
    """Return the audit journal path (``~/.hermes/locks/write.log``)."""
    return locks_dir() / "write.log"


# ── Hashing helper ──────────────────────────────────────────────────────────
def _hash_text(text: Optional[str]) -> Optional[str]:
    """SHA-256 hex digest of ``text`` (UTF-8).

    ``None`` -> ``None``, meaning "no prior content" (file did not exist /
    was not read).  This is deliberately distinct from the hash of an empty
    file, which is the digest of ``""`` — so "missing" and "empty" never
    collide in the journal.
    """
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Re-entrancy tracking (per-thread) ───────────────────────────────────────
# flock is per *open file description*: two separate open() of the same
# sidecar by the same process contend with each other, so a nested
# ``WriteLock(path)`` on the same thread (patch_replace -> write_file) would
# deadlock against itself.  Track held sidecar paths per thread so a nested
# acquisition on the same path is a no-op depth bump, not a re-flock.
_tls = threading.local()


def _held() -> dict:
    held = getattr(_tls, "held", None)
    if held is None:
        held = _tls.held = {}
    return held


class WriteLock:
    """flock(2)-backed cross-process write lock for a single absolute path.

    Re-entrant per thread (nested acquisition of the same path is a no-op),
    and aware of whether *this* acquisition is the outermost one -- the flag
    callers use to journal exactly once per logical write.

    Usage::

        with WriteLock(abs_path) as lock:
            # read -> modify -> write critical section
            if lock.outermost:
                journal_write(..., write_class="patch", before=b, after=a)
    """

    def __init__(self, abs_path: str):
        self._path = os.path.abspath(abs_path)
        self._ctx = None
        self._locked = False
        self._outermost = False

    @property
    def outermost(self) -> bool:
        """True when this acquisition is the outermost for this thread+path."""
        return self._outermost

    @property
    def locked(self) -> bool:
        """True when the cross-process flock is actually held.

        False in degraded (fail-open) mode — e.g. the lock directory was
        unwritable or the flock timed out — where the write proceeded
        WITHOUT cross-process serialization.  Callers journal this flag so
        the audit trail can tell "serialized write" from "raced write".
        """
        return self._locked

    def __enter__(self) -> "WriteLock":
        held = _held()
        depth = held.get(self._path, 0)
        if depth > 0:
            # Nested acquisition on the same thread+path -- already locked.
            held[self._path] = depth + 1
            self._outermost = False
            return self

        self._acquire()
        held[self._path] = 1
        self._outermost = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        held = _held()
        depth = held.get(self._path, 0)
        if depth <= 1:
            held.pop(self._path, None)
            self._release()
        else:
            held[self._path] = depth - 1

    def _acquire(self) -> None:
        sp = sidecar_path(self._path)
        # Reuse the shared flock primitive (hermes_cli.auth._file_lock) rather
        # than duplicating the LOCK_EX|LOCK_NB poll loop.  A fresh holder is
        # passed per acquisition: _file_lock's own reentrancy is keyed on
        # ``holder.depth``, while our per-path reentrancy lives in _held(), so a
        # nested acquisition of a *different* path still takes a real,
        # independent flock.  Imported lazily to avoid an import cycle at module
        # load (auth pulls config / credential_persistence).
        from hermes_cli.auth import _file_lock

        holder = threading.local()
        try:
            self._ctx = _file_lock(
                sp, holder, _LOCK_TIMEOUT_SECONDS, _LOCK_TIMEOUT_MESSAGE
            )
            self._ctx.__enter__()
            self._locked = True
        except TimeoutError:
            logger.error(
                "Timed out after %.0fs waiting for write lock %s "
                "(held by another process). Proceeding unlocked.",
                _LOCK_TIMEOUT_SECONDS, sp,
            )
            self._ctx = None
            self._locked = False
        except OSError as exc:
            logger.warning(
                "concurrent-write lock unavailable for %s (%s); "
                "proceeding without cross-process lock",
                self._path, exc,
            )
            self._ctx = None
            self._locked = False

        # Record the mtime lease (advisory, see module docstring sec. 3).
        # Only touched on a *successful* acquisition — an unlocked (fail-open)
        # writer has no lease to record.
        if self._locked:
            try:
                os.utime(sp, None)
            except OSError:
                pass

    def _release(self) -> None:
        ctx = self._ctx
        self._ctx = None
        if ctx is None:
            return
        try:
            ctx.__exit__(None, None, None)
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "error releasing concurrent-write lock for %s", self._path
            )


# ── Journal ─────────────────────────────────────────────────────────────────
def journal_write(
    abs_path: str,
    write_class: str,
    before_hash: Optional[str],
    after_hash: Optional[str],
    *,
    agent: Optional[str] = None,
    session_id: Optional[str] = None,
    ts: Optional[float] = None,
    locked: bool = True,
) -> None:
    """Append one audit line to ``~/.hermes/locks/write.log``.

    Best-effort: a journal failure must never fail the write it records.  The
    line is a single JSON object so arbitrary paths (spaces, quotes, unicode,
    newlines) round-trip losslessly.  Written with O_APPEND in one
    ``os.write`` call, which POSIX guarantees is atomic for the offset on
    local filesystems, so concurrent appenders don't interleave mid-line.
    (O_APPEND atomicity is NOT guaranteed on NFS — see the module docstring.)

    ``before_hash`` / ``after_hash`` may be ``None`` to mean "no prior / no
    resulting content" (a missing file), distinct from the digest of an empty
    file.  ``locked`` records whether the cross-process flock was actually
    held for this write, so the journal can tell a serialized write from a
    degraded (raced) one.
    """
    try:
        jp = journal_path()
        jp.parent.mkdir(parents=True, exist_ok=True)
        when = datetime.fromtimestamp(
            ts if ts is not None else time.time(), tz=timezone.utc
        ).isoformat()
        record = {
            "ts": when,
            "agent": agent if agent is not None else _resolve_agent(),
            "session": session_id if session_id is not None else _resolve_session_id(),
            "path": os.path.abspath(abs_path),
            "class": write_class,
            "before": before_hash,
            "after": after_hash,
            "locked": bool(locked),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        fd = os.open(jp, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except OSError as exc:
        logger.warning("concurrent-write journal append failed (%s)", exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("concurrent-write journal append failed (%s)", exc)


__all__ = [
    "WriteLock",
    "journal_write",
    "locks_dir",
    "sidecar_path",
    "journal_path",
]
