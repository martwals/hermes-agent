"""Concurrent-write locking for Hermes core write primitives.

Implements the flock(2) sidecar-lock + audit-journal contract from Boole-adopted
Concurrent-Write Locking spec (sections 3–5), adapted for the current Hermes
codebase (v0.21.5+).

The problem
-----------
Hermes spawns independent agent *sessions* (interactive chat + email/chat
webhooks) that operate against the same shared filesystem state -- primarily
skill directories under ``~/.hermes/profiles/<agent>/skills/``.  These sessions
are separate OS processes (or threads in a long-lived gateway process) and share
no coordination primitive.  ``tools.file_state`` already serializes *threads
within one process* (``threading.Lock``) and warns on stale reads, but a
``threading.Lock`` does not coordinate across processes, and a stale-read
*warning* does not prevent a lost update.  This module is the cross-process
floor: a kernel ``flock(2)`` held across the whole read → modify → write
critical section, plus an append-only audit journal.

Write classes (the two-write-class contract)
---------------------------------------------
  patch-class (patch_replace, V4A _apply_update, _patch_skill):
      flock spans the full read→modify→write; a fuzzy-match miss surfaces the
      conflict safely.  Locking is on the *expanded* path (``~`` resolved) so
      callers don't get different sidecars for the same inode.

  overwrite-class (write_file, _edit_skill, _write_file, _create_skill):
      flock serializes the write; lost-update prevention stays stat-before-patch
      (``file_state.check_stale``), per the adopted contract.  The
      ``write_class`` tag recorded in the journal is what makes the distinction
      auditable.

Scope and limitations
---------------------
* Locks serialize writers of the *local* shared filesystem (skill and state
  directories under ``~/.hermes/``).  Remote/container backends are single-
  session by nature and out of scope.
* Sidecar identity is ``os.path.abspath`` (local cwd) -- does not resolve
  symlinks, so two symlinked spellings of one real file map to different
  sidecars.  Accepted limitation; the observed failure mode is same-agent
  sessions patching the same skill by its canonical profile path.
* Journal's ``O_APPEND`` single-``os.write`` append is atomic on local
  filesystems but NOT guaranteed on NFS.

Fail-open semantics
-------------------
If the lock directory cannot be created or the flock cannot be taken, writes
still proceed -- a broken lock mechanism must not brick the agent -- but we log
loudly so the degraded mode is diagnosable.  The journal write is likewise
best-effort.
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

# fcntl is Unix-only; on Windows fall back to msvcrt.
try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - Unix
    msvcrt = None

# Bounded acquisition: if a sibling process wedges while holding the sidecar,
# an unbounded block would freeze every write silently and forever.  Poll
# LOCK_NB against a deadline and fall through to degraded (unlocked) mode on
# timeout.  Mirrors the pattern in cron/jobs.py.
_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_POLL_INTERVAL = 0.1


# ── Identity resolution (best-effort) ────────────────────────────────────────
def _resolve_agent() -> str:
    """Return the active agent (profile) name, best-effort.

    Prefers an explicit env override, then derives the profile name from a
    profile-shaped ``HERMES_HOME`` (``<root>/profiles/<name>``).  Falls back
    to ``"default"`` -- the journal is an audit trail, not a security gate.
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

    Spawners set ``HERMES_SESSION_ID``; when absent we fall back to a
    per-process id so the journal still distinguishes writers.
    """
    val = os.environ.get("HERMES_SESSION_ID", "").strip()
    if val:
        return val
    return f"pid-{os.getpid()}"


# ── Lock directory / sidecar path ────────────────────────────────────────────
def locks_dir() -> Path:
    """Return the shared lock directory (``~/.hermes/locks`` in the standard
    layout).

    ``HERMES_LOCKS_DIR`` overrides the location.  It is deliberately an env
    var rather than an in-process hook: the two-writer verification test
    (spec §6) spawns separate OS processes that must agree on the lock dir
    without sharing monkeypatched state across the process boundary.  The
    default path is resolved lazily so this module never forces an import
    cycle at load time.
    """
    override = os.environ.get("HERMES_LOCKS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "locks"


def sidecar_path(abs_path: str) -> Path:
    """Return the sidecar lock file for ``abs_path``.

    Keyed by SHA-256 of the *absolute* path so two sessions addressing the
    same file (via different relative spellings) collide on the same lock.

    Hashed via ``os.fsencode`` (filesystem encoding with ``surrogateescape``),
    not strict UTF-8: a path carrying surrogate-escaped bytes (a filename the
    locale cannot decode) must still hash — never raise — or a write would
    brick on a lock-key that cannot be computed.  ``os.fsencode`` never raises
    for a ``str`` path.
    """
    digest = hashlib.sha256(os.fsencode(abs_path)).hexdigest()
    return locks_dir() / f"{digest}.lock"


def journal_path() -> Path:
    """Return the audit journal path (``~/.hermes/locks/write.log``)."""
    return locks_dir() / "write.log"


# ── Hashing helper ───────────────────────────────────────────────────────────
def _hash_text(text: Optional[str]) -> Optional[str]:
    """SHA-256 hex digest of ``text`` (UTF-8, errors="surrogateescape").

    ``None`` → ``None`` means "no prior content" (file did not exist / was
    not read).  This is deliberately distinct from the hash of an empty file,
    which is the digest of ``""`` -- so "missing" and "empty" never collide in
    the journal.

    ``surrogateescape`` matches the encoding used by
    ``file_operations.write_file`` (``content.encode("utf-8", "surrogateescape")``),
    ensuring the journal after-hash is byte-identical to what lands on disk.
    """
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


# ── Re-entrancy tracking (per-thread) ────────────────────────────────────────
# flock is per *open file description*: two separate open() of the same
# sidecar by the same process contend with each other, so a nested
# ``WriteLock(path)`` on the same thread (e.g. patch_replace → write_file)
# would deadlock against itself.  Track held sidecar paths per thread so a
# nested acquisition on the same path is a no-op depth bump, not a re-flock.
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
            # read → modify → write critical section
            if lock.outermost:
                journal_write(..., write_class="patch", before=b, after=a)
    """

    def __init__(self, abs_path: str):
        self._path = os.path.abspath(abs_path)
        self._fd = None
        self._outermost = False
        self._no_backend = False

    @property
    def outermost(self) -> bool:
        """True when this acquisition is the outermost for this thread+path."""
        return self._outermost

    @property
    def locked(self) -> bool:
        """True when the cross-process flock is actually held.

        False in degraded (fail-open) mode — e.g. the lock directory was
        unwritable, the flock timed out, or no locking backend exists at all —
        where the write proceeded WITHOUT cross-process serialization.
        Callers journal this flag so the audit trail can tell "serialized
        write" from "raced write".
        """
        return self._fd is not None and not self._no_backend

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
        try:
            sp.parent.mkdir(parents=True, exist_ok=True)
            fd = open(sp, "a+b")
        except OSError as exc:
            logger.warning(
                "concurrent-write lock unavailable for %s (%s); "
                "proceeding without cross-process lock",
                self._path, exc,
            )
            self._fd = None
            return

        self._fd = fd
        if fcntl is not None:
            deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (OSError, IOError):
                    if time.monotonic() >= deadline:
                        logger.error(
                            "Timed out after %.0fs waiting for write lock %s "
                            "(held by another process). Proceeding unlocked.",
                            _LOCK_TIMEOUT_SECONDS, sp,
                        )
                        self._close_fd(fd)
                        self._fd = None
                        return
                    time.sleep(_LOCK_POLL_INTERVAL)
        elif msvcrt is not None:  # pragma: no cover - Windows
            try:
                getattr(msvcrt, "locking")(
                    fd.fileno(), getattr(msvcrt, "LK_LOCK"), 1
                )
            except OSError as exc:
                logger.warning(
                    "concurrent-write lock unavailable for %s (%s); "
                    "proceeding without cross-process lock",
                    self._path, exc,
                )
                self._close_fd(fd)
                self._fd = None
                return
        else:
            logger.debug(
                "no flock/msvcrt available; in-process-only write locking"
            )
            # No locking backend at all: an open descriptor is NOT a lock, so
            # do not let ``locked`` report True.  Keep ``_fd`` for release-time
            # cleanup but record that no cross-process fence is actually held.
            self._no_backend = True

        # Record the mtime lease (advisory, defense-in-depth for hung-but-
        # alive holder only -- never delete a sidecar on age alone without
        # confirming the holder is gone).
        try:
            os.utime(sp, None)
        except OSError:
            pass

    def _release(self) -> None:
        fd = self._fd
        self._fd = None
        if fd is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                getattr(msvcrt, "locking")(
                    fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
                )
        except (OSError, IOError):
            pass
        finally:
            self._close_fd(fd)

    @staticmethod
    def _close_fd(fd) -> None:
        if fd is None:
            return
        try:
            fd.close()
        except (OSError, AttributeError):
            pass


# ── Journal ──────────────────────────────────────────────────────────────────
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

    Parameters
    ----------
    abs_path: absolute path of the file being written.
    write_class: ``"patch"`` or ``"overwrite"``.
    before_hash: SHA-256 of content before the write, or ``None`` if missing.
    after_hash: SHA-256 of content after the write, or ``None`` if missing.
    agent: profile name (best-effort from env).
    session_id: session identifier (best-effort from env).
    ts: UTC timestamp (float); defaults to ``time.time()``.
    locked: whether the cross-process flock was actually held for this write.
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
    "_hash_text",
]
