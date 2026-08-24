"""Tests for the concurrent-write lock + audit journal (spec §3–§5).

Covers:

  * ``WriteLock`` flock sidecar semantics — determinism, acquire/release,
    per-thread re-entrancy (patch_replace -> write_file nesting must not
    deadlock or double-journal).
  * ``journal_write`` — append-only JSON lines with the required fields.
  * ``write_file`` / ``patch_replace`` journal a single ``overwrite`` /
    ``patch`` entry with before/after hashes.
  * Two-writer behavior (spec §6a/§6b): concurrent patch-class writes both
    survive (no lost update); concurrent overwrite-class writes are
    serialized last-writer-wins and both are recorded in the journal.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

import tools.concurrent_write_lock as cwl
from tools.concurrent_write_lock import WriteLock, journal_write, sidecar_path


@pytest.fixture
def locks(tmp_path, monkeypatch):
    """Redirect the shared lock directory to a temp path so tests never touch
    the real ~/.hermes/locks.  ``sidecar_path`` / ``journal_path`` resolve
    ``locks_dir()`` from the module global at call time, so patching the
    module attribute redirects every consumer."""
    d = tmp_path / "locks"
    monkeypatch.setattr(cwl, "locks_dir", lambda: d)
    return d


@pytest.fixture
def ops(tmp_path):
    """ShellFileOperations wired to a real LocalEnvironment rooted in tmp."""
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    return ShellFileOperations(LocalEnvironment(cwd=str(tmp_path), timeout=15))


def _journal_lines(locks: Path) -> list[dict]:
    jp = locks / "write.log"
    if not jp.exists():
        return []
    return [json.loads(line) for line in jp.read_text().splitlines() if line.strip()]


# ── Lock module unit tests ─────────────────────────────────────────────────


class TestSidecarPath:
    def test_deterministic_and_path_keyed(self, locks):
        a = sidecar_path("/tmp/foo/bar.txt")
        b = sidecar_path("/tmp/foo/bar.txt")
        c = sidecar_path("/tmp/foo/baz.txt")
        assert a == b
        assert a != c
        # Sidecar lives under the shared lock dir, keyed by sha256.
        assert a.parent == locks
        assert a.name.endswith(".lock")
        assert len(a.name) == len("0123456789abcdef" * 4) + len(".lock")


class TestWriteLock:
    def test_acquire_release_creates_sidecar(self, locks):
        p = "/abs/path/target.md"
        with WriteLock(p) as lock:
            assert lock.outermost is True
            assert sidecar_path(p).exists()
        # After release the sidecar file remains (flock is advisory; the file
        # is the stable lock anchor), but the lock is free.
        assert sidecar_path(p).exists()
        with WriteLock(p) as lock2:
            assert lock2.outermost is True

    def test_reentrant_nested_is_noop(self, locks):
        p = "/abs/path/target.md"
        with WriteLock(p) as outer:
            assert outer.outermost is True
            with WriteLock(p) as inner:
                assert inner.outermost is False  # nested -> not outermost

    def test_different_paths_do_not_nest(self, locks):
        with WriteLock("/a") as a:
            assert a.outermost is True
            with WriteLock("/b") as b:
                assert b.outermost is True  # different path -> independent


class TestJournal:
    def test_appends_json_line_with_fields(self, locks):
        journal_write("/some/path.md", "patch", "beforehash", "afterhash",
                      agent="dennis", session_id="sess-1")
        lines = _journal_lines(locks)
        assert len(lines) == 1
        rec = lines[0]
        assert rec["path"].endswith("/some/path.md")
        assert rec["class"] == "patch"
        assert rec["before"] == "beforehash"
        assert rec["after"] == "afterhash"
        assert rec["agent"] == "dennis"
        assert rec["session"] == "sess-1"
        assert "ts" in rec

    def test_multiple_appends_accumulate(self, locks):
        journal_write("/p", "overwrite", "", "h1", agent="a", session_id="s")
        journal_write("/p", "overwrite", "h1", "h2", agent="a", session_id="s")
        assert len(_journal_lines(locks)) == 2


# ── Write-primitive journaling (ShellFileOperations) ───────────────────────


class TestWriteFileJournals:
    def test_overwrite_journal_entry(self, locks, ops, tmp_path):
        target = tmp_path / "notes.txt"
        target.write_text("original\n")
        result = ops.write_file(str(target), "replacement\n")
        assert not result.error

        lines = _journal_lines(locks)
        assert len(lines) == 1
        rec = lines[0]
        assert rec["class"] == "overwrite"
        assert rec["before"] == cwl._hash_text("original\n")
        assert rec["after"] == cwl._hash_text("replacement\n")
        assert rec["locked"] is True

    def test_new_file_no_prior_content(self, locks, ops, tmp_path):
        target = tmp_path / "brand-new.txt"
        result = ops.write_file(str(target), "hello\n")
        assert not result.error
        lines = _journal_lines(locks)
        assert len(lines) == 1
        # Missing file -> before is null, NOT the empty-string hash.
        assert lines[0]["before"] is None
        assert lines[0]["after"] == cwl._hash_text("hello\n")
        assert lines[0]["locked"] is True


class TestPatchJournals:
    def test_patch_journal_entry(self, locks, ops, tmp_path):
        target = tmp_path / "notes.txt"
        target.write_text("alpha\nbravo\n")
        result = ops.patch_replace(str(target), "bravo", "BRAVO")
        assert result.success

        lines = _journal_lines(locks)
        assert len(lines) == 1
        rec = lines[0]
        assert rec["class"] == "patch"
        assert rec["before"] == cwl._hash_text("alpha\nbravo\n")
        assert rec["after"] == cwl._hash_text("alpha\nBRAVO\n")

    def test_patch_does_not_double_journal(self, locks, ops, tmp_path):
        """patch_replace calls write_file internally; the nested lock must not
        produce a second journal line."""
        target = tmp_path / "notes.txt"
        target.write_text("x\ny\n")
        result = ops.patch_replace(str(target), "y", "z")
        assert result.success
        assert len(_journal_lines(locks)) == 1


class TestV4aPatchJournals:
    def test_v4a_update_journals_patch(self, locks, ops, tmp_path):
        """V4A ``Update File`` is patch-class: read → modify → write under one
        lock, journaled once as ``patch``."""
        target = tmp_path / "notes.txt"
        target.write_text("one\ntwo\n")
        v4a = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            "@@\n"
            "-two\n"
            "+TWO\n"
            "*** End Patch"
        )
        result = ops.patch_v4a(v4a)
        assert result.success
        assert "TWO" in target.read_text()
        lines = _journal_lines(locks)
        assert len(lines) == 1
        assert lines[0]["class"] == "patch"


# ── Two-writer concurrency (spec §6a / §6b) ────────────────────────────────


def _fresh_ops(cwd):
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    return ShellFileOperations(LocalEnvironment(cwd=str(cwd), timeout=15))


def _patch_worker(cwd, path, old, new, locks_dir, ready, start):
    """One real OS process performing a patch-class write (spec §6a).

    Runs under a spawned interpreter: the lock-dir redirect must therefore be
    delivered via ``HERMES_LOCKS_DIR`` (an env var), not the monkeypatched
    ``locks_dir`` attribute, which does not cross the process boundary."""
    os.environ["HERMES_LOCKS_DIR"] = locks_dir
    ops = _fresh_ops(cwd)
    ready.set()
    start.wait()
    result = ops.patch_replace(path, old, new)
    if not result.success:
        raise SystemExit(f"patch failed: {result.error}")


def _overwrite_worker(cwd, path, content, locks_dir, ready, start):
    """One real OS process performing an overwrite-class write (spec §6b)."""
    os.environ["HERMES_LOCKS_DIR"] = locks_dir
    ops = _fresh_ops(cwd)
    ready.set()
    start.wait()
    result = ops.write_file(path, content)
    if result.error is not None:
        raise SystemExit(f"write failed: {result.error}")


class TestConcurrentPatchNoLostUpdate:
    def test_both_edits_survive(self, tmp_path):
        target = tmp_path / "shared.txt"
        target.write_text("line1\nline2\nline3\n")
        path = str(target)
        cwd = str(tmp_path)
        locks = tmp_path / "locks"

        ctx = mp.get_context("spawn")
        ready_a, ready_b, start = ctx.Event(), ctx.Event(), ctx.Event()
        writers = [
            ctx.Process(target=_patch_worker,
                        args=(cwd, path, "line1", "line1-A", str(locks), ready_a, start)),
            ctx.Process(target=_patch_worker,
                        args=(cwd, path, "line3", "line3-B", str(locks), ready_b, start)),
        ]
        for w in writers:
            w.start()

        try:
            # Both writers must reach the gate before we release them, so the
            # two separate OS processes genuinely contend for the flock.
            assert ready_a.wait(timeout=20), "writer A never reached the gate"
            assert ready_b.wait(timeout=20), "writer B never reached the gate"
            start.set()

            for w in writers:
                w.join(timeout=60)
            assert all(w.exitcode == 0 for w in writers)
        finally:
            # Never leak a live child: a wedged writer (e.g. a genuine flock
            # deadlock — the exact bug class this test exists to catch) must
            # surface as a fast assertion failure, not a hung CI job via
            # multiprocessing's no-timeout atexit join.
            for w in writers:
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=5)

        final = target.read_text()
        # No lost update: both independent edits are present.
        assert "line1-A" in final
        assert "line3-B" in final
        # Two patch-class journal entries.
        patches = [r for r in _journal_lines(locks) if r["class"] == "patch"]
        assert len(patches) == 2


class TestConcurrentOverwriteLastWriterWins:
    def test_serialized_lww_and_both_journaled(self, tmp_path):
        target = tmp_path / "shared.txt"
        target.write_text("seed\n")
        path = str(target)
        cwd = str(tmp_path)
        locks = tmp_path / "locks"

        contents = ["AAAA\n", "BBBB\n"]

        ctx = mp.get_context("spawn")
        ready_a, ready_b, start = ctx.Event(), ctx.Event(), ctx.Event()
        writers = [
            ctx.Process(target=_overwrite_worker,
                        args=(cwd, path, contents[0], str(locks), ready_a, start)),
            ctx.Process(target=_overwrite_worker,
                        args=(cwd, path, contents[1], str(locks), ready_b, start)),
        ]
        for w in writers:
            w.start()

        try:
            assert ready_a.wait(timeout=20), "writer A never reached the gate"
            assert ready_b.wait(timeout=20), "writer B never reached the gate"
            start.set()

            for w in writers:
                w.join(timeout=60)
            assert all(w.exitcode == 0 for w in writers)
        finally:
            # Never leak a live child (see the §6a test): a wedged writer must
            # become a fast assertion failure, not a hung CI job.
            for w in writers:
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=5)

        final = target.read_text()
        # Serialized last-writer-wins: final content is exactly one writer's.
        # (Content alone cannot prove serialization — an atomic temp+rename
        # swap lands as one full write even with no lock; the chain-hash check
        # below is what proves the ordering.)
        assert final in contents

        entries = [r for r in _journal_lines(locks) if r["class"] == "overwrite"]
        assert len(entries) == 2
        # The clobber is auditable: the second writer's before-hash equals the
        # first writer's after-hash (the write chain is recorded).
        afters = {e["after"] for e in entries}
        befores = {e["before"] for e in entries if e["before"]}
        assert afters & befores  # one writer's output is another's input


# ── Sec.4 amendment: verification inside the critical section ──────────────


class TestVerificationJournalOrdering:
    """Sec.4 amendment: verification runs inside the critical section, and the
    journal is written only after verification passes — a failed-persistence
    write is never recorded as a clean "after" hash."""

    def test_overwrite_verification_failure_skips_journal(self, locks, ops, tmp_path):
        from tools.file_operations import ExecuteResult

        target = tmp_path / "notes.txt"
        target.write_text("original\n")
        path = str(target)

        real_exec = ops._exec

        def fake_exec(command, *args, **kwargs):
            if command.startswith("sha256sum"):
                # Report a well-formed but wrong digest: the write "did not
                # persist" from the verifier's point of view.
                return ExecuteResult(stdout="0" * 64 + f"  {path}\n", exit_code=0)
            return real_exec(command, *args, **kwargs)

        ops._exec = fake_exec

        result = ops.write_file(path, "replacement\n")

        assert result.error is not None
        assert "verification failed" in result.error
        # A failed-persistence write is never recorded as a clean after-hash.
        assert _journal_lines(locks) == []

    def test_patch_verification_failure_skips_journal(self, locks, ops, tmp_path):
        from tools.file_operations import WriteResult

        target = tmp_path / "notes.txt"
        target.write_text("alpha\nbravo\n")
        path = str(target)

        # write_file is a no-op here: it reports success but does not touch
        # disk, so patch_replace's post-write re-read sees the ORIGINAL
        # content and the verification fails.
        def fake_write_file(p, content, pre_content=None):
            return WriteResult(bytes_written=len(content), verified=True)

        ops.write_file = fake_write_file

        result = ops.patch_replace(path, "bravo", "BRAVO")

        assert result.success is False
        assert result.error is not None
        assert "verification failed" in result.error
        assert _journal_lines(locks) == []
