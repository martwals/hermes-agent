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
import threading
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


def _fresh_ops(tmp_path):
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    return ShellFileOperations(LocalEnvironment(cwd=str(tmp_path), timeout=15))


class TestConcurrentPatchNoLostUpdate:
    def test_both_edits_survive(self, locks, tmp_path):
        target = tmp_path / "shared.txt"
        target.write_text("line1\nline2\nline3\n")
        path = str(target)

        results = {}
        barrier = threading.Barrier(2)

        def patch_a():
            ops = _fresh_ops(tmp_path)
            barrier.wait()
            results["a"] = ops.patch_replace(path, "line1", "line1-A")

        def patch_b():
            ops = _fresh_ops(tmp_path)
            barrier.wait()
            results["b"] = ops.patch_replace(path, "line3", "line3-B")

        ta = threading.Thread(target=patch_a)
        tb = threading.Thread(target=patch_b)
        ta.start(); tb.start()
        ta.join(); tb.join()

        assert results["a"].success
        assert results["b"].success
        final = target.read_text()
        # No lost update: both independent edits are present.
        assert "line1-A" in final
        assert "line3-B" in final
        # Two patch-class journal entries.
        patches = [r for r in _journal_lines(locks) if r["class"] == "patch"]
        assert len(patches) == 2


class TestConcurrentOverwriteLastWriterWins:
    def test_serialized_lww_and_both_journaled(self, locks, tmp_path):
        target = tmp_path / "shared.txt"
        target.write_text("seed\n")
        path = str(target)

        contents = ["AAAA\n", "BBBB\n"]
        results = {}
        barrier = threading.Barrier(2)

        def writer(idx):
            ops = _fresh_ops(tmp_path)
            barrier.wait()
            results[idx] = ops.write_file(path, contents[idx])

        ta = threading.Thread(target=writer, args=(0,))
        tb = threading.Thread(target=writer, args=(1,))
        ta.start(); tb.start()
        ta.join(); tb.join()

        assert not results[0].error
        assert not results[1].error
        final = target.read_text()
        # Serialized last-writer-wins: final content is exactly one writer's.
        assert final in contents

        entries = [r for r in _journal_lines(locks) if r["class"] == "overwrite"]
        assert len(entries) == 2
        # The clobber is auditable: the second writer's before-hash equals the
        # first writer's after-hash (the write chain is recorded).
        afters = {e["after"] for e in entries}
        befores = {e["before"] for e in entries if e["before"]}
        assert afters & befores  # one writer's output is another's input
