"""Tests for the ``skills.compact_index`` flag (names-only index)."""

from agent.prompt_builder import COMPACT_ALL_CATEGORIES, build_skills_system_prompt
from agent.skill_utils import get_compact_skill_index


def _seed(tmp_path, name="alpha-skill", desc="alpha description", category="demo"):
    d = tmp_path / "skills" / category / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n\nbody\n"
    )


# ─── flag parsing ────────────────────────────────────────────────────────────


def test_get_compact_skill_index_defaults_false(monkeypatch):
    monkeypatch.setattr("agent.skill_utils._load_raw_config", lambda: {})
    assert get_compact_skill_index() is False


def test_get_compact_skill_index_bool(monkeypatch):
    monkeypatch.setattr(
        "agent.skill_utils._load_raw_config",
        lambda: {"skills": {"compact_index": True}},
    )
    assert get_compact_skill_index() is True


def test_get_compact_skill_index_string(monkeypatch):
    monkeypatch.setattr(
        "agent.skill_utils._load_raw_config",
        lambda: {"skills": {"compact_index": "yes"}},
    )
    assert get_compact_skill_index() is True


# ─── compaction behaviour ────────────────────────────────────────────────────


def test_sentinel_compacts_all_categories(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed(tmp_path, "alpha-skill", "alpha description", "demo")
    _seed(tmp_path, "beta-skill", "beta description", "other")
    compact = build_skills_system_prompt(
        compact_categories=frozenset({COMPACT_ALL_CATEGORIES})
    )
    assert "alpha-skill" in compact
    assert "beta-skill" in compact
    assert "alpha description" not in compact
    assert "beta description" not in compact
    assert "[names only]" in compact


def test_flag_off_leaves_descriptions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.skill_utils._load_raw_config", lambda: {})
    _seed(tmp_path, "alpha-skill", "alpha description", "demo")
    full = build_skills_system_prompt()
    assert "alpha description" in full


def test_flag_on_compacts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "agent.skill_utils._load_raw_config",
        lambda: {"skills": {"compact_index": True}},
    )
    _seed(tmp_path, "alpha-skill", "alpha description", "demo")
    compact = build_skills_system_prompt()
    assert "alpha-skill" in compact
    assert "alpha description" not in compact
