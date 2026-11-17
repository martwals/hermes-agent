"""SUP-52 — ``skills.compact_index`` and the names-only skill index.

The flag is the runtime half of progressive disclosure: when it is on, every
skill name stays in the system prompt but descriptions are dropped, cutting
per-turn token cost. The retrieval half is the ``pre_llm_call`` skill-retrieval
plugin, which gates its own injection on this same flag.

These rows pin three things that failed or could fail silently:

1. the flag's coercion — bool, the string forms ``hermes config set`` can leave
   in config.yaml, and every unreadable shape answering False rather than
   compacting the index by accident;
2. that the sentinel ``COMPACT_ALL_CATEGORIES`` actually demotes **every**
   category. It does not do so by matching a category name: before SUP-52 the
   demotion compared top-level segments only, so a frozenset({"*"}) matched
   nothing and the flag would have been a silent no-op;
3. that ``build_skills_system_prompt()`` reads the flag itself, with no caller
   change and no argument, which is what the plugin deploy depends on.
"""

from __future__ import annotations

import pytest

from agent.prompt_builder import COMPACT_ALL_CATEGORIES, build_skills_system_prompt
from agent.skill_utils import get_compact_skill_index


@pytest.fixture
def raw_config(monkeypatch):
    """Control what ``_load_raw_config()`` answers, without touching a real config."""

    def _set(mapping):
        monkeypatch.setattr("agent.skill_utils._load_raw_config", lambda: mapping)

    return _set


# ---- 1. the flag's coercion -------------------------------------------------------------


def test_absent_skills_section_is_off(raw_config):
    raw_config({})
    assert get_compact_skill_index() is False


def test_absent_key_is_off(raw_config):
    raw_config({"skills": {"disabled": ["x"]}})
    assert get_compact_skill_index() is False


def test_booleans_pass_through(raw_config):
    raw_config({"skills": {"compact_index": True}})
    assert get_compact_skill_index() is True
    raw_config({"skills": {"compact_index": False}})
    assert get_compact_skill_index() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "always", " On "])
def test_string_truthy_forms_are_on(raw_config, value):
    raw_config({"skills": {"compact_index": value}})
    assert get_compact_skill_index() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "loud", "2"])
def test_string_falsy_forms_are_off(raw_config, value):
    raw_config({"skills": {"compact_index": value}})
    assert get_compact_skill_index() is False


def test_a_non_mapping_skills_section_is_off(raw_config):
    # ``skills: on`` in config.yaml is a shape a hand edit produces; it must not
    # compact anything (and must not raise).
    raw_config({"skills": "on"})
    assert get_compact_skill_index() is False


def test_it_reads_the_real_config_file(monkeypatch, tmp_path):
    """No monkeypatched loader: the flag must resolve from an actual config.yaml."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("skills:\n  compact_index: true\n", encoding="utf-8")
    assert get_compact_skill_index() is True


# ---- 2. the sentinel demotes every category ---------------------------------------------


def _skills_tree(tmp_path):
    """Two categories plus a nested one, each with a description."""
    root = tmp_path / "skills"
    for category, skill, description in (
        ("tools", "web-search", "Search the web"),
        ("social-media", "thread-writer", "Write threads"),
        ("social-media/twitter", "thread-hook", "Write hooks"),
    ):
        skill_dir = root / category / skill
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill}\ndescription: {description}\n---\n", encoding="utf-8"
        )
    return root


def test_the_sentinel_demotes_every_category(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _skills_tree(tmp_path)

    compact = build_skills_system_prompt(compact_categories=frozenset({COMPACT_ALL_CATEGORIES}))

    # Names survive (nothing is ever hidden); every description is gone.
    for name in ("web-search", "thread-writer", "thread-hook"):
        assert name in compact, f"{name} must stay visible and loadable"
    for description in ("Search the web", "Write threads", "Write hooks"):
        assert description not in compact, f"{description!r} should have been demoted"
    assert "names only" in compact


def test_one_named_category_does_not_demote_the_others(monkeypatch, tmp_path):
    """The sentinel must not change what a specific category list does."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _skills_tree(tmp_path)

    partial = build_skills_system_prompt(compact_categories=frozenset({"tools"}))

    assert "Search the web" not in partial  # demoted on request
    assert "web-search" in partial  # ...and its name still survives the demotion
    assert "Write threads" in partial  # untouched category keeps its description
    # The nested category follows its parent, so a category-level list demotes
    # both "social-media" and "social-media/twitter" — asserted rather than assumed.
    nested = build_skills_system_prompt(compact_categories=frozenset({"social-media"}))
    assert "Write threads" not in nested
    assert "Write hooks" not in nested
    assert "thread-hook" in nested
    assert "Search the web" in nested


# ---- 3. the flag alone is enough — no caller change --------------------------------------


def test_the_flag_compacts_the_index_with_no_argument(monkeypatch, tmp_path):
    """The deploy's whole premise: config on → compact, config off → full."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _skills_tree(tmp_path)
    (tmp_path / "config.yaml").write_text("skills:\n  compact_index: true\n", encoding="utf-8")

    compact = build_skills_system_prompt()

    assert "web-search" in compact
    assert "Search the web" not in compact


def test_the_flag_off_leaves_the_index_uncompacted(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _skills_tree(tmp_path)
    (tmp_path / "config.yaml").write_text("skills:\n  compact_index: false\n", encoding="utf-8")

    full = build_skills_system_prompt()

    assert "Search the web" in full
    assert "Write threads" in full

def test_the_flag_wins_over_a_caller_supplied_coding_posture(monkeypatch, tmp_path):
    """SUP-52 review MAJOR 1: production passes a non-empty set in coding focus.

    ``agent/system_prompt._skills_prompt`` calls
    ``build_skills_system_prompt(compact_categories=coding_compact_skill_categories(...) or None)``,
    so an ``is None`` guard around the flag read would ignore ``skills.compact_index`` in exactly the
    case it exists for and leave every category outside the posture carrying its description.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _skills_tree(tmp_path)
    (tmp_path / "config.yaml").write_text("skills:\n  compact_index: true\n", encoding="utf-8")

    compact = build_skills_system_prompt(compact_categories=frozenset({"tools"}))

    # Not just the posture's own category: the whole index is names-only.
    for description in ("Search the web", "Write threads", "Write hooks"):
        assert description not in compact, f"{description!r} survived a flag-on build"
    for name in ("web-search", "thread-writer", "thread-hook"):
        assert name in compact


def test_flag_off_still_honours_the_callers_posture(monkeypatch, tmp_path):
    """The flag must not take over the posture path when it is off."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _skills_tree(tmp_path)
    (tmp_path / "config.yaml").write_text("skills:\n  compact_index: false\n", encoding="utf-8")

    partial = build_skills_system_prompt(compact_categories=frozenset({"tools"}))

    assert "Search the web" not in partial  # the posture demoted this one
    assert "Write threads" in partial  # and only this one
    assert "Write hooks" in partial
