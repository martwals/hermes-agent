"""Tests for the skill-retrieval plugin (BM25, discovery, graceful failure)."""

import importlib.util
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "skill-retrieval"
sys.path.insert(0, str(PLUGIN_DIR))

import bm25  # noqa: E402
import discovery  # noqa: E402


def _load_plugin_pkg():
    name = "skill_retrieval"
    spec = importlib.util.spec_from_file_location(
        name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so `from .bm25 import ...` resolves
    spec.loader.exec_module(mod)
    return mod


def _seed(tmp_path, rel, name, desc):
    d = tmp_path / rel
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n\nbody\n"
    )


# ─── tokenize ────────────────────────────────────────────────────────────────


def test_tokenize_lowercases_and_strips_punct():
    assert bm25.tokenize("Skill-Retrieval: BM25 (Okapi)!") == [
        "skill",
        "retrieval",
        "bm25",
        "okapi",
    ]


def test_tokenize_non_str_list_of_parts():
    assert bm25.tokenize([{"text": "hello"}, "world"]) == ["hello", "world"]


def test_tokenize_dict_and_none():
    assert bm25.tokenize({"text": "hello world"}) == ["hello", "world"]
    assert bm25.tokenize(None) == []


# ─── BM25 ranking ────────────────────────────────────────────────────────────


def _index(skills):
    idx = bm25.BM25Index()
    idx.build([s["id"] for s in skills], [s["text"] for s in skills])
    return idx


def test_bm25_ranks_matching_term_first():
    idx = _index(
        [
            {"id": "a", "text": "alpha: deploy kubernetes clusters"},
            {"id": "b", "text": "beta: send email via gmail api"},
            {"id": "c", "text": "gamma: render images with stable diffusion"},
            {"id": "d", "text": "delta: query a sqlite database"},
        ]
    )
    results = idx.retrieve("deploy kubernetes", top_k=2)
    assert results and results[0][0] == "a"


def test_bm25_no_match_returns_empty():
    idx = _index([{"id": "a", "text": "alpha: deploy kubernetes clusters"}])
    assert idx.retrieve("zzzznonexistent", top_k=2) == []


def test_bm25_top_k_limit():
    idx = _index(
        [
            {"id": "a", "text": "alpha: one"},
            {"id": "b", "text": "beta: two"},
            {"id": "c", "text": "gamma: three"},
        ]
    )
    assert len(idx.retrieve("one two three", top_k=2)) == 2


# ─── discovery ───────────────────────────────────────────────────────────────


def test_load_active_skills_parses_name_and_desc(tmp_path):
    _seed(tmp_path, "demo/alpha", "alpha-skill", "alpha description")
    skills = discovery.load_active_skills(tmp_path, set())
    assert len(skills) == 1
    assert skills[0]["name"] == "alpha-skill"
    assert skills[0]["description"] == "alpha description"


def test_load_active_skills_filters_disabled(tmp_path):
    _seed(tmp_path, "demo/alpha", "alpha-skill", "alpha description")
    _seed(tmp_path, "demo/beta", "beta-skill", "beta description")
    skills = discovery.load_active_skills(tmp_path, {"alpha-skill"})
    assert [s["name"] for s in skills] == ["beta-skill"]


def test_load_active_skills_skips_missing_description(tmp_path):
    _seed(tmp_path, "demo/alpha", "alpha-skill", "alpha description")
    d = tmp_path / "demo" / "beta"
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text("---\nname: beta-skill\n---\n\nbody\n")
    skills = discovery.load_active_skills(tmp_path, set())
    assert [s["name"] for s in skills] == ["alpha-skill"]


def test_parse_skill_md_malformed_returns_empty(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text("no frontmatter here")
    assert discovery.parse_skill_md(p) == ("", "")


# ─── profile-path resolution (FR-1) ──────────────────────────────────────────


def test_profile_skills_dir_respects_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert discovery.get_profile_skills_dir() == tmp_path / "skills"


# ─── hook (FR-3 / FR-6) ──────────────────────────────────────────────────────


def test_hook_injects_when_compact_enabled(monkeypatch, tmp_path):
    pkg = _load_plugin_pkg()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed(
        tmp_path / "skills",
        "demo/alpha",
        "alpha-skill",
        "deploy kubernetes clusters",
    )
    _seed(
        tmp_path / "skills",
        "demo/beta",
        "beta-skill",
        "send email via gmail api",
    )
    _seed(
        tmp_path / "skills",
        "other/gamma",
        "gamma-skill",
        "render images with stable diffusion",
    )
    monkeypatch.setattr(pkg, "_compact_enabled", lambda: True)
    result = pkg._on_pre_llm_call("deploy kubernetes")
    assert result and result["context"]
    assert "alpha-skill" in result["context"]


def test_hook_noop_when_compact_disabled(monkeypatch):
    pkg = _load_plugin_pkg()
    monkeypatch.setattr(pkg, "_compact_enabled", lambda: False)
    assert pkg._on_pre_llm_call("anything") is None


def test_hook_graceful_on_error(monkeypatch):
    pkg = _load_plugin_pkg()

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(pkg, "_compact_enabled", lambda: True)
    monkeypatch.setattr(pkg, "_ensure_index", boom)
    assert pkg._on_pre_llm_call("anything") is None


def test_lexical_fallback_when_idf_zero(monkeypatch, tmp_path):
    """Small corpus clips BM25 idf to zero; lexical overlap still injects."""
    pkg = _load_plugin_pkg()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed(
        tmp_path / "skills",
        "demo/alpha",
        "alpha-skill",
        "deploy kubernetes clusters",
    )
    _seed(
        tmp_path / "skills",
        "demo/beta",
        "beta-skill",
        "send email via gmail api",
    )
    monkeypatch.setattr(pkg, "_compact_enabled", lambda: True)
    result = pkg._on_pre_llm_call("deploy kubernetes")
    assert result and result["context"]
    assert "alpha-skill" in result["context"]


def test_cache_invalidates_when_disabled_changes(monkeypatch, tmp_path):
    pkg = _load_plugin_pkg()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed(
        tmp_path / "skills",
        "demo/alpha",
        "alpha-skill",
        "deploy kubernetes clusters",
    )
    _seed(
        tmp_path / "skills",
        "demo/beta",
        "beta-skill",
        "send email via gmail api",
    )
    _idx, by_id = pkg._ensure_index()
    names = {s["name"] for s in by_id.values()}
    assert "beta-skill" in names and "alpha-skill" in names

    monkeypatch.setattr(pkg, "get_profile_disabled", lambda: {"beta-skill"})
    _idx2, by_id2 = pkg._ensure_index()
    names2 = {s["name"] for s in by_id2.values()}
    assert "beta-skill" not in names2
    assert "alpha-skill" in names2
