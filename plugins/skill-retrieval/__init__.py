"""skill-retrieval — BM25 progressive disclosure for Hermes skills.

Two halves, gated together by the ``skills.compact_index`` config flag:

* **Compaction** (runtime half, in ``agent/prompt_builder.py``): the flag
  demotes the system-prompt skill index to names-only.
* **Retrieval** (this plugin): a ``pre_llm_call`` hook BM25-ranks the active
  profile's skills against the user message and injects the top-K
  descriptions, so the relevant skills stay in context without paying for
  every description every turn.

The hook is a no-op unless ``skills.compact_index`` is true, so installing the
plugin changes nothing until the feature is explicitly enabled. Any error
degrades to "no injection" — the turn proceeds untouched.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from .bm25 import BM25Index, tokenize
from .discovery import (
    get_profile_disabled,
    get_profile_skills_dir,
    iter_skill_files,
    load_active_skills,
)

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 6
_DESC_TRUNCATE = 200

_HEADER = (
    "## Relevant skills (BM25 top-K)\n"
    "The following skills are likely relevant to this request. Load any with "
    "skill_view(name) to get its full instructions:\n"
)

# Per-profile cache (keyed by skills dir) guarded by a lock, so a multi-profile
# gateway never serves one profile's index to another.
_lock = threading.Lock()
_cache: dict = {}  # str(skills_dir) -> {"sig": ..., "index": ..., "skills": ...}


def _parse_top_k(raw) -> int:
    if raw is None or not str(raw).strip():
        return _DEFAULT_TOP_K
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "Invalid SKILL_RETRIEVAL_TOP_K=%r; using %d", raw, _DEFAULT_TOP_K
        )
        return _DEFAULT_TOP_K
    if value < 1:
        logger.warning(
            "SKILL_RETRIEVAL_TOP_K=%r must be >= 1; using %d", raw, _DEFAULT_TOP_K
        )
        return _DEFAULT_TOP_K
    return value


TOP_K = _parse_top_k(os.environ.get("SKILL_RETRIEVAL_TOP_K"))


def _compact_enabled() -> bool:
    try:
        from agent.skill_utils import get_compact_skill_index

        return bool(get_compact_skill_index())
    except Exception:
        return False


def _skills_signature(skills_dir: Path) -> tuple:
    """Cheap identity of the skill tree (paths + mtimes), no file reads."""
    if not skills_dir.exists():
        return ()
    return tuple(
        sorted(
            (str(skill_md), skill_md.stat().st_mtime_ns)
            for skill_md, _leaf, _sid in iter_skill_files(skills_dir)
        )
    )


def _ensure_index():
    """Return ``(index, skills_by_id)`` for the active profile.

    Rebuilds only when the skill tree or the disabled set changes. The
    disabled set is part of the signature so a runtime ``skills.disabled``
    change is honoured without a restart. Atomic under ``_lock`` and keyed
    per skills dir, so concurrent multi-profile turns can't cross-contaminate.
    """
    skills_dir = get_profile_skills_dir()
    disabled = get_profile_disabled()
    sig = (_skills_signature(skills_dir), tuple(sorted(disabled)))
    key = str(skills_dir)

    with _lock:
        entry = _cache.get(key)
        if entry and entry["sig"] == sig and entry["index"] is not None:
            return entry["index"], entry["skills"]

        skills = load_active_skills(skills_dir, disabled)
        if not skills:
            _cache[key] = {"sig": sig, "index": None, "skills": {}}
            return None, {}

        index = BM25Index()
        index.build([s["skill_id"] for s in skills], [s["text"] for s in skills])
        by_id = {s["skill_id"]: s for s in skills}
        _cache[key] = {"sig": sig, "index": index, "skills": by_id}
        return index, by_id


def _lexical_fallback(query, skills_by_id: dict, top_k: int):
    """Fallback when BM25 returns nothing (small corpus / clipped idf).

    Ranks by raw shared-token overlap so a genuinely relevant skill is still
    surfaced even when every BM25 idf term clips to zero.
    """
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return []
    scored = []
    for skill_id, info in skills_by_id.items():
        overlap = len(q_tokens & set(tokenize(info["text"])))
        if overlap:
            scored.append((skill_id, float(overlap)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k]


def compose_context(results, skills_by_id: dict) -> str:
    """Render the injection block for a list of ``(skill_id, score)`` results."""
    lines = [_HEADER]
    for skill_id, _score in results:
        info = skills_by_id.get(skill_id)
        if not info:
            continue
        desc = info["description"]
        if len(desc) > _DESC_TRUNCATE:
            desc = desc[:_DESC_TRUNCATE].rstrip() + "..."
        lines.append(f"- **{info['name']}**: {desc}")
    return "\n".join(lines)


def _on_pre_llm_call(user_message, **kwargs) -> dict | None:
    """pre_llm_call hook — inject top-K relevant skill descriptions."""
    try:
        if not _compact_enabled():
            return None
        index, skills_by_id = _ensure_index()
        if index is None:
            return None
        results = index.retrieve(user_message, top_k=TOP_K)
        if not results:
            results = _lexical_fallback(user_message, skills_by_id, TOP_K)
        if not results:
            return None
        return {"context": compose_context(results, skills_by_id)}
    except Exception:
        logger.exception("skill-retrieval hook failed; injecting nothing")
        return None


def register(ctx):
    """Register the pre_llm_call retrieval hook (compaction is the runtime half)."""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("skill-retrieval plugin registered (top_k=%d)", TOP_K)
