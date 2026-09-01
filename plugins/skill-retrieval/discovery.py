"""Skill discovery for the skill-retrieval plugin.

Profile-aware: the skills directory and disabled list are resolved from the
active Hermes profile (``HERMES_HOME``), never hardcoded to ``~/.hermes``.
The pure ``load_active_skills(skills_dir, disabled)`` unit-tests against a
seeded directory; the ``get_profile_*`` / ``resolve_active_skills`` helpers
are thin runtime glue.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories under any skill tree to skip (Hermes metadata / curator state).
_SKIP_DIRS = (".archive", ".curator_backups", ".hub")


def parse_skill_md(skill_md: Path) -> tuple[str, str]:
    """Return ``(name, description)`` from a SKILL.md frontmatter.

    Returns ``("", "")`` on any read/parse failure so one bad file never
    aborts the index build. Never raises.
    """
    import yaml

    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read %s: %s", skill_md, exc)
        return "", ""

    lines = text.splitlines()
    if not lines:
        return "", ""
    if lines[0].lstrip("\ufeff").strip() != "---":
        return "", ""

    close = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close = i
            break
    if close is None:
        return "", ""

    fm_text = "\n".join(lines[1:close])
    try:
        data = yaml.safe_load(fm_text)
    except yaml.YAMLError as exc:
        logger.warning("Malformed YAML frontmatter in %s: %s", skill_md, exc)
        return "", ""
    if not isinstance(data, dict):
        return "", ""

    name = data.get("name")
    desc = data.get("description")
    return ("" if name is None else str(name)), ("" if desc is None else str(desc))


def iter_skill_files(root: Path):
    """Yield ``(skill_md_path, leaf_name, skill_id)`` for every SKILL.md.

    ``skill_id`` is the skill's path relative to ``root`` (e.g.
    ``social-media/twitter/thread-writer``), matching how nested categories
    are laid out on disk.
    """
    if not root.exists():
        return
    for skill_md in sorted(root.rglob("SKILL.md")):
        rel_parts = skill_md.relative_to(root).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        skill_dir = skill_md.parent
        yield skill_md, skill_dir.name, str(skill_dir.relative_to(root))


def load_active_skills(skills_dir: Path, disabled: set[str]) -> list[dict]:
    """Load non-disabled skills as ``{name, skill_id, description, text}``.

    A skill with no description is skipped (nothing to rank on) — its name is
    still visible in the compacted index, so discoverability is unaffected.
    """
    skills: list[dict] = []
    for skill_md, leaf_name, skill_id in iter_skill_files(skills_dir):
        name, desc = parse_skill_md(skill_md)
        if not name:
            name = leaf_name
        if name in disabled or skill_id in disabled or leaf_name in disabled:
            continue
        if not desc:
            continue
        skills.append(
            {
                "name": name,
                "skill_id": skill_id,
                "description": desc,
                "text": f"{name}: {desc}",
            }
        )
    return skills


def get_profile_skills_dir() -> Path:
    """Active profile's skills dir (HERMES_HOME-scoped; never hardcoded)."""
    try:
        from hermes_constants import get_skills_dir

        return get_skills_dir()
    except Exception:  # standalone fallback
        home = os.environ.get("HERMES_HOME")
        return (Path(home) if home else Path.home() / ".hermes") / "skills"


def get_profile_disabled() -> set[str]:
    """Disabled skill names from the active profile's config (``skills.disabled``)."""
    try:
        from agent.skill_utils import get_disabled_skill_names

        return get_disabled_skill_names()
    except Exception:  # standalone fallback
        return set()


def resolve_active_skills() -> list[dict]:
    """All non-disabled skills from the active profile, ready for BM25."""
    return load_active_skills(get_profile_skills_dir(), get_profile_disabled())
