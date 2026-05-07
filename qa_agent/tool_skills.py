from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolSkill:
    tool: str
    path: str
    content: str
    loaded: bool


class ANNToolSkills:
    """Load ANN SKILL.md files and render them for planner prompts."""

    def __init__(self, root: Path | None = None, max_chars_per_skill: int = 1800) -> None:
        base = root or Path(__file__).resolve().parent.parent
        self.max_chars_per_skill = max(300, max_chars_per_skill)
        self.skill_paths: list[tuple[str, Path]] = [
            ("text_ann", base / "skills" / "ann-text-recall" / "SKILL.md"),
            ("img_ann", base / "skills" / "ann-image-recall" / "SKILL.md"),
            ("multimodal_ann", base / "skills" / "ann-multimodal-recall" / "SKILL.md"),
            ("bm25_ann", base / "skills" / "ann-bm25-recall" / "SKILL.md"),
            # kn-lookup is auto-triggered after img_ann; included here so the
            # planner is aware of the knowledge enrichment workflow.
            ("kn_lookup", base / "skills" / "kn-lookup" / "SKILL.md"),
        ]
        self.skills = self._load()

    def _load(self) -> list[ToolSkill]:
        out: list[ToolSkill] = []
        for tool, path in self.skill_paths:
            if not path.exists():
                out.append(
                    ToolSkill(
                        tool=tool,
                        path=str(path),
                        content=f"[missing] {path}",
                        loaded=False,
                    )
                )
                continue
            raw = path.read_text(encoding="utf-8").strip()
            content = raw if len(raw) <= self.max_chars_per_skill else raw[: self.max_chars_per_skill] + "\n...<truncated>"
            out.append(
                ToolSkill(
                    tool=tool,
                    path=str(path),
                    content=content,
                    loaded=True,
                )
            )
        return out

    def render_for_prompt(self) -> str:
        blocks: list[str] = []
        for skill in self.skills:
            blocks.append(
                "\n".join(
                    [
                        f"[tool={skill.tool}] skill_file={skill.path}",
                        skill.content,
                    ]
                )
            )
        return "\n\n".join(blocks)

    def to_trace(self) -> list[dict[str, str | bool]]:
        rows: list[dict[str, str | bool]] = []
        for skill in self.skills:
            rows.append(
                {
                    "tool": skill.tool,
                    "skill_file": skill.path,
                    "loaded": skill.loaded,
                }
            )
        return rows
