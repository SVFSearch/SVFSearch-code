from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Optional


@dataclass
class QAItem:
    item_id: str
    image_path: str
    question: str
    options: list[str]
    gt_answer: Optional[str] = None


@dataclass
class QAPrediction:
    item_id: str
    image_path: str
    question: str
    options: list[str]
    pred_index: int
    pred_answer: str
    rationale: str
    raw_output: str
    gt_answer: Optional[str] = None
    is_llm_correct: Optional[bool] = None
    route: Optional[dict[str, Any]] = None
    plan: Optional[list[str]] = None
    evidence: Optional[str] = None
    trace: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
