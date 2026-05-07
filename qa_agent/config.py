from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = ""
    llm_timeout: int = 120
    llm_temperature: float = 0.2
    llm_top_p: float = 0.95
    llm_max_tokens: int = 32768

    text_ann_url: str = "http://127.0.0.1:8004/text_ann"
    img_ann_url: str = "http://127.0.0.1:8001/img_ann"
    multimodal_ann_url: str = "http://127.0.0.1:8003/multimodal_ann"
    bm25_ann_url: str = "http://127.0.0.1:8005/bm25_ann"
    ann_timeout: int = 500
    ann_topk: int = 5

    max_plan_rounds: int = 6
    plan_max_attempts: int = 6
    answer_max_attempts: int = 6
    allow_tool_repeat: bool = False
    max_evidence_items: int = 10

    default_choice_index: int = 0
    debug: bool = False

    # Knowledge enrichment (auto-triggered after img_ann)
    kn_lookup_url: str = "http://127.0.0.1:8002/kn_lookup"
    kn_lookup_timeout: int = 10
    img_ann_kn_top_queries: int = 2       # how many queries to look up per img_ann call
    img_ann_kn_select_mode: str = "majority"  # "majority" | "llm"
    kn_max_content_chars: int = 800       # max chars per content snippet sent to LLM

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=os.getenv("LLM_API_KEY", cls.llm_api_key),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            llm_timeout=int(os.getenv("LLM_TIMEOUT", str(cls.llm_timeout))),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", str(cls.llm_temperature))),
            llm_top_p=float(os.getenv("LLM_TOP_P", str(cls.llm_top_p))),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(cls.llm_max_tokens))),
            text_ann_url=os.getenv("TEXT_ANN_URL", cls.text_ann_url),
            img_ann_url=os.getenv("IMG_ANN_URL", cls.img_ann_url),
            multimodal_ann_url=os.getenv("MULTIMODAL_ANN_URL", cls.multimodal_ann_url),
            bm25_ann_url=os.getenv("BM25_ANN_URL", cls.bm25_ann_url),
            ann_timeout=int(os.getenv("ANN_TIMEOUT", str(cls.ann_timeout))),
            ann_topk=int(os.getenv("ANN_TOPK", str(cls.ann_topk))),
            max_plan_rounds=int(os.getenv("MAX_PLAN_ROUNDS", str(cls.max_plan_rounds))),
            plan_max_attempts=int(os.getenv("PLAN_MAX_ATTEMPTS", str(cls.plan_max_attempts))),
            answer_max_attempts=int(os.getenv("ANSWER_MAX_ATTEMPTS", str(cls.answer_max_attempts))),
            allow_tool_repeat=os.getenv("ALLOW_TOOL_REPEAT", "0") == "1",
            max_evidence_items=int(os.getenv("MAX_EVIDENCE_ITEMS", str(cls.max_evidence_items))),
            default_choice_index=int(os.getenv("DEFAULT_CHOICE_INDEX", str(cls.default_choice_index))),
            debug=os.getenv("DEBUG", "0") == "1",
            kn_lookup_url=os.getenv("KN_LOOKUP_URL", cls.kn_lookup_url),
            kn_lookup_timeout=int(os.getenv("KN_LOOKUP_TIMEOUT", str(cls.kn_lookup_timeout))),
            img_ann_kn_top_queries=int(os.getenv("IMG_ANN_KN_TOP_QUERIES", str(cls.img_ann_kn_top_queries))),
            img_ann_kn_select_mode=os.getenv("IMG_ANN_KN_SELECT_MODE", cls.img_ann_kn_select_mode),
            kn_max_content_chars=int(os.getenv("KN_MAX_CONTENT_CHARS", str(cls.kn_max_content_chars))),
        )
