#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单文件部署版：图文 Embedding + ANN 召回服务

启动示例：
MULTIMODAL_EMB_MODEL_PATH=/path/to/your/model \
MULTIMODAL_ANN_PATH=/path/to/query_multimodal.final.jsonl \
python tools/multimodal_emb_server.py

请求示例：
curl -X POST 'http://127.0.0.1:8003/multimodal_ann' \
  -H 'Content-Type: application/json' \
  -d '{
        "query": "英雄联盟 雷克塞",
        "image_path": "./img/180782408256_001.jpg",
        "top_k": 5
      }'
"""

import json
import os
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from PIL import Image

from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image


# =========================
# 默认配置
# =========================
DEFAULT_INSTRUCTION = "Represent the user's input."
DEFAULT_ANN_PATH = (
    os.getenv(
        "MULTIMODAL_ANN_PATH"
    )
)
DEFAULT_MAX_EDGE = 1280
DEFAULT_TOP_K = 10
DEFAULT_MODEL_PATH = os.getenv(
    "MULTIMODAL_EMB_MODEL_PATH"
)
DEFAULT_TP_SIZE = int(os.getenv("MULTIMODAL_EMB_TP_SIZE", "1"))
DEFAULT_HOST = os.getenv("MULTIMODAL_ANN_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MULTIMODAL_ANN_PORT", "8003"))


# =========================
# emb 后处理
# 与你原始脚本一致：截断到 dim，做 L2 normalize
# 原脚本是转 float16 存盘；在线检索这里保留 float32 以便计算更稳更快
# =========================
def postprocess_emb_float32(emb, dim: int = 512, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(emb[:dim], dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm >= eps:
        v = v / norm
    return v


# =========================
# 图像处理
# 与你原脚本一致：长边限制
# =========================
def resize_long_edge(img: Image.Image, max_edge: int = 1280) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    scale = max_edge / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)


# =========================
# vLLM multimodal inputs
# 基本沿用你原始 infer 脚本
# =========================
def _apply_chat_template(
    llm: LLM,
    text: str,
    image=None,
    instruction: str = DEFAULT_INSTRUCTION,
) -> str:
    content: List[Dict[str, Any]] = []

    if image is not None:
        image_content = None
        if isinstance(image, str):
            if image.startswith(("http://", "https://", "oss://")):
                image_content = image
            else:
                abs_image_path = os.path.abspath(image)
                image_content = "file://" + abs_image_path
        else:
            image_content = image

        if image_content is not None:
            content.append({"type": "image", "image": image_content})

    content.append({"type": "text", "text": text or ""})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]
    tok = llm.llm_engine.tokenizer
    return tok.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_vllm_item(
    llm: LLM,
    text: str = "",
    image=None,
    instruction: str = DEFAULT_INSTRUCTION,
    max_edge: int = DEFAULT_MAX_EDGE,
) -> Dict[str, Any]:
    prompt = _apply_chat_template(llm, text or "", image=image, instruction=instruction)
    item: Dict[str, Any] = {"prompt": prompt}

    if image is None:
        return item

    img_obj = None
    if isinstance(image, Image.Image):
        img_obj = resize_long_edge(image.convert("RGB"), max_edge=max_edge)
    elif isinstance(image, str):
        if image.startswith(("http://", "https://", "oss://")):
            img_obj = fetch_image(image)
            if isinstance(img_obj, Image.Image):
                img_obj = resize_long_edge(img_obj.convert("RGB"), max_edge=max_edge)
        else:
            abs_path = os.path.abspath(image)
            with Image.open(abs_path) as im:
                img_obj = resize_long_edge(im.convert("RGB"), max_edge=max_edge)
    else:
        img_obj = image

    item["multi_modal_data"] = {"image": img_obj}
    return item


# =========================
# Embedding 服务
# =========================
class EmbeddingService:
    def __init__(
        self,
        model_path: str,
        tp_size: int,
        instruction: str = DEFAULT_INSTRUCTION,
        max_edge: int = DEFAULT_MAX_EDGE,
    ) -> None:
        self.model_path = model_path
        self.tp_size = tp_size
        self.instruction = instruction
        self.max_edge = max_edge

        engine_args = EngineArgs(
            model=self.model_path,
            runner="pooling",
            tensor_parallel_size=self.tp_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.6,
            max_model_len=32768,
        )
        self.llm = LLM(**vars(engine_args))

    def encode(self, query: str, image_path: str) -> np.ndarray:
        item = build_vllm_item(
            self.llm,
            text=query,
            image=image_path,
            instruction=self.instruction,
            max_edge=self.max_edge,
        )
        outs = self.llm.embed([item])
        emb = outs[0].outputs.embedding
        return postprocess_emb_float32(emb)


# =========================
# ANN 索引
# 从 query_multimodal.final.jsonl 中读取 chunk_result[].emb
# =========================
class ANNIndex:
    def __init__(self, ann_jsonl_path: str) -> None:
        self.ann_jsonl_path = ann_jsonl_path
        self.embs: Optional[np.ndarray] = None
        self.meta: List[Dict[str, Any]] = []

        self._load()

    def _load(self) -> None:
        embs: List[np.ndarray] = []
        meta: List[Dict[str, Any]] = []

        total_lines = 0
        total_chunks = 0

        with open(self.ann_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                root_query = data.get("query", "")
                root_best_img = data.get("best_img", "")

                for chunk in data.get("chunk_result", []):
                    emb = chunk.get("emb")
                    if not emb:
                        continue

                    vec = np.asarray(emb, dtype=np.float32)
                    # 刷库里理论上已经 normalize 过，这里再兜底一次
                    norm = np.linalg.norm(vec)
                    if norm > 1e-12:
                        vec = vec / norm

                    embs.append(vec)
                    meta.append(
                        {
                            "chunk_id": chunk.get("chunk_id", ""),
                            "title": chunk.get("title", ""),
                            "content": chunk.get("content", ""),
                            "source_query": root_query,
                            "source_best_img": root_best_img,
                        }
                    )
                    total_chunks += 1

        if not embs:
            raise RuntimeError(f"ANN 库为空：{self.ann_jsonl_path}")

        self.embs = np.stack(embs).astype(np.float32, copy=False)
        self.meta = meta

        print(
            f"[ANN] loaded done. lines={total_lines}, "
            f"chunks={total_chunks}, dim={self.embs.shape[1]}"
        )

    def search(self, query_emb: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if self.embs is None or len(self.meta) == 0:
            raise RuntimeError("ANN 索引未初始化")

        n = self.embs.shape[0]
        top_k = max(1, min(int(top_k), n))

        # 因为 emb 已经做了 normalize，这里 dot 就是 cosine
        scores = np.dot(self.embs, query_emb)  # shape: (N,)

        # 比全量 argsort 更快
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]

        results: List[Dict[str, Any]] = []
        for i in idx:
            item = dict(self.meta[int(i)])
            item["score"] = float(scores[int(i)])
            results.append(item)
        return results


# =========================
# 请求 / 返回结构
# =========================
class InferRequest(BaseModel):
    query: str = Field(default="", description="输入文本")
    image_path: str = Field(..., description="输入图片路径")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, description="返回top_k")


# =========================
# 全局对象
# =========================
APP = FastAPI(title="Multimodal Embedding + ANN Service")
EMBEDDER: Optional[EmbeddingService] = None
ANN: Optional[ANNIndex] = None


@APP.get("/multimodal_health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": EMBEDDER is not None,
        "ann_ready": ANN is not None,
    }


@APP.post("/multimodal_ann")
def infer(req: InferRequest) -> Dict[str, Any]:
    try:
        if EMBEDDER is None or ANN is None:
            raise RuntimeError("服务尚未初始化完成")

        if not os.path.exists(req.image_path):
            raise FileNotFoundError(f"image_path 不存在: {req.image_path}")

        query_emb = EMBEDDER.encode(query=req.query, image_path=req.image_path)
        results = ANN.search(query_emb=query_emb, top_k=req.top_k)

        return {
            "query": req.query,
            "scores": results,
        }
    except Exception as e:
        return {
            "query": req.query,
            "scores": [],
            "error": str(e),
            "traceback": traceback.format_exc(limit=3),
        }


# =========================
# 启动入口
# =========================


def main():
    global EMBEDDER, ANN

    print("[Init] loading embedding model...")
    EMBEDDER = EmbeddingService(
        model_path=DEFAULT_MODEL_PATH,
        tp_size=DEFAULT_TP_SIZE,
        instruction=DEFAULT_INSTRUCTION,
        max_edge=DEFAULT_MAX_EDGE,
    )
    print("[Init] embedding model ready.")

    print("[Init] loading ANN index...")
    ANN = ANNIndex(DEFAULT_ANN_PATH)
    print("[Init] ANN ready.")

    uvicorn.run(APP, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
