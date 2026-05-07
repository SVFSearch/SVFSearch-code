# server_text_ann.py

import json
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from vllm import LLM
import faiss


# =========================
# config
# =========================
MODEL_PATH = os.getenv("TEXT_EMB_MODEL_PATH")
ANN_PATH = os.getenv(
    "TEXT_ANN_PATH"
)
TOP_K_DEFAULT = int(os.getenv("TEXT_ANN_TOP_K_DEFAULT", "10"))
TEXT_ANN_PORT = int(os.getenv("TEXT_ANN_PORT", "8004"))


# =========================
# postprocess（和你刷库一致）
# =========================
def postprocess_emb_fp16(emb, dim=512, eps=1e-12):
    v = np.asarray(emb[:dim], dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm >= eps:
        v = v / norm
    return v.astype(np.float32)   # ANN用float32


# =========================
# 全局变量
# =========================
app = FastAPI()

LLM_MODEL = None
INDEX = None
META = None


# =========================
# 初始化模型
# =========================
def init_model():
    global LLM_MODEL

    print("Loading vLLM embedding model...")
    LLM_MODEL = LLM(
        model=MODEL_PATH, 
        task="embed",
        gpu_memory_utilization=0.3,
        )

    print("✅ model ready")


# =========================
# 加载ANN库
# =========================
def load_ann():
    global INDEX, META

    embs = []
    metas = []

    print("Loading ANN jsonl...")

    with open(ANN_PATH, "r") as f:
        for line in f:
            data = json.loads(line)

            for c in data.get("chunk_result", []):
                emb = np.array(c["emb"], dtype=np.float32)

                embs.append(emb)
                metas.append({
                    "chunk_id": c.get("chunk_id"),
                    "title": c.get("title"),
                    "content": c.get("content")
                })

    embs = np.stack(embs)
    dim = embs.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)  # cosine（因为normalize过）
    index.add(embs)

    INDEX = index
    META = metas

    print(f"✅ ANN ready: {index.ntotal} vectors")


# =========================
# embedding（在线）
# =========================
def encode_query(text):
    outs = LLM_MODEL.embed([text])
    emb = outs[0].outputs.embedding
    emb = postprocess_emb_fp16(emb)
    return emb


# =========================
# ANN检索
# =========================
def ann_search(query_emb, top_k):
    query_emb = np.array(query_emb, dtype=np.float32)[None, :]

    scores, indices = INDEX.search(query_emb, top_k)

    results = []
    for s, i in zip(scores[0], indices[0]):
        item = META[i]

        results.append({
            "chunk_id": item["chunk_id"],
            "title": item["title"],
            "content": item["content"],
            "score": float(s)
        })

    return results


# =========================
# request
# =========================
class Req(BaseModel):
    query: str
    top_k: int = TOP_K_DEFAULT


# =========================
# API
# =========================
@app.post("/text_ann")
def search(req: Req):
    emb = encode_query(req.query)

    results = ann_search(emb, req.top_k)

    return {
        "query": req.query,
        "scores": results
    }


# =========================
# main
# =========================
if __name__ == "__main__":
    init_model()
    load_ann()
    uvicorn.run(app, host="0.0.0.0", port=TEXT_ANN_PORT)
