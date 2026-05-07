# server_dino_ann.py

import torch
import numpy as np
import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import uvicorn

from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch.nn.functional as F

# =========================
# torch.load patch
# =========================
_real_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


# =========================
# config
# =========================
ANN_PATH = os.getenv(
    "IMG_ANN_PATH"
)
DEVICE = os.getenv("IMG_ANN_DEVICE", "cuda")
TOP_K_DEFAULT = int(os.getenv("IMG_ANN_TOP_K_DEFAULT", "10"))
IMG_EMB_MODEL_PATH = os.getenv(
    "IMG_EMB_MODEL_PATH"
)
IMG_EMB_CKPT_PATH = os.getenv(
    "IMG_EMB_CKPT_PATH"
)
IMG_ANN_PORT = int(os.getenv("IMG_ANN_PORT", "8001"))


# =========================
# postprocess
# =========================
def postprocess_emb_fp16(emb, dim=512, eps=1e-12):
    v = emb[:dim]
    norm = np.linalg.norm(v)
    if norm >= eps:
        v = v / norm
    return v.astype(np.float32)  # ANN阶段用float32更快


# =========================
# image
# =========================
def pad_resize(img, size=448):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    img = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - new_w)//2, (size - new_h)//2))
    return canvas


def to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


# =========================
# model
# =========================
class ProjectionHeadDINOv3(nn.Module):
    def __init__(self, embedding_dim, projection_dim, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class DINOInferModel(torch.nn.Module):
    def __init__(self, model_path, emb_dim):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(model_path)

        self.proj = ProjectionHeadDINOv3(
            self.config.hidden_size,
            emb_dim,
        )

    def forward(self, images):
        feat = self.encoder(images).last_hidden_state[:, 0, :]
        feat = self.proj(feat)
        feat = F.normalize(feat, dim=-1)
        return feat


def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("encoder.") or k.startswith("proj."):
            new_sd[k] = v

    model.load_state_dict(new_sd, strict=False)
    print("[CKPT] loaded")


# =========================
# globals
# =========================
app = FastAPI()

MODEL = None
ANN_EMB = None   # (N, D)
ANN_META = None  # 对应item信息


# =========================
# load ANN库
# =========================
def load_ann():
    global ANN_EMB, ANN_META

    embs = []
    metas = []

    with open(ANN_PATH) as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            for item in data.get("pool", []):
                emb = np.array(item["emb"], dtype=np.float32)
                embs.append(emb)
                metas.append({
                    "pid": item.get("pid", 0),
                    "img": item["img"],
                    "query": query
                })

    ANN_EMB = np.stack(embs)  # (N, D)
    ANN_META = metas

    print(f"✅ ANN loaded: {ANN_EMB.shape}")


# =========================
# model init
# =========================
def init_model():
    global MODEL

    model = DINOInferModel(
        model_path=IMG_EMB_MODEL_PATH,
        emb_dim=256,
    )
    load_ckpt(model, IMG_EMB_CKPT_PATH)

    model = model.to(DEVICE).eval()
    MODEL = model

    print("✅ model ready")


# =========================
# encode
# =========================
@torch.no_grad()
def encode_single(img_tensor):
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    feat = MODEL(img_tensor)
    return feat.cpu().numpy()[0]


def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
    except:
        img = Image.new("RGB", (448, 448), (0, 0, 0))

    img = pad_resize(img, 448)
    return to_tensor(img)


# =========================
# ANN search
# =========================
def ann_search(query_emb, top_k):
    # cosine = dot
    scores = np.dot(ANN_EMB, query_emb)  # (N,)

    idx = np.argsort(-scores)[:top_k]

    results = []
    for i in idx:
        results.append({
            "query": ANN_META[i]["query"],
            "pid": ANN_META[i]["pid"],
            "img": ANN_META[i]["img"],
            "score": float(scores[i])
        })

    return results


# =========================
# request
# =========================
class Item(BaseModel):
    img: str
    top_k: int = TOP_K_DEFAULT


# =========================
# API
# =========================
@app.post("/img_ann")
def infer(item: Item):

    # ===== embedding =====
    img_tensor = load_image(item.img)
    emb = encode_single(img_tensor)
    emb = postprocess_emb_fp16(emb)

    # ===== ANN =====
    results = ann_search(emb, item.top_k)

    return {
        "scores": results
    }


# =========================
# main
# =========================
if __name__ == "__main__":
    init_model()
    load_ann()
    uvicorn.run(app, host="0.0.0.0", port=IMG_ANN_PORT)
