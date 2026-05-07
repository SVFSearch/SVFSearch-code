#!/usr/bin/env python3
"""Minimal OpenAI-compatible vLLM client (text / vision)."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from typing import Any

from openai import OpenAI


def _to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("utf-8")
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def _extract_text_content(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "type") and getattr(item, "type") == "text":
                parts.append(str(getattr(item, "text", "")))
        return "\n".join(x for x in parts if x).strip()
    return str(content).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a local vLLM OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--prompt", default="描述这张图片")
    parser.add_argument("--image", default="", help="Optional local image path for vision input")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("Missing model. Pass --model or set LLM_MODEL.")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"Image not found: {args.image}")
        content = [
            {"type": "text", "text": args.prompt},
            {"type": "image_url", "image_url": {"url": _to_data_url(args.image)}},
        ]
    else:
        content = args.prompt

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max(1, args.max_tokens),
        temperature=args.temperature,
        top_p=args.top_p,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    print(_extract_text_content(response))


if __name__ == "__main__":
    main()
