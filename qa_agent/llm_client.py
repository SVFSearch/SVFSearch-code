from __future__ import annotations

import base64
import os
import re
from functools import lru_cache
from typing import Any
from typing import Optional

from openai import OpenAI

_THINK_TAG_RE = re.compile(r"(?:<think>.*?</think>|</think>)", re.DOTALL | re.IGNORECASE)


@lru_cache(maxsize=64)
def _to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as handle:
        raw = handle.read()
    encoded = base64.b64encode(raw).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    return f"data:{mime};base64,{encoded}"


def _extract_text_content(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        # Qwen3 thinking mode may return the full chain-of-thought inline as
        # "<think>...</think>\n{actual answer}" when enable_thinking is not
        # honoured by the server.  Strip those blocks so callers always receive
        # clean text without the thinking preamble.
        return _THINK_TAG_RE.sub("", content).strip()
    if isinstance(content, list):
        # Structured response: vLLM returns [{"type": "thinking", ...},
        # {"type": "text", "text": "..."}] when thinking is enabled.
        # We intentionally skip the "thinking" entries.
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
            elif hasattr(item, "type") and getattr(item, "type") == "text":
                texts.append(str(getattr(item, "text", "")))
        return "\n".join(t for t in texts if t).strip()
    return str(content).strip()


class MultimodalLLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        disable_thinking: bool = True,
    ) -> None:
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        # When False, chain-of-thought thinking is left enabled (model decides).
        # _extract_text_content will still strip <think>...</think> blocks from
        # the final output so callers always receive clean answer text.
        self.disable_thinking = disable_thinking

    def _extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"top_k": 20}
        if self.disable_thinking:
            # Disable Qwen3 chain-of-thought thinking.
            # thinking_budget_tokens=0 is the server-side hard cap;
            # chat_template_kwargs is the template-level switch.
            # Both are set so the constraint holds across vLLM versions.
            body["chat_template_kwargs"] = {"enable_thinking": False}
            body["thinking_budget_tokens"] = 0
        return body

    def chat_text(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body=self._extra_body(),
        )
        return _extract_text_content(response)

    def chat_with_image(self, image_path: str, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _to_data_url(image_path)}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body=self._extra_body(),
        )
        return _extract_text_content(response)
