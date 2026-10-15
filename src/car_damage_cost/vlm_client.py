"""Step 4: local VLM client (Ollama /api/chat).

The VLM is only used for things the segmentation models can't do:
  1. sanity-check that a photo shows a car
  2. guess the part when mask overlap is inconclusive
  3. (optional) refine severity

Every method returns None on failure so the pipeline never crashes
because of a flaky 4B model.
"""
from __future__ import annotations

import base64
import json
import re

import cv2
import httpx
import numpy as np

from .schemas import Detection, Severity


class VlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        max_side: int = 1024,
        api: str = "ollama",          # "ollama" (local /api/chat) | "openai" (any OpenAI-compatible host)
        api_key: str = "",
        extra: dict | None = None,    # extra JSON fields merged into hosted-API requests
    ):
        base = base_url.rstrip("/")
        self.api = api
        self.url = base + ("/chat/completions" if api == "openai" else "/api/chat")
        self.api_key = api_key
        self.extra = extra or {}
        self.model = model
        self.timeout = timeout
        self.max_side = max_side

    # ---------- low level ----------
    def _encode(self, img_bgr: np.ndarray) -> str:
        h, w = img_bgr.shape[:2]
        scale = self.max_side / max(h, w)
        if scale < 1:
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("could not encode image")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _chat_json(self, prompt: str, images: list[np.ndarray]) -> dict | None:
        b64s = [self._encode(i) for i in images]
        headers = {}
        if self.api == "openai":
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in b64s
            ]
            payload = {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": content}],
            }
            payload.update(self.extra)
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        else:  # ollama
            payload = {
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
                "messages": [{"role": "user", "content": prompt, "images": b64s}],
            }
        try:
            r = httpx.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            body = r.json()
            text = body["choices"][0]["message"]["content"] if self.api == "openai" else body["message"]["content"]
        except httpx.HTTPStatusError as e:
            print(f"[vlm] HTTP {e.response.status_code}: {e.response.text[:200]}")
            return None
        except Exception as e:  # network, schema
            print(f"[vlm] request failed: {e}")
            return None
        return self._parse_json(text)

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        print(f"[vlm] could not parse JSON: {text[:200]!r}")
        return None

    @staticmethod
    def _annotate(img: np.ndarray, det: Detection) -> np.ndarray:
        out = img.copy()
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), max(2, img.shape[1] // 300))
        return out

    # ---------- tasks ----------
    def check_photo(self, img: np.ndarray) -> dict | None:
        prompt = (
            "Look at this photo. Answer ONLY with JSON: "
            '{"is_car": true|false, "usable": true|false, "notes": "short reason"}. '
            "is_car = the photo shows a car or a visible part of a car. "
            "usable = it is clear enough to assess visible damage."
        )
        return self._chat_json(prompt, [img])

    def identify_part(self, img: np.ndarray, det: Detection, valid_parts: list[str]) -> str | None:
        prompt = (
            "The red rectangle marks damage on a car. Which car part is the damage on? "
            f"Choose exactly one from this list: {valid_parts}. "
            'If none fits or you cannot tell, use "unknown". '
            'Answer ONLY with JSON: {"part": "<name>"}.'
        )
        res = self._chat_json(prompt, [self._annotate(img, det)])
        if not res:
            return None
        part = str(res.get("part", "")).strip()
        lookup = {p.lower(): p for p in valid_parts}
        return lookup.get(part.lower())  # None if "unknown" or invalid

    def assess_severity(self, img: np.ndarray, det: Detection, part: str) -> Severity | None:
        prompt = (
            f"The red rectangle marks a {det.cls_name} on the {part} of a car. "
            "Rate the severity: minor (cosmetic, repairable by polish/paint), "
            "moderate (needs body work), severe (part likely needs replacement). "
            'Answer ONLY with JSON: {"severity": "minor|moderate|severe"}.'
        )
        res = self._chat_json(prompt, [self._annotate(img, det)])
        if not res:
            return None
        try:
            return Severity(str(res.get("severity", "")).strip().lower())
        except ValueError:
            return None