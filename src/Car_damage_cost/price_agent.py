"""Step 7: the (very simple) price agent.

Your Gemini key works for plain calls but Google Search grounding is out of quota,
so the search is done separately:

    1. DuckDuckGo (no key)  -> search results (title, url, snippet) for this car + part
    2. Gemini plain call    -> read ONLY those results and extract a price range in EGP
    3. checks               -> every number and the source URL must literally appear in
                               the search results, otherwise the answer is rejected

Anything that fails returns None and the cost engine falls back to the placeholder
table. Results are cached on disk (30 days) to save quota.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from schemas import CarInfo

CACHE_TTL_DAYS = 30
MAX_RESULTS_PER_QUERY = 6
MAX_CONTEXT_CHARS = 4000

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _clean_part(part: str) -> str:
    """left_tail-light -> tail-light (left/right cost about the same)."""
    return re.sub(r"^(left|right)_", "", part)


def _normalize_numbers(text: str) -> str:
    """Arabic digits -> ascii, and drop thousands separators: 4,500 / 4.500 / ٤٬٥٠٠ -> 4500."""
    text = text.translate(_AR_DIGITS)
    return re.sub(r"(?<=\d)[.,\u066c](?=\d{3}(?!\d))", "", text)


class PriceAgent:
    def __init__(self, api_key: str, model: str, cache_path: str | Path = "data/price_cache.json"):
        from google import genai            # pip install google-genai

        self.client = genai.Client(api_key=api_key)   # keep the reference!
        self.model = model
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()
        self.disabled = False                # set when Gemini quota is exhausted

    # ---------- cache ----------
    def _load_cache(self) -> dict:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _key(car: CarInfo, name: str) -> str:
        return f"{car.make}|{car.model}|{car.year}|{name}".lower()

    # ---------- public ----------
    def get_price(self, car: CarInfo, part: str, expected: float | None = None):
        """-> (min_egp, max_egp, source_url) or None"""
        name = _clean_part(part)
        key = self._key(car, name)
        hit = self.cache.get(key)
        if hit and time.time() - hit["ts"] < CACHE_TTL_DAYS * 86400:
            return hit["min"], hit["max"], hit["source"]

        if self.disabled:
            return None

        context = self._gather(car, name)
        if not context:
            print(f"[price_agent] no search results for {car.make} {car.model} {name}")
            return None

        text = self._extract(car, name, context)
        parsed = self.parse_and_validate(text, expected, context)
        if parsed is None:
            return None

        lo, hi, src = parsed
        self.cache[key] = {"min": lo, "max": hi, "source": src, "ts": time.time()}
        self._save_cache()
        return lo, hi, src

    # ---------- 1. search ----------
    def _search(self, query: str) -> list[dict]:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
        except ImportError:
            print("[price_agent] search library missing: pip install ddgs")
            return []
        try:
            return list(DDGS().text(query, max_results=MAX_RESULTS_PER_QUERY)) or []
        except Exception as e:
            print(f"[price_agent] search failed: {e}")
            return []

    def _gather(self, car: CarInfo, name: str) -> str:
        human = name.replace("-", " ")
        queries = [
            f"{car.year} {car.make} {car.model} {human} price Egypt EGP",
            f"سعر {human} {car.make} {car.model} {car.year} مصر جنيه",
        ]
        seen, rows = set(), []
        for i, q in enumerate(queries):
            if i:
                time.sleep(1)                       # be gentle with DuckDuckGo
            for r in self._search(q):
                url = r.get("href") or r.get("url") or ""
                if not url or url in seen:
                    continue
                seen.add(url)
                rows.append(f"[{len(rows) + 1}] {r.get('title', '')} | {url}\n{r.get('body', '')}")

        context = "\n\n".join(rows)
        return context[:MAX_CONTEXT_CHARS]

    # ---------- 2. extract ----------
    def _extract(self, car: CarInfo, name: str, context: str) -> str:
        human = name.replace("-", " ")
        prompt = (
            f"Below are web search results. From them ONLY, extract the price range in Egyptian pounds "
            f"(EGP / جنيه / L.E.) of ONE replacement '{human}' for a {car.year} {car.make} {car.model} "
            f"(part only, no labor).\n"
            f"Rules: use only prices that appear in the results; ignore other currencies, other cars and other parts; "
            f"if a single price is given use it as both min and max; do not estimate or convert.\n"
            f'Answer ONLY with JSON: {{"min": number, "max": number, "source": "the exact URL of the result you used"}}. '
            f'If the results do not clearly contain such a price answer {{"min": null, "max": null, "source": null}}.\n\n'
            f"SEARCH RESULTS:\n{context}"
        )
        for attempt in range(2):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": 0},
                )
                return resp.text or ""
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    print("[price_agent] Gemini quota exhausted -> using the placeholder table for the rest of this run")
                    self.disabled = True
                    return ""
                print(f"[price_agent] Gemini request failed: {e}")
                if attempt == 0:
                    time.sleep(2)
        return ""

    # ---------- 3. validate ----------
    @staticmethod
    def parse_and_validate(text: str, expected: float | None, context: str | None = None):
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not m:
            return None
        try:
            d = json.loads(m.group(0))
            lo, hi = float(d["min"]), float(d["max"])
        except (ValueError, TypeError, KeyError):
            return None
        if lo <= 0 or hi < lo:
            return None

        # very wide sanity band around the (placeholder) table price
        if expected and not (expected * 0.1 <= lo and hi <= expected * 10):
            print(f"[price_agent] rejected {lo:.0f}-{hi:.0f}: far from expected ~{expected:.0f}")
            return None

        src = str(d.get("source") or "").strip()
        if context is not None:
            if not src or src not in context:
                print("[price_agent] rejected: source URL not found in the search results")
                return None
            norm = _normalize_numbers(context)
            for n in (lo, hi):
                if str(int(round(n))) not in norm:
                    print(f"[price_agent] rejected: number {n:.0f} does not appear in the search results")
                    return None
        return lo, hi, src or "web search"