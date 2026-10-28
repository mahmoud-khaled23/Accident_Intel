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

from .schemas import CarInfo

CACHE_TTL_DAYS = 30
"""Number of days before a cached price entry is considered stale and re-fetched."""

MAX_RESULTS_PER_QUERY = 6
"""Maximum number of web search results to fetch per DuckDuckGo query."""

MAX_CONTEXT_CHARS = 4000
"""Maximum character length of the search-result context sent to Gemini."""

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _clean_part(part: str) -> str:
    """Strip left/right side prefix from a part name for price lookup.

    Left and right variants of the same part cost about the same, so the
    side prefix is irrelevant to the price search.

    Args:
        part: Full part name such as ``"left_tail-light"``.

    Returns:
        Base part name without the side prefix, e.g. ``"tail-light"``.
    """
    return re.sub(r"^(left|right)_", "", part)


def _normalize_numbers(text: str) -> str:
    """Normalise Arabic-Indic digits and thousands separators to ASCII.

    Converts extended Arabic digits (``٠١٢٣٤٥٦٧٨٩``) and Persian digits
    (``۰۱۲۳۴۵۶۷۸۹``) to their ASCII equivalents, and removes thousands
    separators (commas, dots, or Arabic comma ``٬``) from numbers like
    ``4,500`` / ``4.500`` / ``٤٬٥٠٠`` → ``4500``.

    Args:
        text: Raw text that may contain non-ASCII digit forms.

    Returns:
        Text with all digits normalised to ASCII and separators removed.
    """
    text = text.translate(_AR_DIGITS)
    return re.sub(r"(?<=\d)[.,\u066c](?=\d{3}(?!\d))", "", text)


class PriceAgent:
    """Web-search-backed part price agent using DuckDuckGo and Gemini.

    Performs a two-step price lookup:
    1. Search DuckDuckGo for the part price for the given car in Egypt.
    2. Ask Gemini to extract a price range in EGP from the search results only.

    Every result is cached to disk for :data:`CACHE_TTL_DAYS` days to
    conserve API quota.  Any failure returns ``None`` so the cost engine
    falls back gracefully to the placeholder price table.

    Attributes:
        client: Gemini API client.
        model: Gemini model identifier string.
        cache_path: Path to the JSON cache file.
        cache: In-memory cache dict (loaded from disk on init).
        disabled: Set to ``True`` when Gemini quota is exhausted; prevents
            further API calls for the rest of the run.
    """

    def __init__(self, api_key: str, model: str, cache_path: str | Path = "data/price_cache.json"):
        """Initialise the price agent.

        Args:
            api_key: Google Gemini API key.
            model: Gemini model identifier (e.g. ``"gemini-flash-lite-latest"``).
            cache_path: Path to the on-disk JSON cache. Created automatically
                if it does not exist.
        """
        from google import genai            # installed by uv sync

        self.client = genai.Client(api_key=api_key)   # keep the reference!
        self.model = model
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()
        self.disabled = False                # set when Gemini quota is exhausted

    # ---------- cache ----------
    def _load_cache(self) -> dict:
        """Load the price cache from disk, returning an empty dict on failure.

        Returns:
            The parsed cache dictionary, or ``{}`` if the file does not exist
            or cannot be parsed.
        """
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self) -> None:
        """Persist the in-memory cache to disk.

        Creates parent directories as needed.
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _key(car: CarInfo, name: str) -> str:
        """Build a canonical cache key for a car + part combination.

        Args:
            car: Vehicle information (make, model, year).
            name: Base part name (no side prefix).

        Returns:
            A lowercase string of the form ``"make|model|year|part"``.
        """
        return f"{car.make}|{car.model}|{car.year}|{name}".lower()

    # ---------- public ----------
    def get_price(self, car: CarInfo, part: str, expected: float | None = None):
        """Return a price range for replacing ``part`` on ``car``.

        Checks the disk cache first. On a cache miss, searches DuckDuckGo for
        recent prices, asks Gemini to extract the range, validates the result,
        and saves it to the cache.

        Args:
            car: Vehicle information used for the search query.
            part: Part name (side prefix is stripped automatically).
            expected: Optional placeholder price used as a sanity-band anchor.
                If the extracted price is outside 10 × / ÷ 10 of this value,
                the result is rejected.

        Returns:
            A ``(min_egp, max_egp, source_url)`` tuple on success, or ``None``
            when no reliable price could be found.
        """
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
        """Run a DuckDuckGo text search and return raw result dicts.

        Tries the ``ddgs`` package first, falling back to
        ``duckduckgo_search``.  Returns an empty list on any failure so the
        caller can handle it gracefully.

        Args:
            query: Search query string.

        Returns:
            A list of result dicts with at least ``href``/``url``, ``title``,
            and ``body`` keys.
        """
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
        except ImportError:
            print("[price_agent] search library missing: run uv sync --locked")
            return []
        try:
            return list(DDGS().text(query, max_results=MAX_RESULTS_PER_QUERY)) or []
        except Exception as e:
            print(f"[price_agent] search failed: {e}")
            return []

    def _gather(self, car: CarInfo, name: str) -> str:
        """Run English and Arabic searches and concatenate unique results.

        Issues two DuckDuckGo queries — one in English, one in Arabic — and
        deduplicates by URL. The combined context is truncated to
        :data:`MAX_CONTEXT_CHARS` characters.

        Args:
            car: Vehicle information for the search queries.
            name: Base part name.

        Returns:
            Concatenated search result text ready to send to Gemini, or an
            empty string when no results were found.
        """
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
        """Ask Gemini to extract a price range from the search result context.

        Sends a structured prompt instructing Gemini to read only the provided
        search results and output a JSON ``{"min": …, "max": …, "source": …}``.
        Retries once on transient failures.  Sets :attr:`disabled` when the
        Gemini quota is exhausted.

        Args:
            car: Vehicle information for the prompt.
            name: Base part name for the prompt.
            context: Concatenated search result text from :meth:`_gather`.

        Returns:
            The raw text response from Gemini (may contain JSON), or ``""`` on
            failure.
        """
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
                try:
                    from src.api.model_cost.usage import record_usage  # noqa: PLC0415
                    usage_meta = getattr(resp, "usage_metadata", None)
                    if usage_meta is not None:
                        pt = getattr(usage_meta, "prompt_token_count", None) or 0
                        ct = getattr(usage_meta, "candidates_token_count", None) or 0
                    else:
                        try:
                            counted = self.client.models.count_tokens(
                                model=self.model, contents=prompt
                            )
                            pt = getattr(counted, "total_tokens", None) or len(prompt) // 4
                        except Exception:
                            pt = len(prompt) // 4
                        ct = len(resp.text or "") // 4
                    record_usage(
                        model_id=self.model,
                        stage="price_agent",
                        prompt_tokens=pt,
                        completion_tokens=ct,
                    )
                except Exception as _exc:
                    import logging  # noqa: PLC0415
                    logging.getLogger(__name__).warning(
                        "model_cost: instrumentation skipped: %s", _exc
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
        """Parse and validate a Gemini JSON response containing a price range.

        Validation rules:
        * The response must contain a JSON object with ``min``, ``max``, and
          ``source`` keys.
        * ``min`` and ``max`` must be positive and ``min <= max``.
        * Both numbers and the source URL must literally appear in the
          search-result context (prevents hallucination).
        * The price must be within 10 × / ÷ 10 of ``expected`` when provided.

        Args:
            text: Raw Gemini response text to parse.
            expected: Optional anchor price from the placeholder table used as
                a sanity-band check.
            context: The original search-result text. When provided, numbers
                and the source URL must appear literally in this text.

        Returns:
            A ``(min_egp, max_egp, source_url)`` tuple when validation passes,
            or ``None`` when any check fails.
        """
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
