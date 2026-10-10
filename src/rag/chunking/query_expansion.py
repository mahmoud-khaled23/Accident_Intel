"""
query_expansion.py
------------------
Legal Query Normalization + Expansion, to run BEFORE the hybrid (dense + BM25) search.

    from query_expansion import expand_queries
    expand_queries(["he didn't yield and changed lanes"], method="rules")

Methods:
  none  : only light normalization
  rules : normalization + glossary expansion (EN <-> AR, dialect -> formal, article refs). Free, offline.
  llm   : normalization + Claude rewrites the question into legal-style queries (pip install anthropic,
          needs ANTHROPIC_API_KEY; model via QE_MODEL env var)
  both  : rules + llm

Why it helps hybrid search:
  * BM25 only matches exact words. If the user writes "موبايل" or "ran a red light" and the law text says
    "الهاتف المحمول" / "إشارة المرور", BM25 finds nothing. Expansion adds the formal terms in BOTH languages.
  * Dense search benefits from a short keyword-style "offense name" query next to the long natural question.

Output order: ALL normalized originals first, then the variants. The round-robin merge in retrieve()
therefore always serves the user's own wording first and the expansions only fill in.

Extend the glossary without touching code:  --glossary my_glossary.json
  [{"id": "...", "en": ["canonical term", "synonym"], "ar": ["المصطلح", "مرادف"]}, ...]
  A concept with the same id replaces the built-in one. The first terms of each list are the "canonical" ones
  (they are what gets added to the query); later terms act as extra triggers (colloquial wording).
"""
import json
import os
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------- normalization
_AR_DIAC = re.compile(r"[\u064B-\u0652\u0670\u0640]")              # tashkeel + tatweel
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)    # Arabic-Indic / Persian digits -> 0-9

# colloquial (Egyptian-style) -> formal wording. Keys are in "light" form (ة kept).
DIALECT = {
    "عربية": "مركبة", "العربية": "المركبة",
    "موبايل": "هاتف محمول", "الموبايل": "الهاتف المحمول",
    "تليفون": "هاتف", "التليفون": "الهاتف",
    "موتوسيكل": "دراجة نارية", "الموتوسيكل": "الدراجة النارية",
    "سواقة": "قيادة", "السواقة": "القيادة",
}


def normalize_query(q):
    """Light normalization that is safe for the embedding model: Unicode NFKC, remove tashkeel/tatweel,
    digits -> 0-9, collapse spaces, dialect words -> formal words. (Keeps ة / أ so the dense model still sees real text.)"""
    t = unicodedata.normalize("NFKC", q)
    t = _AR_DIAC.sub("", t).translate(_DIGITS)
    for k in sorted(DIALECT, key=len, reverse=True):
        t = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", DIALECT[k], t)
    return re.sub(r"\s+", " ", t).strip()


def _stem_en(w):
    if not w.isascii() or len(w) <= 4:
        return w.rstrip("e") if (w.isascii() and len(w) > 3) else w
    for suf, min_rest in (("ing", 4), ("ed", 4), ("es", 3), ("s", 3)):
        if w.endswith(suf) and len(w) - len(suf) >= min_rest:
            w = w[: -len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def key_tokens(text):
    """Aggressive normalization used ONLY for matching glossary triggers (never sent to a model)."""
    t = _AR_DIAC.sub("", text.lower()).translate(_DIGITS)
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    out = []
    for w in re.findall(r"\w+", t):
        if len(w) > 4 and w.startswith("ال"):
            w = w[2:]
        out.append(_stem_en(w))
    return out


def _contains(tokens, sub):
    n = len(sub)
    return n > 0 and any(tokens[i:i + n] == sub for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------- glossary
DEFAULT_GLOSSARY = [
    {"id": "right_of_way",
     "en": ["right of way", "yield", "give way"],
     "ar": ["أولوية المرور", "حق الأسبقية", "إعطاء الأولوية", "الأسبقية"]},
    {"id": "lane_change",
     "en": ["lane change", "changing lanes", "switch lanes"],
     "ar": ["تغيير المسار", "تغيير الحارة", "الانتقال بين الحارات", "يغير الحارة"]},
    {"id": "speeding",
     "en": ["speeding", "exceeding the speed limit", "speed limit", "excessive speed", "speed"],
     "ar": ["تجاوز السرعة", "السرعة القصوى", "السرعة المقررة", "زيادة السرعة", "سرعة زيادة", "السرعة"]},
    {"id": "red_light",
     "en": ["red light", "traffic signal", "traffic light"],
     "ar": ["الإشارة الحمراء", "إشارة المرور", "قطع الإشارة", "كسر الإشارة", "تجاوز الإشارة"]},
    {"id": "dui",
     "en": ["driving under the influence", "drunk driving", "alcohol", "narcotics"],
     "ar": ["القيادة تحت تأثير الكحول", "القيادة تحت تأثير المخدرات", "الكحول", "المخدرات", "قيادة سكران"]},
    {"id": "seat_belt",
     "en": ["seat belt", "safety belt"],
     "ar": ["حزام الأمان", "حزام المقعد"]},
    {"id": "mobile_phone",
     "en": ["mobile phone while driving", "handheld phone", "mobile phone"],
     "ar": ["استخدام الهاتف المحمول أثناء القيادة", "الهاتف المحمول"]},
    {"id": "license",
     "en": ["driving license", "driver's license", "unlicensed driving", "expired license"],
     "ar": ["رخصة القيادة", "قيادة بدون رخصة", "رخصة منتهية", "بدون رخصة"]},
    {"id": "parking",
     "en": ["parking", "illegal parking", "no parking", "double parking"],
     "ar": ["الانتظار", "ركن السيارة", "ممنوع الانتظار", "الوقوف", "صف السيارة"]},
    {"id": "collision",
     "en": ["collision", "traffic accident", "hit and run", "leaving the scene of an accident"],
     "ar": ["حادث مروري", "تصادم", "اصطدام", "الهروب من موقع الحادث"]},
    {"id": "helmet",
     "en": ["helmet", "motorcycle helmet"],
     "ar": ["الخوذة", "الدراجة النارية"]},
]


def load_glossary(path=None):
    """Built-in glossary, optionally extended/overridden by a JSON file (same id replaces the built-in concept)."""
    by_id = {c["id"]: c for c in DEFAULT_GLOSSARY}
    if path:
        for c in json.loads(Path(path).read_text(encoding="utf-8")):
            by_id[c["id"]] = c
    glossary = []
    for c in by_id.values():
        terms = list(c.get("en", [])) + list(c.get("ar", []))
        glossary.append({"id": c["id"], "en": c.get("en", []), "ar": c.get("ar", []),
                         "triggers": [key_tokens(t) for t in terms]})
    return glossary


def find_concepts(query, glossary):
    """Glossary concepts whose terms appear in the query, in order of first appearance."""
    toks = key_tokens(query)
    hits = []
    for c in glossary:
        pos = [i for trig in c["triggers"] if trig
               for i in range(len(toks) - len(trig) + 1) if toks[i:i + len(trig)] == trig]
        if pos:
            hits.append((min(pos), c))
    return [c for _, c in sorted(hits, key=lambda x: x[0])]


_ART_EN = re.compile(r"\b(?:article|art\.?|section)\s*\(?(\d+)\)?", re.I)
_ART_AR = re.compile(r"(?:المادة|مادة|المادّة)\s*\(?(\d+)\)?")


def find_article_refs(query):
    t = _AR_DIAC.sub("", query).translate(_DIGITS)
    refs = _ART_EN.findall(t) + _ART_AR.findall(t)
    return list(dict.fromkeys(refs))


def rule_variants(query, glossary, n_variants=3, terms_per_lang=3):
    """Extra queries from the glossary. Each matched concept -> one keyword-style query with its canonical
    English + Arabic terms (minus the ones already in the query)."""
    qtoks = key_tokens(query)
    variants = []
    refs = find_article_refs(query)
    if refs:
        variants.append(" ".join(f"article {n} المادة {n}" for n in refs[:2]))
    for c in find_concepts(query, glossary):
        terms = [t for t in c["en"][:terms_per_lang] + c["ar"][:terms_per_lang]
                 if not _contains(qtoks, key_tokens(t))]
        if terms:
            variants.append(", ".join(terms))
    return variants[:n_variants]


# ---------------------------------------------------------------- LLM expansion (optional)
_LLM_CACHE = {}
_LLM_SYSTEM = (
    "You rewrite a user's question into search queries for a traffic-law corpus written in English and Arabic. "
    "Return ONLY JSON: {\"queries\": [\"...\", \"...\"]}. Rules: use formal legal terminology and the names of "
    "the offenses; include synonyms; at least one query in English and one in Arabic; each query under 25 words; "
    "do NOT invent article numbers, penalties or facts that are not in the question."
)


def llm_variants(query, n_variants=3, model=None):
    key = (query, n_variants)
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model or os.environ.get("QE_MODEL", "claude-sonnet-5-5"),
            max_tokens=400,
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": f"Question: {query}\nReturn at most {n_variants} queries."}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        out = [normalize_query(str(x)) for x in data.get("queries", [])][:n_variants]
    except Exception as e:                         # never let expansion break retrieval
        print(f"[query-expansion] LLM expansion skipped: {e}")
        out = []
    _LLM_CACHE[key] = out
    return out


# ---------------------------------------------------------------- public API
def expand_queries(queries, method="rules", n_variants=3, glossary_path=None, llm_model=None, debug=False):
    """
    queries -> flat, de-duplicated list: [normalized originals..., variants of q1, variants of q2, ...]
    n_variants = max extra queries PER original query.
    """
    if method == "none":
        return [normalize_query(q) for q in queries]
    glossary = load_glossary(glossary_path) if method in ("rules", "both") else []
    originals = [normalize_query(q) for q in queries]
    extras = []
    for q in originals:
        v = []
        if method in ("rules", "both"):
            v += rule_variants(q, glossary, n_variants)
        if method in ("llm", "both"):
            v += llm_variants(q, n_variants, llm_model)
        extras.append(v[:n_variants + (n_variants if method == "both" else 0)])
        if debug:
            print(f"\n[expand] {q}")
            for x in extras[-1]:
                print(f"   + {x}")
    result, seen = [], set()
    for q in originals + [x for lst in extras for x in lst]:
        if q.lower() not in seen:
            seen.add(q.lower())
            result.append(q)
    return result


if __name__ == "__main__":
    import sys
    for q in sys.argv[1:] or ["he didn't yield and changed lanes",
                              "سواق بيعدي الاشارة الحمراء وبيكلم في الموبايل",
                              "ما هي عقوبة المادة ١٢ في حالة السرعة؟"]:
        print(q, "->")
        for x in expand_queries([q], "rules", debug=False):
            print("   ", x)