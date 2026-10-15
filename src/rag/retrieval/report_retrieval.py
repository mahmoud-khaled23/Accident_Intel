"""Behavior-aware law retrieval for long accident reports.

Finds the most relevant traffic-law sections for a free-text or JSON accident
report without requiring an LLM for the retrieval step.

The strategy has three phases:

1. **Keyword detection** — :data:`RULES` maps named driving behaviors (e.g.
   ``"left turn"``, ``"speed"``) to regexes matched against the report text.
   When a behavior is detected its anchor section numbers are fetched directly.
2. **Query construction** — each matching rule contributes short, law-style
   queries that are fed to the dense/BM25 vector search.
3. **Slot filling** — anchors from keyword rules fill the first
   ``max_anchor_share * top_k`` slots; vector search fills the remainder.

Key entry point: :func:`retrieve_for_report`.
Can also be run as a script (see module docstring below).

Usage:
    python report_retrieval.py --out ./out --file report.txt --top-k 8
    python report_retrieval.py --out ./out --file report.txt --save laws.json

From another script (for example your analyze_accident.py):
    from report_retrieval import retrieve_for_report
    laws = retrieve_for_report(description, "./out", extra_queries=queries, top_k=8)
"""

import argparse
import json
import re
from pathlib import Path

from .index_and_retrieve import load, retrieve


# (rule name, regex on the report text, short queries in the law's own wording, anchor section numbers)
# The order matters: earlier rules get their anchors placed first. Edit freely.
RULES = [
    ("traffic signal",
     r"traffic[- ]lights?|red (?:light|signal)|green (?:light|signal)|yellow (?:light|signal)|"
     r"traffic[- ]signals?|stop ?lights?|ran (?:a|the) (?:red|light)|signal(?:ed)? intersection",
     ["traffic-control signal red indication vehicular traffic shall stop",
      "traffic-control signal green indication traffic may proceed",
      "obey instructions of official traffic-control device"],
     ["1111", "1110"]),
    ("stop or yield sign",
     r"stop sign|yield sign|failed to stop at (?:a|the) sign",
     ["driver approaching a stop sign shall stop at marked stop line",
      "vehicle entering stop or yield intersection shall yield right of way"],
     ["1172", "1142"]),
    ("left turn",
     r"left[- ]turn|turn(?:s|ed|ing)? left|turn(?:s|ed|ing)? across",
     ["driver intending to turn left shall yield to oncoming vehicle"],
     ["1141"]),
    ("turn / lane signal",
     r"without (?:a )?(?:turn )?signal|no (?:turn )?signal|did not (?:use|indicate)|turn signal|blinker|indicator",
     ["signal of intention to turn or change lane"],
     ["1163"]),
    ("right of way at intersection",
     r"intersection|right[- ]of[- ]way|crossing paths|t[- ]?bone|broadside|side[- ]impact|entered the intersection",
     ["driver approaching an intersection shall yield the right of way to vehicle that entered from different highway"],
     ["1140"]),
    ("lane change / merge",
     r"chang\w+ lanes?|lane change|merg\w+|swerv\w+|cut\w* (?:in|off)|drift\w+ (?:into|out of)|sideswip\w+",
     ["driving on roadway laned for traffic shall drive within a single lane and not move from it unsafely"],
     ["1128"]),
    ("passing / overtaking",
     r"overtak\w+|pass(?:ed|ing) (?:on|a) |illegal pass|wrong side of the (?:road|roadway)",
     ["overtaking and passing vehicle proceeding in the same direction", "limitations on overtaking on the left"],
     ["1122", "1126"]),
    ("wrong way / opposite direction",
     r"wrong way|head[- ]on|opposite direction|oncoming (?:lane|traffic)",
     ["driving on right side of roadway", "passing vehicles proceeding in opposite directions"],
     ["1120", "1121"]),
    ("speed",
     r"speeding|excessive speed|high speed|too fast|racing|exceed\w* the (?:speed )?limit|high rate of speed",
     ["speed not reasonable and prudent under the conditions", "maximum speed limits"],
     ["1180"]),
    ("following too closely",
     r"rear[- ]end|tailgat\w+|following too closely|hit .{0,20} from behind|struck .{0,20} from behind",
     ["driver shall not follow another vehicle more closely than is reasonable and prudent"],
     ["1129"]),
    ("pedestrian",
     r"(?:hit|struck|knocked|ran over|collided with)\w* (?:a |the )?pedestrian|pedestrian (?:was|were) (?:hit|struck)",
     ["driver shall yield right of way to pedestrian crossing in crosswalk"],
     ["1151", "1152"]),
    ("reckless / intoxicated",
     r"reckless|aggressive driving|drunk|intoxicat\w+|under the influence",
     ["reckless driving", "driving while intoxicated"],
     ["1212", "1192"]),
]


SKIP_KEYS = {"queries", "relevant_sections"}      # JSON keys that are instructions, not facts


def _flatten(obj, key=""):
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in SKIP_KEYS:
                lines += _flatten(v, str(k).replace("_", " "))
    elif isinstance(obj, list):
        for x in obj:
            lines += _flatten(x, key)
    elif obj is not None and str(obj).strip():
        lines.append(f"{key}: {obj}" if key else str(obj))
    return lines


def json_to_text(obj):
    """Flatten any JSON (dict / list / nested) into plain 'key: value' lines.

    Recursively walks ``obj``, building lines of the form ``"key: value"``
    for each leaf string/number.  Keys listed in :data:`SKIP_KEYS` (``"queries"``
    and ``"relevant_sections"``) are omitted so that LLM instruction artifacts
    do not pollute the accident description text.

    Args:
        obj: Any JSON-compatible value — a ``dict``, ``list``, scalar, or
            ``None``.

    Returns:
        str: Newline-joined flat representation of all non-skipped leaf values.

    Example:
        >>> json_to_text({"impact": "front to rear", "vehicles": [{"id": "A"}]})
        'impact: front to rear\\nvehicles: id: A'
    """
    return "\n".join(_flatten(obj))


def load_report(raw):
    """Parse a raw report string that may be plain text or a JSON document.

    Accepts plain text OR a JSON document (a dict like ``{"vehicles": [...],
    "impact": "..."}``, or a list of per-frame descriptions).  Returns the
    text form of the description, plus any extra search queries and section
    numbers extracted from the JSON.

    If the JSON has a ``"queries"`` or ``"relevant_sections"`` list those are
    returned as ``extra_queries`` / ``extra_sections`` to seed the retrieval
    step with additional signals without polluting the description text.

    Args:
        raw: A string that is either plain text or a JSON-encoded dict/list.

    Returns:
        tuple[str, list[str], list[str]]: A 3-tuple of:
            - ``description`` — the flat text form of the report.
            - ``extra_queries`` — search queries from the JSON (may be empty).
            - ``extra_sections`` — section numbers from the JSON (may be empty).
    """
    try:
        obj = json.loads(raw)
    except ValueError:
        return raw, [], []
    if not isinstance(obj, (dict, list)):
        return raw, [], []
    queries = obj.get("queries", []) if isinstance(obj, dict) else []
    sections = obj.get("relevant_sections", []) if isinstance(obj, dict) else []
    return json_to_text(obj), [str(q) for q in queries or []], [str(x) for x in sections or []]


def detect(description):
    """Return rules whose regex pattern matches the accident report text.

    Iterates over :data:`RULES` and tests each rule's compiled regex against
    ``description`` (case-insensitive).

    Args:
        description: The accident report text (plain English or flattened JSON).

    Returns:
        list[tuple[str, list[str], list[str]]]: Each element is
        ``(rule_name, query_list, anchor_section_numbers)`` for every rule
        whose regex matched.

    Example:
        >>> fired = detect("The car ran a red light at the intersection.")
        >>> [name for name, _, _ in fired]
        ['traffic signal', 'right of way at intersection']
    """
    return [(name, qs, anchors) for name, rx, qs, anchors in RULES
            if re.search(rx, description, re.I)]


def retrieve_for_report(description, out, extra_queries=None, extra_sections=None, top_k=8,
                        rerank=False, max_anchor_share=0.8, return_details=False, mode="hybrid"):
    """Retrieve the most relevant traffic-law sections for an accident report.

    Combines keyword-rule anchors with vector search to return at most
    ``top_k`` law sections.  Anchor sections (determined by the fired
    :data:`RULES`) are placed first (capped at ``max_anchor_share * top_k``),
    and the remaining slots are filled by vector/BM25 search.

    Args:
        description: Accident report text (plain or flattened JSON).
        out: Path to the directory that holds ``parents.json``,
            ``children.jsonl``, and the Chroma database.
        extra_queries: Additional search query strings to merge with the
            rule-generated queries.  Defaults to ``None``.
        extra_sections: Additional section numbers (without the ``"VTL-"``
            prefix) to promote as anchors.  Defaults to ``None``.
        top_k: Maximum number of law sections to return.  Defaults to ``8``.
        rerank: If ``True``, re-score candidates with the BGE cross-encoder
            before returning.  Slower but more precise.  Defaults to ``False``.
        max_anchor_share: Maximum fraction of ``top_k`` slots that keyword-rule
            anchors may occupy.  Defaults to ``0.8``.
        return_details: If ``True``, return a 3-tuple instead of just the
            sections list.  Defaults to ``False``.
        mode: Retrieval strategy — ``"dense"``, ``"bm25"``, or ``"hybrid"``
            (default).

    Returns:
        list[dict] | tuple[list[dict], list[str], list[str]]:
            - When ``return_details=False`` (default): a list of up to
              ``top_k`` law-section dicts in the same format as
              :func:`~index_and_retrieve.retrieve`.
            - When ``return_details=True``: a 3-tuple of
              ``(sections, rule_names_that_fired, all_queries_used)``.

    Example:
        >>> laws = retrieve_for_report("Car A ran a red light.", "./out", top_k=5)
        >>> laws[0]["id"]
        'VTL-1111'
    """
    parents, _ = load(out)
    fired = detect(description)

    queries, anchors, seen = [], [], set()
    for name, qs, nums in fired:
        queries += qs
        for n in nums:
            pid = f"VTL-{n}"
            if pid in parents and parents[pid]["category"] == "core" and pid not in seen:
                seen.add(pid)
                p = dict(parents[pid])
                p["matched_queries"] = [f"rule: {name}"]
                p["score"], p["score_kind"] = None, "keyword-rule"
                anchors.append(p)
    for x in (extra_sections or []):                  # sections suggested by an LLM or by your JSON
        pid = "VTL-" + str(x).strip().lower().replace("section", "").replace("§", "").strip()
        if pid in parents and parents[pid]["category"] == "core" and pid not in seen:
            seen.add(pid)
            p = dict(parents[pid])
            p["matched_queries"] = ["suggested section (LLM / JSON)"]
            p["score"], p["score_kind"] = None, "suggested"
            anchors.append(p)
    queries += [q for q in (extra_queries or []) if q not in queries]
    if not queries:                                   # nothing detected and no extra queries
        queries = [description[:300]]

    n_anchor = min(len(anchors), max(1, int(top_k * max_anchor_share)))
    laws = anchors[:n_anchor]
    for p in retrieve(queries, out, top_k=top_k, expand_refs=True, rerank=rerank, mode=mode):
        if len(laws) >= top_k:
            break
        if p["id"] not in {x["id"] for x in laws}:
            laws.append(p)
    laws = laws[:top_k]
    return (laws, [name for name, _, _ in fired], queries) if return_details else laws


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out")
    ap.add_argument("--file", help="accident report: a .txt file or a .json file")
    ap.add_argument("--text", help="accident report given directly")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--save", help="write the retrieved sections to this .json file")
    a = ap.parse_args()
    if not (a.file or a.text):
        ap.error("give --file or --text")
    raw = a.text or Path(a.file).read_text(encoding="utf-8")
    description, extra_q, extra_s = load_report(raw)          # works for plain text and for JSON

    laws, rules, queries = retrieve_for_report(description, a.out, extra_queries=extra_q,
                                               extra_sections=extra_s, top_k=a.top_k,
                                               rerank=a.rerank, return_details=True)
    print("Behaviors detected:", rules or "none")
    print("Queries:", *[f"  - {q}" for q in queries], sep="\n")
    print(f"\nReturned {len(laws)} sections (top_k={a.top_k})")
    for i, p in enumerate(laws, 1):
        print(f"  #{i} {p['id']:<12} {p['title'][:55]:<55} <- {p['matched_queries']}")
    if a.save:
        Path(a.save).write_text(json.dumps(laws, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved to {a.save}")
