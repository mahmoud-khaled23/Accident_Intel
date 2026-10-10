"""
report_retrieval.py
-------------------
Find the relevant law sections for a LONG accident report (for example a video description),
WITHOUT needing an LLM for the retrieval step.

Why: putting a whole report into one search query does not work well - the embedding gets
diluted by scene details (night, tram tracks, headlights ...) and every section scores about the same.
Instead this script:
  1. detects the driving behaviors in the report with keyword rules (editable below),
  2. turns each behavior into short queries written in the style of the law,
  3. puts the sections that govern those behaviors FIRST ("anchors"),
  4. lets vector search fill the remaining slots (also using any extra queries you pass in).

Usage:
    python report_retrieval.py --out ./out --file report.txt --top-k 8
    python report_retrieval.py --out ./out --file report.txt --save laws.json

From another script (for example your analyze_accident.py):
    from report_retrieval import retrieve_for_report
    laws = retrieve_for_report(description, "./out", extra_queries=queries, top_k=8)

import argparse
import json
import re
from pathlib import Path

from index_and_retrieve import load, retrieve

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
    """Flatten any JSON (dict / list / nested) into plain 'key: value' lines."""
    return "\n".join(_flatten(obj))


def load_report(raw):
    """
    Accepts plain text OR a JSON document (a dict like {"vehicles": [...], "impact": "..."},
    or a list of per-frame descriptions). Returns (text, extra_queries, extra_sections).
    If the JSON has a "queries" or "relevant_sections" list, those are used as extra search input.
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
    """Return [(rule name, queries, anchors)] for every rule whose regex matches the report."""
    return [(name, qs, anchors) for name, rx, qs, anchors in RULES
            if re.search(rx, description, re.I)]


def retrieve_for_report(description, out, extra_queries=None, extra_sections=None, top_k=8,
                        rerank=False, max_anchor_share=0.8, return_details=False):
    """
    Returns at most top_k full law sections:
      - anchors from the keyword rules first (at most max_anchor_share * top_k of them),
      - the rest filled by vector search over the rule queries + extra_queries.
    Returns the list of sections (same dict format as retrieve()).
    With return_details=True it returns (sections, rule_names_that_fired, all_queries).
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
    for p in retrieve(queries, out, top_k=top_k, expand_refs=True, rerank=rerank):
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