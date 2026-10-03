"""
analyze_accident.py
-------------------
Accident description  ->  structured facts + search queries (LLM call 1)
                      ->  relevant law sections (RAG retrieval)
                      ->  fault analysis based ONLY on those sections (LLM call 2)

Setup:
    pip install anthropic
    set ANTHROPIC_API_KEY=your_key          (Windows)   |   export ANTHROPIC_API_KEY=your_key  (Mac/Linux)

Usage:
    python analyze_accident.py --out ./out --file accident_example.txt
    python analyze_accident.py --out ./out --text "Car A ran a red light and hit car B ..." --lang ar
"""
import argparse
import json
import re
from pathlib import Path


from index_and_retrieve import retrieve
# remove:  import anthropic
from transformers import pipeline

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # multilingual, good with JSON and Arabic
MAX_WORDS_PER_LAW = 1200                     # lower cap, local models have less context/VRAM
MAX_WORDS_PER_LAW = 2500      # safety cap for very long sections

EXTRACT_SYSTEM = """You are a traffic-accident analyst. Read the accident description (it may be in
any language) and return ONLY a JSON object, no markdown, with these keys:

{
  "summary": "2-3 sentence neutral summary in English",
  "road_type": "intersection / highway / roundabout / parking lot / unknown",
  "traffic_control": "signal color, stop sign, yield sign, none, or unknown",
  "vehicles": [
    {"id": "A", "action": "what it did", "direction": "...", "speed": "...", "signals_used": "..."}
  ],
  "impact": "which part of which vehicle hit which part of which vehicle",
  "behaviors": ["short neutral descriptions of each notable driver behavior"],
  "unknowns": ["important facts that are NOT stated in the description"],
  "queries": ["4-8 short English search queries written in the style of New York Vehicle and Traffic Law"]
}

Rules:
- Describe only what the text states. If something is not stated, put it in "unknowns", do not guess.
- Queries must target the LEGAL RULE for each behavior, e.g. "disobeying traffic control signal red",
  "failure to yield right of way at intersection", "unsafe turning without signal",
  "following too closely", "speed not reasonable and prudent".
- Do not mention fault or blame in the JSON."""

ANALYZE_SYSTEM = """You are a traffic-law analyst assisting a human reviewer. You receive (1) facts of an
accident and (2) excerpts of law retrieved from a database.

Strict rules:
- Use ONLY the law excerpts provided. Never cite a section that is not in them.
- Cite every point as "VTL § <number>(<subdivision>)".
- If the excerpts do not cover a behavior, say so instead of inventing a rule.
- Do not assume facts that are not in the facts section. List what is missing.
- This is an analysis aid, not a legal judgment."""

ANALYZE_USER = """# ACCIDENT FACTS
{facts}

# RETRIEVED LAW EXCERPTS
{laws}

# TASK
Write the answer in {language} using exactly these sections:
1. **Facts used** (short bullets)
2. **Per vehicle**: for each vehicle, the possible violations, each with the section/subdivision and why it applies
3. **Most likely at fault**: who, and an approximate share of responsibility if it can be estimated; mention comparative fault if both drivers violated a rule
4. **Confidence**: high / medium / low, and the missing information that would change the conclusion
5. **Retrieved laws that do not apply** (one line each)"""

def load_llm(model_id):
    return pipeline("text-generation", model=model_id,
                    torch_dtype="auto", device_map="auto")


def call_llm(pipe, model, system, user, max_tokens=2000):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    out = pipe(messages, max_new_tokens=max_tokens,
               do_sample=False, return_full_text=False)
    return out[0]["generated_text"]


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in model output:\n{text[:500]}")
    return json.loads(text[start:end + 1])


def format_laws(laws):
    blocks = []
    for p in laws:
        words = p["text"].split()
        body = p["text"] if len(words) <= MAX_WORDS_PER_LAW else " ".join(words[:MAX_WORDS_PER_LAW]) + " [truncated]"
        blocks.append(f"## {p['id']} | matched by: {'; '.join(p.get('matched_queries', []))}\n{body}")
    return "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out")
    ap.add_argument("--file", help="text file with the accident description")
    ap.add_argument("--text", help="accident description given directly")
    ap.add_argument("--lang", default="en", choices=["en", "ar"], help="language of the final answer")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--per-query", type=int, default=3)
    ap.add_argument("--max-total", type=int, default=8)
    ap.add_argument("--show-laws", action="store_true", help="print the full retrieved law text")
    ap.add_argument("--save", help="save the report to this .md file")
    a = ap.parse_args()

    if not (a.file or a.text):
        ap.error("give --file or --text")
    description = a.text or Path(a.file).read_text(encoding="utf-8")
    client = load_llm(a.model)

    # 1) facts + queries
    print("[1/3] Extracting facts and search queries ...")
    facts = parse_json(call_llm(client, a.model, EXTRACT_SYSTEM, description))
    queries = facts.pop("queries", [])
    if not queries:
        raise SystemExit("The model returned no search queries.")
    print("Queries:", *[f"  - {q}" for q in queries], sep="\n")

    # 2) retrieval
    print("\n[2/3] Retrieving laws ...")
    laws = retrieve(queries, a.out, per_query=a.per_query, max_total=a.max_total, expand_refs=True)
    for p in laws:
        print(f"  {p['id']:<14} {p['title'][:60]:<60} <- {p['matched_queries']}")
    if a.show_laws:
        print("\n" + format_laws(laws))

    # 3) analysis
    print("\n[3/3] Analysing fault ...\n")
    language = "Arabic" if a.lang == "ar" else "English"
    answer = call_llm(
        client, a.model, ANALYZE_SYSTEM,
        ANALYZE_USER.format(facts=json.dumps(facts, ensure_ascii=False, indent=2),
                            laws=format_laws(laws), language=language),
        max_tokens=3000,
    )
    print(answer)

    if a.save:
        report = (f"# Accident analysis\n\n## Description\n{description}\n\n## Extracted facts\n```json\n"
                  f"{json.dumps(facts, ensure_ascii=False, indent=2)}\n```\n\n## Queries\n"
                  + "\n".join(f"- {q}" for q in queries)
                  + f"\n\n## Retrieved sections\n" + ", ".join(p["id"] for p in laws)
                  + f"\n\n## Analysis\n{answer}\n")
        Path(a.save).write_text(report, encoding="utf-8")
        print(f"\nSaved to {a.save}")


if __name__ == "__main__":
    main()
