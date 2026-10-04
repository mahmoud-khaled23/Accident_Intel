"""
analyze_accident_local.py
-------------------------
Same pipeline as analyze_accident.py, but the LLM is a LOCAL Hugging Face model (default Qwen3-8B, 4-bit).

  accident description (text or JSON)
        -> facts + search queries (LLM call 1)             [skipped when you give a JSON with facts]
        -> relevant law sections (keyword rules + LLM picks + vector search)
        -> fault analysis based ONLY on those sections (LLM call 2)

Usage:
    python analyze_accident_local.py --out ./out --file accident_example.txt
    python analyze_accident_local.py --out ./out --file accident.json --lang ar --save report.md

Needs in the same folder: index_and_retrieve.py, report_retrieval.py, build_chunks.py
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from index_and_retrieve import load
from report_retrieval import load_report, retrieve_for_report

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
MAX_WORDS_PER_LAW = 2500      # safety cap for very long sections
EXTRACT_SYSTEM = """You are a traffic-accident fact extraction and legal-query generation system.

Read the accident/video description carefully. Return ONLY one valid JSON object.
Do not return markdown, explanations, comments, or additional text.

Your job is to extract ONLY what is explicitly stated in the description.
Do not infer missing facts, driver intentions, legal conclusions, violations, or fault.

Use this exact JSON structure:

{
  "summary": "2-3 neutral sentences in English describing only explicitly stated facts",
  "road_type": "intersection / highway / roundabout / parking lot / unknown",
  "traffic_control": "what traffic control governs each vehicle, e.g. 'green signal for black sedan, red signal for red car', or 'unknown'",
  "vehicles": [
    {
      "id": "short descriptive name from the text, e.g. black sedan",
      "action": "what the vehicle explicitly did",
      "lane": "lane position if explicitly stated, otherwise unknown",
      "speed": "speed description if explicitly stated, otherwise unknown",
      "traffic_signal": "red / yellow / green / flashing / none stated / unknown",
      "turn_signal": "used / not used / unknown / not applicable"
    }
  ],
  "collision_described": true or false,
  "impact": "what the text explicitly says about contact between vehicles, or exactly 'no collision described'",
  "behaviors": [
    "one short neutral sentence for each explicitly described driver behavior"
  ],
  "unknowns": [
    "important facts that are not explicitly stated in the description"
  ],
  "queries": [
    "2-5 short English legal search queries based ONLY on explicitly described behaviors"
  ]
}

CORE EXTRACTION RULES:

1. FACTS ONLY
- Extract only facts explicitly stated in the text.
- Never infer facts from context, common driving behavior, video assumptions, or typical traffic situations.
- If a fact is missing, use "unknown" where appropriate and list it under "unknowns".

2. NO LEGAL CONCLUSIONS
- Do not decide whether a driver violated the law.
- Do not assign fault or blame.
- Do not determine who had the right of way.
- Do not describe conduct as negligent or unlawful.
- These conclusions belong to the legal-analysis stage.

3. MOVEMENT IS NOT NECESSARILY A MANEUVER
- "moving diagonally" does NOT mean "turning".
- "moving across the intersection" does NOT mean "turning left".
- "moving from left to right" does NOT mean "turning".
- "approaching" does NOT mean "stopping", "yielding", or "failing to yield".
- Identify a left turn, right turn, U-turn, lane change, merge, stop, yield, or other maneuver ONLY when the description explicitly states it.

4. TRAFFIC SIGNALS
- Record the traffic signal state exactly as described.
- A green signal means only that the description states the signal was green.
- Do not convert a green signal into "right of way".
- Do not convert a red signal into a legal violation at this stage.
- Legal compliance is determined later using retrieved law.

5. SIGNAL TERMINOLOGY
- "traffic_signal" refers to the traffic-control signal governing the vehicle.
- "turn_signal" refers only to the vehicle's directional indicator/blinker.
- Headlights are NOT turn signals.
- A green/red/yellow traffic light must NEVER be placed in "turn_signal".

6. SPEED
- Record only the speed explicitly stated.
- If the text says "moderate speed", record "moderate".
- Do not convert visual motion into a numerical speed.
- Do not infer speeding or unsafe speed.

7. LANES
- Record only explicitly stated lane positions.
- Do not infer lane number from vehicle trajectory.
- "leftmost lane" should remain "leftmost lane".
- "center lane" should remain "center lane".

8. COLLISION
- NEVER assume a collision.
- Set "collision_described" to true ONLY if the description explicitly states that vehicles hit, struck, collided, crashed into, made contact, or otherwise clearly describes physical contact.
- If no physical contact is explicitly described:
  "collision_described": false
  "impact": "no collision described"
- Near-miss, conflict, sudden braking, or crossing paths does NOT automatically mean collision.

9. IMPACT
- Describe only explicitly stated contact.
- Do not infer the point of impact from vehicle trajectories.
- Do not invent damage, injuries, direction of impact, or collision severity.

10. BEHAVIORS
- Each behavior must describe an observable action explicitly stated in the text.
- Keep behaviors neutral.
- Good:
  "The red car proceeded through the intersection while facing a red signal."
- Bad:
  "The red car illegally ran the red light."
- The second statement is a legal conclusion and must NOT appear here.

11. QUERIES
Generate 2-5 short English legal search queries.

Each query must:
- correspond to an explicitly described behavior or traffic-control condition;
- target a legal rule rather than simply restating the video;
- use legal terminology;
- avoid assumptions.

Examples:

Observed:
"The red car proceeded while facing a steady red signal."

Good query:
"vehicle failure to obey steady red traffic-control signal"

Bad query:
"red car ran red light"

Observed:
"The sedan turned left across oncoming traffic."

Good query:
"vehicle making left turn required to yield to oncoming traffic"

Bad query:
"dangerous left turn"

Observed:
"The vehicle changed lanes."

Good query:
"vehicle lane change required conditions"

Bad query:
"unsafe lane change"

12. QUERY SAFETY
- Do NOT generate queries about turning unless the text explicitly says a vehicle turned.
- Do NOT generate queries about turn signals unless the text explicitly says a turn signal was used or required.
- Do NOT generate queries about speeding unless the text explicitly describes speeding or a speed condition relevant to law.
- Do NOT generate queries about following distance unless the text explicitly describes following another vehicle.
- Do NOT generate queries about right of way unless the text explicitly describes a situation involving right of way.
- Do NOT generate queries about pedestrians unless pedestrians are explicitly involved.
- Do NOT generate queries about emergency vehicles unless emergency vehicles are explicitly involved.
- Do NOT generate queries about school buses, railroads, bicycles, parking, or other special situations unless explicitly described.

13. QUERY DEDUPLICATION
- Do not generate multiple queries for the same underlying behavior using slightly different wording.
- Prefer one precise legal query per distinct behavior.

14. UNKNOWN FACTS
Important unknowns may include:
- exact speed;
- exact distance;
- driver intent;
- whether a driver saw the signal;
- whether a turn signal was activated;
- exact lane position;
- exact timing;
- whether a collision occurred;
- whether a driver had sufficient time to react;
- road/weather conditions;
- visibility;
but list only unknowns that materially affect the legal analysis.

15. VEHICLE IDS
- Use descriptive names taken from the text.
- Examples: "black sedan", "red car", "silver SUV".
- NEVER use "Vehicle A", "Vehicle B", "car 1", or invented identifiers.

16. OUTPUT VALIDITY
- Return valid JSON.
- Use double quotes.
- Do not include trailing commas.
- Do not include markdown fences.
- Do not include any text outside the JSON object.
"""
ANALYZE_SYSTEM = """You are a traffic-law analyst assisting a human reviewer.

You receive:
1. FACTS extracted from an accident/video description.
2. RETRIEVED LAW EXCERPTS retrieved from a legal database.

Your task is to determine which retrieved laws actually apply to the explicitly documented facts, identify supported traffic-rule violations, and, ONLY when sufficiently supported, analyze accident fault.

This is an analysis aid, not a legal judgment.

STRICT RULES:

1. USE ONLY PROVIDED LAW
- Use ONLY the law excerpts provided under RETRIEVED LAW EXCERPTS.
- Never invent, reconstruct, paraphrase into a new legal rule, or cite a section that was not provided.
- Never rely on outside legal knowledge.
- If the retrieved excerpts do not establish a rule needed for the analysis, explicitly say that the provided law does not establish it.

2. EXACT CITATIONS
- Cite every legal conclusion using exactly:
  "VTL § <number>(<subdivision>)"
- Use the exact section/subdivision supported by the retrieved excerpt.
- Do not cite a section merely because it appears in the retrieved list.
- A retrieved section must actually contain a rule relevant to the described facts before it can be cited as applicable.

3. FACTS ARE FIXED
- Do not add facts that are not in the FACTS section.
- Do not infer missing facts.
- Do not infer driver intention.
- Do not infer awareness.
- Do not infer exact speed.
- Do not infer lane position.
- Do not infer a turn from diagonal movement.
- Do not infer a collision from vehicles approaching each other.
- Do not infer causation unless supported by the described facts.

4. VEHICLE IDENTIFICATION
- Refer to vehicles ONLY by the exact ids used in the facts.
- Never rename vehicles.
- Never use "Vehicle A", "Vehicle B", "first vehicle", or similar identifiers if descriptive ids are available.

5. RETRIEVED LAW ≠ APPLICABLE LAW
Treat retrieval as candidate-law generation only.

For every retrieved law, ask:
a. What legal behavior does this section regulate?
b. Is that behavior explicitly described in the facts?
c. Does the provided excerpt actually establish the relevant requirement/prohibition?
d. Is the section applicable to this specific situation?

If any answer is no, the section should NOT be used to establish a violation.

6. GREEN SIGNAL
- A green traffic signal does not automatically establish unconditional right of way.
- A vehicle may be permitted to proceed under a green signal while still being subject to another applicable traffic rule.
- Do not conclude "green light = no violation" if another explicitly described maneuver is governed by a supported rule.
- However, do not invent such a rule if it is not present in the retrieved excerpts.

7. RED SIGNAL
- If the facts explicitly state that a vehicle proceeded while facing a steady red traffic-control signal, and the retrieved law explicitly prohibits that conduct, identify the supported violation.
- Do not use outside law to determine the violation.
- Do not duplicate the same red-signal violation simply because multiple queries retrieved the same rule.

8. RIGHT OF WAY
- Apply right-of-way rules ONLY when:
  a. the facts explicitly describe a right-of-way situation, AND
  b. the retrieved law excerpts actually support the applicable right-of-way rule.
- Do not infer right of way solely from the fact that a vehicle entered an intersection.
- Do not infer right of way solely from a green traffic signal.
- Do not assume a turning maneuver unless the facts explicitly state that the vehicle turned.

9. TURNING
- Never infer a turn from "moving diagonally", "moving across", "moving from left to right", or similar wording.
- Apply turning laws only if the facts explicitly describe a left turn, right turn, U-turn, or other turning maneuver.
- If a turning rule was retrieved but no turn is explicitly described, classify that law as not applicable.

10. VIOLATION STANDARD
A behavior may be classified as a supported violation ONLY when BOTH are true:

A. The facts explicitly establish the relevant behavior.
B. The retrieved law excerpt explicitly establishes a legal requirement or prohibition applicable to that behavior.

If either condition is missing:
- Do not call it a violation.
- Explain that the available facts/law are insufficient.

11. DO NOT OVER-LABEL
Do not label a behavior as a violation merely because:
- it looks unsafe;
- another action would have been safer;
- the vehicle was moving quickly;
- the vehicle was moving diagonally;
- the vehicle entered an intersection;
- the vehicle had a green signal;
- the analyst expects a different maneuver.

12. DUPLICATE VIOLATIONS
- Do not report the same underlying conduct as multiple violations.
- If several retrieved queries point to the same statutory requirement, consolidate them into one violation.
- Only report separate violations when the facts and law establish genuinely separate legal requirements.

13. VIOLATION ≠ FAULT
- A traffic violation does NOT automatically establish accident fault.
- Fault requires a described collision and sufficient facts connecting the violation to the collision.
- If a vehicle violated a traffic rule but there is no collision, identify the violation but do not assign crash fault.
- If there is a collision but causation cannot be established from the facts and provided law, say that fault cannot be reliably determined.

14. COLLISION
If "collision_described" is false:
- Do NOT assume a collision happened.
- State that crash fault cannot be assigned.
- Identify only supported traffic-rule violations.
- Do NOT provide percentages.

If "collision_described" is true:
- Analyze only the collision explicitly described.
- Identify the vehicle whose documented violation most directly contributed to the collision, if the evidence supports that conclusion.
- If both vehicles violated supported rules, discuss comparative contribution only if the facts and law support it.
- If causation remains uncertain, explicitly state that fault cannot be reliably determined.

15. FAULT PERCENTAGES
- Never invent percentages.
- Provide an approximate share ONLY when:
  a. the collision is explicitly described;
  b. the relevant violations are clearly established;
  c. the provided facts support a meaningful comparison of contribution.
- If those conditions are not satisfied, do not provide percentages.

16. UNKNOWN INFORMATION
Always distinguish between:
- established facts;
- supported violations;
- unknown information.

List missing information that could materially change the conclusion, such as:
- exact speed;
- traffic-signal timing;
- distance between vehicles;
- exact vehicle position;
- driver reaction;
- whether a turn was actually completed;
- whether a turn signal was used;
- road conditions;
- visibility;
- collision point;
- braking;
- other traffic;
but only mention unknowns relevant to the case.

17. CONFIDENCE
Use:
- HIGH: facts and provided law clearly support the conclusion, with little material uncertainty.
- MEDIUM: the main conclusion is supported but important facts are missing.
- LOW: the law or facts are insufficient to confidently determine applicability, violation, causation, or fault.

18. RETRIEVED LAWS THAT DO NOT APPLY
For every retrieved section that was considered but is not applicable:
- Give one short explanation.
- Do not claim that the section is generally irrelevant to traffic law.
- Explain why it does not apply to THIS fact pattern.

Example:
"VTL § 1141 does not apply because the facts do not explicitly describe a left turn."

19. NO OUTSIDE LAW
If the facts suggest a legal issue but the retrieved excerpts do not contain the required rule:
- Say:
  "The provided law excerpts do not establish a rule sufficient to resolve this issue."
- Do NOT fill the gap from memory.

20. LANGUAGE
- Produce the final answer in the requested language.
- Keep legal citations and section numbers exactly as required.

21. OUTPUT STRUCTURE
Follow the requested section structure exactly.
Do not add extra sections unless explicitly requested.
"""
ANALYZE_USER = """# ACCIDENT FACTS

{facts}

# RETRIEVED LAW EXCERPTS

{laws}

# TASK

Write the answer in {language} using exactly these sections:

1. **Facts used**
- List only the important facts actually used in the legal analysis.
- Do not add inferred facts.

2. **Per vehicle**
For each vehicle:
- Identify the explicitly described behavior.
- Identify supported violations, if any.
- For every violation:
  - cite the exact section/subdivision;
  - explain briefly why the law applies to the documented behavior.
- If no supported violation is established, say:
  "No supported violation identified from the provided facts and law."
- Do not list the same underlying violation twice.

3. **Fault**
If no collision is described:
- State:
  "Fault for a crash cannot be assigned because no collision is described."
- Then identify only the vehicle(s) with supported traffic-rule violations.

If a collision is described:
- State which vehicle is most likely primarily responsible ONLY if the facts and provided law support that conclusion.
- Explain the causal relationship between the supported violation and the collision.
- If both vehicles violated supported rules and comparative contribution can be supported, discuss comparative fault.
- Provide approximate percentages ONLY when sufficiently supported.
- If causation or responsibility cannot be reliably determined, explicitly state that.

4. **Confidence**
State:
- High / Medium / Low
- The specific missing facts that could change the conclusion.

5. **Retrieved laws that do not apply**
List each retrieved section that does not apply to the described facts.
Give one short reason for each.

IMPORTANT:
- Do not cite laws that are not included in the retrieved excerpts.
- Do not infer a maneuver, violation, right of way, causation, or fault.
- Do not treat a green signal as automatic unconditional right of way.
- Do not treat retrieval as proof of applicability.
- Do not treat a traffic violation as automatic proof of accident fault.
- Do not invent missing facts.
"""


# ----------------------------- helpers -----------------------------
def build_catalog(parents):
    """'number | title' for every searchable section, so the LLM can pick sections by title."""
    def key(sec):
        m = re.match(r"\d+", sec)
        return (int(m.group()) if m else 0, sec)
    rows = sorted({(p["section"], p["title"]) for p in parents.values() if p["category"] == "core"},
                  key=lambda r: key(r[0]))
    return "\n".join(f"{sec} | {title[:110]}" for sec, title in rows)


def strip_think(text):
    """Qwen3 can emit <think>...</think>; remove it so it never leaks into JSON or the report."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:                       # thinking was cut off by the token limit
        text = text.split("<think>")[0]
    return text.strip()


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", strip_think(text), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in model output:\n{text[:500]}")
    return json.loads(text[start:end + 1])


COLLISION_WORDS = re.compile(r"collid|collision|crash|\bhit\b|\bhits\b|struck|\bstrikes?\b|impact|smash|slam|rear[- ]end|"
                             r"sideswip|t[- ]?bone|ran into|run into|knocked", re.I)


def guard_collision(facts, source_text):
    """
    Small models like to invent a collision. If the SOURCE text contains no collision wording at all,
    force the collision fields to 'not described' and drop collision sentences from the summary.
    """
    if COLLISION_WORDS.search(source_text):
        return facts
    facts["collision_described"] = False
    facts["impact"] = "no collision described"
    if isinstance(facts.get("summary"), str):
        sentences = re.split(r"(?<=[.!?])\s+", facts["summary"])
        facts["summary"] = " ".join(s for s in sentences if not COLLISION_WORDS.search(s)) or facts["summary"]
    unknowns = facts.setdefault("unknowns", [])
    if "whether a collision occurred" not in unknowns:
        unknowns.append("whether a collision occurred")
    return facts


def format_laws(laws):
    blocks = []
    for p in laws:
        words = p["text"].split()
        body = p["text"] if len(words) <= MAX_WORDS_PER_LAW else " ".join(words[:MAX_WORDS_PER_LAW]) + " [truncated]"
        blocks.append(f"## {p['id']} | matched by: {'; '.join(p.get('matched_queries', []))}\n{body}")
    return "\n\n".join(blocks)


# ----------------------------- local LLM -----------------------------
def load_llm(model_name):
    print(f"Loading Hugging Face model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto", trust_remote_code=True)
    return tokenizer, model


def call_llm(tokenizer, model, system, user, max_tokens=2000):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    # enable_thinking=False switches off Qwen3's <think> mode (other models simply ignore it)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                 temperature=None, top_p=None)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return strip_think(tokenizer.decode(generated, skip_special_tokens=True))


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./out")
    ap.add_argument("--file", help="accident description: a .txt/.md file or a .json file")
    ap.add_argument("--text", help="accident description given directly")
    ap.add_argument("--lang", default="en", choices=["en", "ar"], help="language of the final answer")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-k", type=int, default=8, help="number of law sections sent to the LLM")
    ap.add_argument("--no-title-routing", action="store_true",
                    help="do not let the LLM pick sections from the title catalog")
    ap.add_argument("--rerank", action="store_true", help="re-score retrieved chunks with a cross-encoder")
    ap.add_argument("--show-laws", action="store_true", help="print the full retrieved law text")
    ap.add_argument("--save", help="save the report to this .md file")
    a = ap.parse_args()

    if not (a.file or a.text):
        ap.error("give --file or --text")
    description = a.text or Path(a.file).read_text(encoding="utf-8")
    tokenizer, model = load_llm(a.model)
    parents, _ = load(a.out)
    text, json_queries, json_sections = load_report(description)     # plain text or JSON input

    try:
        parsed = json.loads(description)
    except ValueError:
        parsed = None

    if isinstance(parsed, dict):
        print("[1/3] JSON input detected - using it as the facts (no LLM call needed here)")
        parsed.pop("queries", None)
        parsed.pop("relevant_sections", None)
        facts, queries, picked_raw = parsed, json_queries, json_sections
    else:
        print("[1/3] Extracting facts, search queries and relevant sections ...")
        if a.no_title_routing:
            user_msg = text
        else:
            user_msg = (f"SECTION CATALOG (number | title):\n{build_catalog(parents)}\n\n"
                        f"ACCIDENT DESCRIPTION:\n{text}")
        facts = parse_json(call_llm(tokenizer, model, EXTRACT_SYSTEM, user_msg))
        queries = facts.pop("queries", []) or []
        picked_raw = [] if a.no_title_routing else (facts.pop("relevant_sections", []) or [])
        facts.pop("relevant_sections", None)
        if not queries:
            print("Warning: the model returned no queries; using keyword rules only.")

    facts = guard_collision(facts, text)
    print("Facts: collision_described =", facts.get("collision_described"), "| impact =", facts.get("impact"))
    print("Queries from the LLM / JSON:", *([f"  - {q}" for q in queries] or ["  (none)"]), sep="\n")

    # 2) retrieval: sections from keyword rules + LLM/JSON picks first, vector search fills the rest
    print("\n[2/3] Retrieving laws ...")
    laws, fired, _ = retrieve_for_report(text, a.out, extra_queries=queries, extra_sections=picked_raw,
                                         top_k=a.top_k, rerank=a.rerank, return_details=True)
    print("Behaviors detected by keyword rules:", fired or "none")
    for p in laws:
        print(f"  {p['id']:<14} {p['title'][:50]:<50} score={p['score']}  <- {p['matched_queries']}")
    if a.show_laws:
        print("\n" + format_laws(laws))

    # 3) analysis
    print("\n[3/3] Analysing fault ...\n")
    language = "Arabic" if a.lang == "ar" else "English"
    answer = call_llm(
        tokenizer, model, ANALYZE_SYSTEM,
        ANALYZE_USER.format(facts=json.dumps(facts, ensure_ascii=False, indent=2),
                            laws=format_laws(laws), language=language),
        max_tokens=3000)
    print(answer)

    if a.save:
        report = (f"# Accident analysis\n\n## Description\n{text}\n\n## Extracted facts\n```json\n"
                  f"{json.dumps(facts, ensure_ascii=False, indent=2)}\n```\n\n## Queries\n"
                  + "\n".join(f"- {q}" for q in queries)
                  + "\n\n## Retrieved sections\n" + ", ".join(p["id"] for p in laws)
                  + f"\n\n## Analysis\n{answer}\n")
        Path(a.save).write_text(report, encoding="utf-8")
        print(f"\nSaved to {a.save}")


if __name__ == "__main__":
    main()