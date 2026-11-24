"""
Evaluation. `uv run python main.py evaluate`

1. Component A (risk model): the held-out test set from models/model_card.json, and
   the cases in eval/cases.csv, both against the points-grid baseline.
2. End to end, on eval/cases.csv: is the risk class right, is the suitable / not
   suitable verdict right, is an out-of-scope input refused. Compared with the
   baseline (the grid's class, checked with the same rules), with bootstrap 95% CIs.
3. Component B (agent): an LLM judge grades each statement against a rubric, and
   its agreement with eval/human_labels.csv (your own PASS/FAIL labels).
4. Cost and latency per component.

The `expected` column of eval/cases.csv:
    risk=3;suitable=no      the right class and verdict
    refuse                  an out-of-scope input the product must refuse
Writes eval/results.csv, eval/judge.csv and eval/summary.json. Commit them.
"""

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import env
import pipeline
import suitability

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "eval" / "cases.csv"
RESULTS = ROOT / "eval" / "results.csv"
JUDGE_RESULTS = ROOT / "eval" / "judge.csv"
HUMAN_LABELS = ROOT / "eval" / "human_labels.csv"
SUMMARY = ROOT / "eval" / "summary.json"
CARD = ROOT / "models" / "model_card.json"
JUDGE_CACHE = ROOT / "cache" / "judge"
JUDGE_VERSION = "judge-v1"

JUDGE_PROMPT = """You grade a draft MiFID II suitability statement written by an AI for a private-banking advisor.
You get the client's questionnaire, the risk class, the result of the deterministic suitability check, and the statement.

The statement PASSES only if all of these hold:
1. Its verdict (suitable / not suitable) is the same as the check's.
2. If the check found issues, it states every one of them.
3. It names the funds it discusses with their correct SRI, and invents no fund data.
4. It does not promise or predict returns.
5. If the client needs liquidity within 2 years, it says so.
6. It ends with "Draft for advisor review and signature."

Reply with JSON only: {"verdict": "PASS" or "FAIL", "reason": "<one sentence>"}"""


def parse_expected(text: str) -> dict:
    text = (text or "").strip().lower()
    if text.startswith("refuse"):
        return {"refuse": True}
    parts = dict(p.split("=", 1) for p in text.replace(" ", "").split(";") if "=" in p)
    return {
        "refuse": False,
        "risk": int(parts["risk"]) if parts.get("risk", "").isdigit() else None,
        "suitable": {"yes": True, "no": False}.get(parts.get("suitable")),
    }


def is_correct(case: dict, result: dict) -> bool | None:
    """Right class and right verdict; or refused, when it should be."""
    expected = parse_expected(case["expected"])
    a, b = result["outputs"]["risk_model"], result["outputs"]["suitability_agent"]
    if expected["refuse"]:
        return bool(a.get("refused"))
    if a.get("refused") or expected["risk"] is None or expected["suitable"] is None:
        return False if a.get("refused") else None
    return a["risk_class"] == expected["risk"] and b["suitable"] == expected["suitable"]


def baseline_correct(case: dict) -> bool | None:
    """The same question for today's process: the points grid, checked with the same rules."""
    expected = parse_expected(case["expected"])
    try:
        profile = suitability.parse(case["input"])
    except suitability.OutOfScope:
        return expected["refuse"]  # an invalid form is rejected today too
    if expected["refuse"] or expected["risk"] is None:
        return False if expected["refuse"] else None
    grid = suitability.grid_class(profile)
    verdict = suitability.check_suitability(grid, profile["portfolio"])["suitable"]
    return grid == expected["risk"] and verdict == expected["suitable"]


def bootstrap_ci(values, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray([v for v in values if v is not None], float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (n, len(values))).mean(axis=1)
    return round(float(np.quantile(means, 0.025)), 3), round(float(np.quantile(means, 0.975)), 3)


def judge(case: dict, result: dict) -> dict:
    """LLM judge for one statement (cached). None when it cannot run."""
    b = result["outputs"]["suitability_agent"]
    if b.get("refused") or b.get("llm") == "offline-template" or not os.environ.get("GEMINI_API_KEY"):
        return {"verdict": None, "reason": "not judged (refused, or no LLM statement)"}
    profile = suitability.parse(case["input"])
    check = suitability.check_suitability(result["outputs"]["risk_model"]["risk_class"], profile["portfolio"])
    payload = json.dumps({"questionnaire": {k: v for k, v in profile.items() if k != "portfolio"},
                          "portfolio": profile["portfolio"], "check": check, "statement": b["statement"]})
    key = hashlib.sha256((pipeline.GEMINI_MODEL + JUDGE_VERSION + payload).encode()).hexdigest()[:24]
    cached = JUDGE_CACHE / f"{key}.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(api_version="v1beta"))
    response = client.models.generate_content(
        model=pipeline.GEMINI_MODEL, contents=payload,
        config=types.GenerateContentConfig(system_instruction=JUDGE_PROMPT, temperature=0,
                                           response_mime_type="application/json"),
    )
    try:
        verdict = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        verdict = {"verdict": None, "reason": f"unparseable judge output: {str(response.text)[:100]}"}
    JUDGE_CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(verdict), encoding="utf-8")
    return verdict


def cohen_kappa(a: list[str], b: list[str]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    observed = float(np.mean(a == b))
    labels = np.union1d(a, b)
    expected = float(sum(np.mean(a == l) * np.mean(b == l) for l in labels))
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else 1.0


def main() -> None:
    env.load_env()
    with CASES.open(newline="", encoding="utf-8") as f:
        cases = [r for r in csv.DictReader(f) if (r.get("input") or "").strip()]
    if not cases:
        raise SystemExit("eval/cases.csv has no cases yet. See eval/README.md.")
    rows, judged = [], []
    for i, case in enumerate(cases, 1):
        print(f"  {i}/{len(cases)}", end="\r", flush=True)
        started = time.perf_counter()
        result = pipeline.run(case["input"])
        latency = round(time.perf_counter() - started, 3)
        a, b = result["outputs"]["risk_model"], result["outputs"]["suitability_agent"]
        expected = parse_expected(case["expected"])
        verdict = judge(case, result)
        judged.append({"id": case["id"], "judge": verdict.get("verdict"), "reason": verdict.get("reason")})
        rows.append({
            "id": case["id"],
            "kind": case["kind"].strip().lower(),
            "expected": case["expected"],
            "risk_class": a.get("risk_class"),
            "grid_class": a.get("grid_baseline_class"),
            "confidence": a.get("confidence"),
            "suitable": b.get("suitable"),
            "review": result["review"],
            "correct": is_correct(case, result),
            "baseline_correct": baseline_correct(case),
            "risk_correct": None if expected.get("risk") is None or a.get("refused") else a["risk_class"] == expected["risk"],
            "grid_risk_correct": None if expected.get("risk") is None or a.get("refused") else a["grid_baseline_class"] == expected["risk"],
            "judge": verdict.get("verdict"),
            "latency_s": latency,
            "latency_a_s": a.get("latency_s"),
            "latency_b_s": b.get("latency_s"),
            "agent_cached": b.get("cached"),
            "tokens": result["tokens"],
            "cost_eur": result["cost_eur"],
            "outputs": json.dumps(result["outputs"]),
        })
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS, index=False)
    pd.DataFrame(judged).to_csv(JUDGE_RESULTS, index=False)
    print(f"Wrote {RESULTS.relative_to(ROOT)} and {JUDGE_RESULTS.relative_to(ROOT)} ({len(results)} cases).\n")

    summary = {}
    if CARD.is_file():
        card = json.loads(CARD.read_text(encoding="utf-8"))
        summary["component_a_held_out"] = card["held_out_test"] | {"label_source": card["label_source"]}
        print(f"Component A, held-out test set ({card['held_out_test']['n']} clients, {card['label_source']} labels):")
        print(pd.DataFrame({"model": card["held_out_test"]["model"], "grid": card["held_out_test"]["grid_baseline"]}).to_string(), "\n")

    def rate(column: str, frame: pd.DataFrame = results) -> dict:
        values = [v for v in frame[column] if v is not None and not pd.isna(v)]
        return {"n": len(values), "mean": round(float(np.mean(values)), 3) if values else None, "ci95": bootstrap_ci(values)}

    summary["end_to_end"] = {"product": rate("correct"), "baseline": rate("baseline_correct")}
    summary["risk_class_on_cases"] = {"model": rate("risk_correct"), "grid": rate("grid_risk_correct")}
    review_kind = results[results["kind"] == "review"]
    summary["review_recall"] = round(float(review_kind["review"].mean()), 3) if len(review_kind) else None
    print("End to end on eval/cases.csv (mean, bootstrap 95% CI):")
    for name, value in summary["end_to_end"].items():
        print(f"  {name:<9} {value['mean']}  {value['ci95']}  (n={value['n']})")
    print("Risk class only:")
    for name, value in summary["risk_class_on_cases"].items():
        print(f"  {name:<9} {value['mean']}  {value['ci95']}  (n={value['n']})")
    print(f"Share of `review` cases actually sent to review: {summary['review_recall']}\n")

    by_kind = results.groupby("kind").agg(
        cases=("id", "count"), correct=("correct", "mean"), baseline=("baseline_correct", "mean"),
        sent_to_review=("review", "mean"), median_latency_s=("latency_s", "median"), mean_cost_eur=("cost_eur", "mean"),
    )
    print(by_kind.round(3).to_string(), "\n")

    # The judge, and how far to trust it.
    graded = results["judge"].notna().sum()
    summary["judge"] = {"judged": int(graded), "pass_rate": round(float((results["judge"] == "PASS").sum() / graded), 3) if graded else None}
    if HUMAN_LABELS.is_file():
        humans = pd.read_csv(HUMAN_LABELS, dtype=str)
        both = humans.merge(results[["id", "judge"]], on="id").dropna(subset=["label", "judge"])
        if len(both):
            human, machine = both["label"].str.upper().str.strip().tolist(), both["judge"].str.upper().tolist()
            summary["judge"] |= {"agreement_n": len(both), "agreement": round(float(np.mean(np.array(human) == np.array(machine))), 3),
                                 "cohen_kappa": cohen_kappa(human, machine)}
    print(f"LLM judge: {summary['judge']}")

    # Cost and latency. Cached agent calls report the cost and tokens of the original call.
    live = results[~results["agent_cached"].fillna(False).astype(bool)]
    summary["cost_latency"] = {
        "mean_tokens": round(float(results["tokens"].mean()), 1),
        "mean_cost_eur": round(float(results["cost_eur"].mean()), 6),
        "median_latency_a_s": round(float(results["latency_a_s"].median()), 4) if results["latency_a_s"].notna().any() else None,
        "median_latency_b_s_uncached": round(float(live["latency_b_s"].median()), 3) if len(live) and live["latency_b_s"].notna().any() else None,
    }
    print(f"Cost and latency: {summary['cost_latency']}")
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nWrote {SUMMARY.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
