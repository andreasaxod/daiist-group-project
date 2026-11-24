"""
The product: a suitability copilot for private-banking advisors.

    request (questionnaire + proposed portfolio, JSON)
      -> component A, risk model (scikit-learn): risk class 1-5 and its confidence
      -> component B, suitability agent (Gemini + tools): looks up every fund's KID
         data, runs the deterministic suitability check for that class, and drafts
         the suitability statement for the advisor to sign.

The risk class from A is B's input: the agent checks the portfolio against it.

`run()` keeps the template's return format: app.py, evaluate.py, main.py and
check.py read it.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import joblib
import pandas as pd

import suitability
from suitability import OutOfScope

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "risk_model.joblib"
CACHE_DIR = ROOT / "cache" / "agent"

PROMPT_VERSION = "statement-v1"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# TEAM DECISION: below this confidence, the risk class is too uncertain to rely on.
REVIEW_BELOW_CONFIDENCE = 0.6

# TODO: check the current Gemini price for the model you use (USD per million tokens)
# and the exchange rate, and cite them in REPORT.md. The free tier costs nothing,
# but report what the product would cost on the paid tier.
USD_PER_M_INPUT, USD_PER_M_OUTPUT, EUR_PER_USD = 0.30, 2.50, 0.86

MAX_AGENT_STEPS = 6

SYSTEM_PROMPT = """You assist a private-banking advisor in the EU with the MiFID II suitability assessment.
You receive a client's questionnaire answers, the risk class predicted by a model, and a proposed portfolio.

Steps:
1. Call lookup_fund for every fund in the portfolio.
2. Call check_suitability once with the predicted risk class. Its verdict is final: never contradict it.
3. Write the suitability statement for the advisor, in English, 4 to 7 sentences:
   - the client's profile in one sentence (risk class, horizon, objective, capacity to bear losses);
   - whether the portfolio is suitable, and why, naming the funds and their SRI;
   - if it is not suitable: every issue the check found, and what would have to change;
   - mention the liquidity need if the client has one;
   - never promise or predict returns; never invent fund data; no client names;
   - end with: "Draft for advisor review and signature."
Reply with the statement only."""


# --- Component A: the risk model ------------------------------------------------

def _load_model() -> dict:
    if not MODEL_FILE.is_file():
        raise SystemExit("models/risk_model.joblib is missing. Run `uv run python train.py` and commit it.")
    return joblib.load(MODEL_FILE)


_MODEL = _load_model()


def component_a(profile: dict) -> dict:
    """Risk class 1-5 with its probability, plus the points-grid class for comparison."""
    started = time.perf_counter()
    features = pd.DataFrame([suitability.model_features(profile)])[_MODEL["features"]]
    probabilities = _MODEL["model"].predict_proba(features)[0]
    classes = [int(c) for c in _MODEL["model"].classes_]
    best = int(probabilities.argmax())
    risk_class = classes[best]
    return {
        "risk_class": risk_class,
        "profile_name": suitability.RISK_NAMES[risk_class],
        "confidence": round(float(probabilities[best]), 3),
        "probabilities": {str(c): round(float(p), 3) for c, p in zip(classes, probabilities)},
        "grid_baseline_class": suitability.grid_class(profile),
        "latency_s": round(time.perf_counter() - started, 4),
    }


# --- Component B: the suitability agent -------------------------------------------

def _tool_declarations():
    from google.genai import types

    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="lookup_fund",
            description="KID data of one fund: name, asset class, SRI (1-7) and ongoing charges.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"fund_id": types.Schema(type="STRING", description="Fund id as given in the portfolio")},
                required=["fund_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="check_suitability",
            description="Deterministic suitability check of the proposed portfolio against a risk class (1-5).",
            parameters=types.Schema(
                type="OBJECT",
                properties={"risk_class": types.Schema(type="INTEGER", description="Risk class 1-5")},
                required=["risk_class"],
            ),
        ),
    ])]


def _cache_key(profile: dict, a: dict) -> str:
    payload = json.dumps([GEMINI_MODEL, PROMPT_VERSION, profile, a["risk_class"]], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _offline_statement(profile: dict, check: dict, reason: str) -> str:
    verdict = "suitable" if check["suitable"] else "NOT suitable"
    issues = " ".join(check["issues"])
    return (
        f"[No LLM: {reason}] Risk class {check['risk_class']} ({check['profile_name']}), "
        f"horizon {profile['horizon_years']} years, objective {profile['objective']}. "
        f"Weighted SRI {check['weighted_sri']}: the portfolio is {verdict}. {issues} "
        "Draft for advisor review and signature."
    ).replace("  ", " ")


def _run_agent(profile: dict, a: dict) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    tools = {
        "lookup_fund": lambda args: suitability.lookup_fund(args.get("fund_id", "")),
        "check_suitability": lambda args: suitability.check_suitability(args.get("risk_class", a["risk_class"]), profile["portfolio"]),
    }
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=_tool_declarations(),
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    client_view = {k: v for k, v in profile.items() if k != "portfolio"}
    prompt = json.dumps({
        "questionnaire": client_view,
        "predicted_risk_class": a["risk_class"],
        "model_confidence": a["confidence"],
        "proposed_portfolio": profile["portfolio"],
    })
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    tokens_in = tokens_out = 0
    trace, text = [], ""
    for _ in range(MAX_AGENT_STEPS):
        response = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
        usage = response.usage_metadata
        if usage:
            tokens_in += usage.prompt_token_count or 0
            tokens_out += (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        calls = response.function_calls or []
        if not calls:
            text = (response.text or "").strip()
            break
        contents.append(response.candidates[0].content)
        parts = []
        for call in calls:
            args = dict(call.args or {})
            result = tools[call.name](args) if call.name in tools else {"error": f"unknown tool {call.name}"}
            trace.append({"tool": call.name, "args": args})
            parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))
    cost = (tokens_in * USD_PER_M_INPUT + tokens_out * USD_PER_M_OUTPUT) / 1e6 * EUR_PER_USD
    return {"statement": text, "tool_calls": trace, "tokens": tokens_in + tokens_out, "cost_eur": round(cost, 6)}


def component_b(profile: dict, a: dict) -> dict:
    """Suitability verdict (deterministic) and the statement drafted by the agent."""
    started = time.perf_counter()
    check = suitability.check_suitability(a["risk_class"], profile["portfolio"])
    out = {
        "suitable": check["suitable"],
        "issues": check["issues"],
        "weighted_sri": check["weighted_sri"],
        "llm": f"{GEMINI_MODEL}/{PROMPT_VERSION}",
        "cached": False,
        "error": None,
    }
    cache_file = CACHE_DIR / f"{_cache_key(profile, a)}.json"
    if cache_file.is_file():
        agent = json.loads(cache_file.read_text(encoding="utf-8"))
        out["cached"] = True
    elif not os.environ.get("GEMINI_API_KEY", "").strip():
        agent = {"statement": _offline_statement(profile, check, "GEMINI_API_KEY is not set"), "tool_calls": [], "tokens": 0, "cost_eur": 0.0}
        out["llm"], out["error"] = "offline-template", "GEMINI_API_KEY is not set"
    else:
        try:
            agent = _run_agent(profile, a)
            if not agent["statement"]:
                raise RuntimeError("the agent returned no statement")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(agent), encoding="utf-8")
        except Exception as error:  # noqa: BLE001 - the advisor still gets the verdict, and the case goes to review
            reason = f"{type(error).__name__}: {str(error)[:200]}"
            agent = {"statement": _offline_statement(profile, check, "the LLM call failed"), "tool_calls": [], "tokens": 0, "cost_eur": 0.0}
            out["llm"], out["error"] = "offline-template", reason
    out |= agent
    out["checked_by_agent"] = any(t["tool"] == "check_suitability" for t in agent["tool_calls"])
    out["latency_s"] = round(time.perf_counter() - started, 3)
    return out


# --- The whole product ------------------------------------------------------------

def input_features(profile: dict, a: dict) -> dict[str, float]:
    """What the drift check watches: who the clients are, and what the model says about them."""
    return {
        "in_scope": 1.0,
        "age": float(profile["age"]),
        "horizon_years": float(profile["horizon_years"]),
        "log10_wealth": round(math.log10(profile["liquid_wealth_k"]), 3),
        "experience": float(profile["experience"]),
        "n_funds": float(len(profile["portfolio"])),
        "risk_class": float(a["risk_class"]),
        "confidence": a["confidence"],
    }


def summary(profile: dict) -> str:
    """Short and without personal data (there is none in the input, but keep it short)."""
    funds = ", ".join(f"{p['fund_id']} {p['weight_pct']:g}%" for p in profile["portfolio"])
    return (f"age {profile['age']}, horizon {profile['horizon_years']}y, {profile['objective']}, "
            f"liquidity need {profile['needs_liquidity_2y']} | {funds}")[:200]


def run(raw_input: str) -> dict:
    try:
        profile = suitability.parse(raw_input)
    except OutOfScope as reason:
        return {
            "input_summary": "out of scope: " + str(reason)[:150],
            "features": {"in_scope": 0.0},
            "outputs": {
                "risk_model": {"refused": True, "reason": str(reason)},
                "suitability_agent": {"refused": True, "statement": f"I can only assess a client questionnaire with a proposed portfolio. {reason}"},
            },
            "versions": {"risk_model": _MODEL["version"], "suitability_agent": f"{GEMINI_MODEL}/{PROMPT_VERSION}"},
            "review": False,
            "tokens": 0,
            "cost_eur": 0.0,
        }
    a = component_a(profile)
    b = component_b(profile, a)
    review = (
        a["confidence"] < REVIEW_BELOW_CONFIDENCE
        or not b["suitable"]
        or b["error"] is not None
    )
    return {
        "input_summary": summary(profile),
        "features": input_features(profile, a),
        "outputs": {"risk_model": a, "suitability_agent": b},
        "versions": {"risk_model": _MODEL["version"], "suitability_agent": b["llm"]},
        "review": bool(review),
        "tokens": int(b["tokens"]),
        "cost_eur": float(b["cost_eur"]),
    }
