"""
Domain rules shared by the pipeline, the training script and the data generator:
the client questionnaire, the fund table, the points-grid baseline, and the
deterministic suitability check that the agent calls as a tool.

Everything marked TEAM DECISION is a choice the team must confirm and justify in
REPORT.md (it is a starting point, not a standard).
"""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FUNDS_FILE = ROOT / "data" / "funds.csv"

RISK_NAMES = {1: "Conservative", 2: "Cautious", 3: "Balanced", 4: "Dynamic", 5: "Aggressive"}

LOSS_REACTIONS = ["sell_all", "sell_some", "hold", "buy_more"]
OBJECTIVES = ["preservation", "income", "balanced", "growth", "aggressive_growth"]

# The questionnaire: field -> (type, min, max) or list of allowed values.
QUESTIONNAIRE = {
    "age": (int, 18, 100),
    "horizon_years": (int, 1, 40),
    "annual_income_k": (float, 0, 10_000),
    "liquid_wealth_k": (float, 1, 1_000_000),
    "experience": (int, 0, 3),  # 0 none, 1 basic, 2 informed, 3 expert
    "loss_reaction": LOSS_REACTIONS,  # reaction to a 20% fall in the portfolio
    "objective": OBJECTIVES,
    "max_drawdown_ok_pct": (float, 0, 100),  # largest temporary loss the client accepts
    "needs_liquidity_2y": bool,  # needs a large part of the money within 2 years
    "dependents": (int, 0, 15),
}

# TEAM DECISION: the most risk each profile may take. A fund above `max_fund_sri`
# with a weight of at least MIN_WEIGHT_PCT, or a weighted SRI above `max_avg_sri`,
# makes the portfolio unsuitable.
LIMITS = {
    1: {"max_fund_sri": 2, "max_avg_sri": 2.0},
    2: {"max_fund_sri": 3, "max_avg_sri": 2.5},
    3: {"max_fund_sri": 4, "max_avg_sri": 3.5},
    4: {"max_fund_sri": 5, "max_avg_sri": 4.5},
    5: {"max_fund_sri": 7, "max_avg_sri": 6.0},
}
MIN_WEIGHT_PCT = 5.0


class OutOfScope(ValueError):
    """The input is not a client questionnaire with a portfolio."""


# --- Input ----------------------------------------------------------------------

def parse(raw_input) -> dict:
    """The request as a dict {questionnaire fields..., "portfolio": [{"fund_id", "weight_pct"}]}.

    Accepts a JSON string or the path of a .json file. Raises OutOfScope with a
    short reason the app can show.
    """
    text = str(raw_input).strip()
    path = Path(text)
    if len(text) < 300 and text.endswith(".json") and path.is_file():
        text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise OutOfScope("The input is not a client questionnaire (expected the JSON the form builds).") from None
    if not isinstance(data, dict):
        raise OutOfScope("The input is not a client questionnaire.")

    profile = {}
    for field, rule in QUESTIONNAIRE.items():
        if field not in data:
            raise OutOfScope(f"Missing questionnaire answer: {field}.")
        value = data[field]
        if rule is bool:
            if not isinstance(value, bool):
                raise OutOfScope(f"{field} must be true or false.")
        elif isinstance(rule, list):
            if value not in rule:
                raise OutOfScope(f"{field} must be one of {', '.join(rule)}.")
        else:
            kind, low, high = rule
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OutOfScope(f"{field} must be a number.")
            if not low <= value <= high:
                raise OutOfScope(f"{field} must be between {low} and {high}.")
            value = kind(value)
        profile[field] = value

    portfolio = data.get("portfolio")
    if not isinstance(portfolio, list) or not portfolio:
        raise OutOfScope("No proposed portfolio: add at least one fund with its weight.")
    lines = []
    for line in portfolio:
        if not isinstance(line, dict) or "fund_id" not in line or "weight_pct" not in line:
            raise OutOfScope("Each portfolio line needs fund_id and weight_pct.")
        weight = line["weight_pct"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise OutOfScope("Portfolio weights must be positive numbers.")
        lines.append({"fund_id": str(line["fund_id"]).strip().upper(), "weight_pct": float(weight)})
    total = sum(line["weight_pct"] for line in lines)
    if abs(total - 100) > 1:
        raise OutOfScope(f"Portfolio weights add up to {total:g}%, not 100%.")
    profile["portfolio"] = lines
    return profile


def model_features(profile: dict) -> dict[str, float]:
    """The numbers the risk model reads. Same order and names at training and serving time."""
    return {
        "age": float(profile["age"]),
        "horizon_years": float(profile["horizon_years"]),
        "log10_income": math.log10(1 + profile["annual_income_k"]),
        "log10_wealth": math.log10(profile["liquid_wealth_k"]),
        "experience": float(profile["experience"]),
        "loss_reaction": float(LOSS_REACTIONS.index(profile["loss_reaction"])),
        "objective": float(OBJECTIVES.index(profile["objective"])),
        "max_drawdown_ok_pct": float(profile["max_drawdown_ok_pct"]),
        "needs_liquidity_2y": float(profile["needs_liquidity_2y"]),
        "dependents": float(profile["dependents"]),
    }


FEATURE_NAMES = list(model_features({
    "age": 40, "horizon_years": 10, "annual_income_k": 100, "liquid_wealth_k": 500, "experience": 1,
    "loss_reaction": "hold", "objective": "balanced", "max_drawdown_ok_pct": 15,
    "needs_liquidity_2y": False, "dependents": 0,
}))


# --- Baseline: the points grid ----------------------------------------------------

def grid_points(profile: dict) -> int:
    """TEAM DECISION: a points grid like the questionnaires banks score by hand today.

    Six answers, 1 to 4 points each (6-24 in total). It ignores liquidity needs and
    inconsistent answers, which is exactly what an advisor corrects for.
    """
    age = profile["age"]
    horizon = profile["horizon_years"]
    drawdown = profile["max_drawdown_ok_pct"]
    points = 4 if age < 35 else 3 if age < 50 else 2 if age < 65 else 1
    points += 4 if horizon >= 10 else 3 if horizon >= 5 else 2 if horizon >= 3 else 1
    points += profile["experience"] + 1
    points += LOSS_REACTIONS.index(profile["loss_reaction"]) + 1
    points += [1, 1, 2, 3, 4][OBJECTIVES.index(profile["objective"])]
    points += 1 if drawdown < 10 else 2 if drawdown < 20 else 3 if drawdown < 30 else 4
    return points


def grid_class(profile: dict) -> int:
    points = grid_points(profile)
    return 1 if points <= 9 else 2 if points <= 12 else 3 if points <= 16 else 4 if points <= 20 else 5


# --- Funds and the suitability check (the agent's tools) -----------------------------

def load_funds() -> dict[str, dict]:
    with FUNDS_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        r["fund_id"].strip().upper(): r | {"sri": int(r["sri"]), "ongoing_charges_pct": float(r["ongoing_charges_pct"])}
        for r in rows
    }


FUNDS = load_funds()


def lookup_fund(fund_id: str) -> dict:
    """One fund's KID data: name, asset class, SRI (1-7) and ongoing charges."""
    fund = FUNDS.get(str(fund_id).strip().upper())
    if fund is None:
        return {"fund_id": fund_id, "found": False}
    return {"found": True} | {k: fund[k] for k in ("fund_id", "name", "isin", "asset_class", "sri", "ongoing_charges_pct")}


def check_suitability(risk_class: int, portfolio: list[dict]) -> dict:
    """Deterministic MiFID-style check of a portfolio against a risk class.

    The LLM never decides suitability on its own: it calls this and explains it.
    """
    risk_class = int(risk_class)
    limits = LIMITS[risk_class]
    issues, unknown, weighted, known_weight = [], [], 0.0, 0.0
    for line in portfolio:
        fund = FUNDS.get(line["fund_id"])
        if fund is None:
            unknown.append(line["fund_id"])
            continue
        weighted += fund["sri"] * line["weight_pct"]
        known_weight += line["weight_pct"]
        if fund["sri"] > limits["max_fund_sri"] and line["weight_pct"] >= MIN_WEIGHT_PCT:
            issues.append(
                f"{line['fund_id']} has SRI {fund['sri']}, above the maximum of {limits['max_fund_sri']} "
                f"for a {RISK_NAMES[risk_class]} profile ({line['weight_pct']:g}% of the portfolio)."
            )
    avg_sri = round(weighted / known_weight, 2) if known_weight else None
    if avg_sri is not None and avg_sri > limits["max_avg_sri"]:
        issues.append(f"The weighted SRI of the portfolio is {avg_sri}, above {limits['max_avg_sri']} for this profile.")
    if unknown:
        issues.append(f"Unknown fund(s), no KID data: {', '.join(unknown)}.")
    return {
        "risk_class": risk_class,
        "profile_name": RISK_NAMES[risk_class],
        "limits": limits,
        "weighted_sri": avg_sri,
        "suitable": not issues,
        "issues": issues,
        "unknown_funds": unknown,
    }
