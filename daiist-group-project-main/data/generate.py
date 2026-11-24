"""
Synthetic clients. `uv run python data/generate.py [--force]`

Writes:
- data/clients_to_label.csv: 150 clients for the advisor to label (column
  `advisor_label`, 1-5). Never overwritten unless --force, so labels are safe.
- eval/reference.csv: 40 normal requests (questionnaire + proposed portfolio),
  from the same population, for the drift check.
- eval/shifted.csv: 15 requests from a shifted segment (retirees), for the drift demo.

How the clients are made: each client has a hidden risk tolerance between 0 and 1.
Their answers are drawn from it with noise, so some clients answer
inconsistently (an "aggressive" objective with a 2-year liquidity need), as real
clients do. `draft_label` is the hidden tolerance turned into a class, with two
caps an advisor would apply (short horizon, liquidity need). It is only a
stand-in until the advisor labels are filled in: see train.py.

No personal data: no names, no ids that point to anyone, every value is random.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from suitability import FUNDS, LOSS_REACTIONS, OBJECTIVES  # noqa: E402

LABEL_FILE = ROOT / "data" / "clients_to_label.csv"
REFERENCE_FILE = ROOT / "eval" / "reference.csv"
SHIFTED_FILE = ROOT / "eval" / "shifted.csv"

QUESTION_COLUMNS = [
    "age", "horizon_years", "annual_income_k", "liquid_wealth_k", "experience", "loss_reaction",
    "objective", "max_drawdown_ok_pct", "needs_liquidity_2y", "dependents",
]


def _ordinal(score: np.ndarray, levels: int, rng, noise: float) -> np.ndarray:
    noisy = np.clip(score + rng.normal(0, noise, len(score)), 0, 0.999)
    return (noisy * levels).astype(int)


def make_clients(n: int, rng, segment: str = "private_banking") -> list[dict]:
    if segment == "retirees":
        age = rng.integers(68, 90, n)
    else:
        age = np.clip(rng.normal(50, 12, n), 25, 80).astype(int)
    wealth = np.round(np.exp(rng.normal(np.log(900), 0.8, n)), -1)  # thousands of euros
    income = np.round(np.clip(np.exp(rng.normal(np.log(140), 0.6, n)), 20, 2000))
    if segment == "retirees":
        income = np.round(income * 0.45)

    # The hidden risk tolerance: younger, wealthier clients tolerate more, on average.
    tolerance = np.clip(
        0.55 - 0.012 * (age - 50) + 0.08 * np.log(wealth / 900) + rng.normal(0, 0.18, n), 0.02, 0.98
    )
    horizon = np.clip(np.round(2 + 20 * tolerance * (85 - age) / 40 + rng.normal(0, 3, n)), 1, 40).astype(int)
    experience = _ordinal(tolerance, 4, rng, 0.25)
    reaction = _ordinal(tolerance, 4, rng, 0.2)
    objective = _ordinal(tolerance, 5, rng, 0.2)
    drawdown = np.clip(np.round(5 + 35 * tolerance + rng.normal(0, 6, n)), 0, 60)
    liquidity = rng.random(n) < np.where(age > 65, 0.25, 0.1)
    dependents = rng.integers(0, 4, n)

    clients = []
    for i in range(n):
        label = int(np.clip(np.ceil(tolerance[i] * 5), 1, 5))
        if liquidity[i] or horizon[i] < 3:
            label = min(label, 2)
        elif horizon[i] < 5:
            label = min(label, 3)
        clients.append({
            "age": int(age[i]),
            "horizon_years": int(horizon[i]),
            "annual_income_k": float(income[i]),
            "liquid_wealth_k": float(wealth[i]),
            "experience": int(experience[i]),
            "loss_reaction": LOSS_REACTIONS[reaction[i]],
            "objective": OBJECTIVES[objective[i]],
            "max_drawdown_ok_pct": float(drawdown[i]),
            "needs_liquidity_2y": bool(liquidity[i]),
            "dependents": int(dependents[i]),
            "_draft_label": label,
        })
    return clients


def make_portfolio(target_class: int, rng) -> list[dict]:
    """A proposed portfolio of 2-4 funds, usually close to the client's class, sometimes too risky."""
    aim = target_class + (rng.integers(1, 3) if rng.random() < 0.3 else 0)  # about 30% too risky
    by_sri = {}
    for fund_id, fund in FUNDS.items():
        by_sri.setdefault(fund["sri"], []).append(fund_id)
    max_sri = {1: 2, 2: 3, 3: 4, 4: 5, 5: 7}[min(aim, 5)] + (1 if aim > 5 else 0)
    pool = [f for sri, funds in by_sri.items() if sri <= max_sri for f in funds]
    risky = [f for sri, funds in by_sri.items() if max_sri - 1 <= sri <= max_sri for f in funds]
    n = int(rng.integers(2, 5))
    chosen = list(dict.fromkeys([str(rng.choice(risky))] + [str(f) for f in rng.choice(pool, n - 1)]))
    raw = rng.dirichlet(np.ones(len(chosen))) * 100
    weights = np.maximum(np.round(raw / 5) * 5, 5)
    weights[0] += 100 - weights.sum()
    if weights[0] <= 0:
        weights = np.full(len(chosen), 100 / len(chosen)).round(1)
        weights[0] += round(100 - weights.sum(), 1)
    return [{"fund_id": f, "weight_pct": float(w)} for f, w in zip(chosen, weights)]


def as_request(client: dict, rng) -> str:
    request = {k: client[k] for k in QUESTION_COLUMNS}
    request["portfolio"] = make_portfolio(client["_draft_label"], rng)
    return json.dumps(request)


def write_inputs(path: Path, inputs: list[str]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input"])
        writer.writerows([[i] for i in inputs])
    print(f"Wrote {path.relative_to(ROOT)}: {len(inputs)} inputs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="also overwrite data/clients_to_label.csv (loses labels)")
    args = parser.parse_args()

    if LABEL_FILE.exists() and not args.force:
        print(f"Kept {LABEL_FILE.relative_to(ROOT)} (it may hold advisor labels; --force to overwrite).")
    else:
        clients = make_clients(150, np.random.default_rng(42))
        with LABEL_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["client_id", *QUESTION_COLUMNS, "draft_label", "advisor_label", "advisor_comment"])
            for i, c in enumerate(clients, 1):
                writer.writerow([f"t{i:03d}", *[c[k] for k in QUESTION_COLUMNS], c["_draft_label"], "", ""])
        print(f"Wrote {LABEL_FILE.relative_to(ROOT)}: {len(clients)} clients to label.")

    rng = np.random.default_rng(7)
    write_inputs(REFERENCE_FILE, [as_request(c, rng) for c in make_clients(40, rng)])
    rng = np.random.default_rng(11)
    write_inputs(SHIFTED_FILE, [as_request(c, rng) for c in make_clients(15, rng, segment="retirees")])


if __name__ == "__main__":
    main()
