"""
Bonus: a Thompson-sampling bandit that picks the style of the suitability statement.

The loop:  the app picks a style -> the advisor clicks 👍/👎 -> the feedback is
logged -> the next request recomputes each style's posterior from the logs ->
the choice of style changes. Nothing else changes: the risk class, the verdict
and the issues come from the model and the rules, whatever the style.

Arms   : three statement styles (STYLES below), i.e. three prompt variants.
Reward : 👍 = 1, 👎 = 0, on in-scope requests whose statement came from Gemini.
         No click = no reward (not a 0). The reward arrives when the advisor
         clicks, usually seconds to minutes after the statement.
Policy : Thompson sampling with a Beta(1, 1) prior per style, plus a warm-up
         from a simulator (monitoring/bandit_prior.json), counted at half weight
         so the real feedback from the logs soon outweighs it.

    uv run python bandit.py simulate   # writes the simulated warm-up (say so in REPORT.md)
    uv run python bandit.py status     # posterior per style: simulated part, real part
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRIOR_FILE = ROOT / "monitoring" / "bandit_prior.json"

# The arms. "standard" is the prompt the evaluation uses; the other two add an instruction.
STYLES = {
    "standard": "",
    "concise": (
        "Style: concise. At most 4 short sentences. Name only the funds that cause an issue, "
        "with their SRI; if the portfolio is suitable, say so in one sentence with the weighted SRI."
    ),
    "regulatory": (
        "Style: regulatory. Write it as a MiFID II suitability report: start with 'Suitability assessment:', "
        "state the client's risk profile, knowledge and experience, capacity for loss and objectives, then the "
        "verdict with each issue, then the action required. Formal wording."
    ),
}

PRIOR_WEIGHT = 0.5  # one simulated vote counts as half a real one
CACHE_SECONDS = 30  # recompute the posterior from the logs at most this often
_cache = {"at": 0.0, "counts": None}


# --- Rewards from the logs ---------------------------------------------------------

def real_counts(logs: pd.DataFrame) -> dict[str, dict[str, int]]:
    """👍 / 👎 per style, from the app's own logs."""
    counts = {s: {"up": 0, "down": 0, "shown": 0} for s in STYLES}
    if logs is None or logs.empty or "outputs" not in logs:
        return counts
    for outputs, feedback in zip(logs["outputs"], logs.get("feedback", pd.Series([None] * len(logs)))):
        agent = (outputs or {}).get("suitability_agent", {}) if isinstance(outputs, dict) else {}
        style = agent.get("style")
        if style not in counts or agent.get("refused") or agent.get("llm") == "offline-template":
            continue  # no reward for refusals or template statements: they say nothing about the style
        counts[style]["shown"] += 1
        if feedback is not None and not pd.isna(feedback):
            counts[style]["up" if feedback > 0 else "down"] += 1
    return counts


def prior_counts() -> dict[str, dict[str, float]]:
    """The simulated warm-up, already weighted."""
    if not PRIOR_FILE.is_file():
        return {s: {"up": 0.0, "down": 0.0} for s in STYLES}
    raw = json.loads(PRIOR_FILE.read_text(encoding="utf-8"))["counts"]
    return {s: {k: PRIOR_WEIGHT * raw.get(s, {}).get(k, 0) for k in ("up", "down")} for s in STYLES}


def posterior(logs: pd.DataFrame | None = None) -> dict[str, tuple[float, float]]:
    """Beta(alpha, beta) per style = Beta(1, 1) + simulated warm-up + real feedback."""
    if logs is None:
        if time.time() - _cache["at"] > CACHE_SECONDS or _cache["counts"] is None:
            import logbook

            _cache["counts"], _cache["at"] = real_counts(logbook.load()), time.time()
        real = _cache["counts"]
    else:
        real = real_counts(logs)
    prior = prior_counts()
    return {s: (1 + prior[s]["up"] + real[s]["up"], 1 + prior[s]["down"] + real[s]["down"]) for s in STYLES}


# --- The policy --------------------------------------------------------------------

def choose(rng: np.random.Generator | None = None) -> dict:
    """Thompson sampling: draw a 👍 rate for each style from its posterior, take the highest."""
    rng = rng or np.random.default_rng()
    post = posterior()
    draws = {s: float(rng.beta(a, b)) for s, (a, b) in post.items()}
    style = max(draws, key=draws.get)
    return {"style": style, "draws": {s: round(d, 3) for s, d in draws.items()}}


def forget_cache() -> None:
    """Called after new feedback, so the next request sees it."""
    _cache["at"] = 0.0


def table(logs: pd.DataFrame) -> pd.DataFrame:
    """The Monitoring tab's view: what the bandit knows and how often it picks each style now."""
    real, prior = real_counts(logs), prior_counts()
    post = posterior(logs)
    rng = np.random.default_rng(0)
    draws = np.column_stack([rng.beta(a, b, 4000) for a, b in post.values()])
    picked = np.bincount(draws.argmax(axis=1), minlength=len(STYLES)) / len(draws)
    rows = []
    for i, style in enumerate(STYLES):
        a, b = post[style]
        rows.append({
            "style": style,
            "simulated 👍/👎 (x0.5)": f"{prior[style]['up']:g} / {prior[style]['down']:g}",
            "real 👍/👎": f"{real[style]['up']} / {real[style]['down']}",
            "shown (real)": real[style]["shown"],
            "estimated 👍 rate": f"{a / (a + b):.0%}",
            "chance of being picked now": f"{picked[i]:.0%}",
        })
    return pd.DataFrame(rows)


# --- Warm-up simulator ----------------------------------------------------------------

# ASSUMPTION (simulated, not measured): how often a simulated advisor gives 👍 to each style.
# Advisors are assumed to prefer short drafts they can edit, and to like regulatory wording least.
SIMULATED_UP_RATE = {"standard": 0.60, "concise": 0.70, "regulatory": 0.50}
SIMULATED_FEEDBACK_RATE = 0.6  # share of requests where the simulated advisor clicks at all


def simulate(n_requests: int = 60, seed: int = 42) -> dict:
    """Run the bandit itself against simulated advisors; keep the resulting counts as a warm-up."""
    rng = np.random.default_rng(seed)
    counts = {s: {"up": 0, "down": 0} for s in STYLES}
    for _ in range(n_requests):
        draws = {s: rng.beta(1 + c["up"], 1 + c["down"]) for s, c in counts.items()}
        style = max(draws, key=draws.get)
        if rng.random() < SIMULATED_FEEDBACK_RATE:
            counts[style]["up" if rng.random() < SIMULATED_UP_RATE[style] else "down"] += 1
    result = {
        "simulated": True,
        "note": "Warm-up from a simulator, not from real advisors. Counted at weight PRIOR_WEIGHT in bandit.py.",
        "assumed_up_rate": SIMULATED_UP_RATE,
        "feedback_rate": SIMULATED_FEEDBACK_RATE,
        "n_requests": n_requests,
        "seed": seed,
        "counts": counts,
    }
    PRIOR_FILE.parent.mkdir(exist_ok=True)
    PRIOR_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "simulate":
        out = simulate()
        print(f"Wrote {PRIOR_FILE.relative_to(ROOT)}: {json.dumps(out['counts'])}")
    elif command == "status":
        import env
        import logbook

        env.load_env()
        logbook.start()
        print(table(logbook.load()).to_string(index=False))
    else:
        raise SystemExit("usage: uv run python bandit.py [simulate|status]")
