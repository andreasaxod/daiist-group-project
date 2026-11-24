"""
Component A: train the risk-profile model. `uv run python train.py`

Reads data/clients_to_label.csv. Uses the `advisor_label` column when it is fully
filled in (the point of the project: the model learns an advisor's judgement), and
falls back to `draft_label` otherwise, saying so loudly and in the model version.

Compares a few scikit-learn models with 5-fold stratified cross-validation, keeps
the best by macro-F1, measures it on a held-out test set against the points-grid
baseline, and writes models/risk_model.joblib and models/model_card.json (commit both).
"""

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import suitability

ROOT = Path(__file__).resolve().parent
LABEL_FILE = ROOT / "data" / "clients_to_label.csv"
MODEL_FILE = ROOT / "models" / "risk_model.joblib"
CARD_FILE = ROOT / "models" / "model_card.json"
SEED = 42

CANDIDATES = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, random_state=SEED),
}


def expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray, classes: np.ndarray, bins: int = 5) -> float:
    """How far the model's confidence is from its accuracy (0 = perfectly calibrated)."""
    confidence = probabilities.max(axis=1)
    correct = classes[probabilities.argmax(axis=1)] == y_true
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def main() -> None:
    data = pd.read_csv(LABEL_FILE)
    advisor = pd.to_numeric(data["advisor_label"], errors="coerce")
    if advisor.notna().all():
        label_source, y = "advisor", advisor.astype(int).to_numpy()
    else:
        label_source, y = "draft", data["draft_label"].astype(int).to_numpy()
        print(
            f"WARNING: {advisor.notna().sum()}/{len(data)} advisor labels filled in. Training on `draft_label`,\n"
            "the generator's stand-in. Fill in every `advisor_label` (1-5) and run this again before you report results.\n"
        )
    profiles = data.to_dict("records")
    for p in profiles:
        p["needs_liquidity_2y"] = str(p["needs_liquidity_2y"]).strip().lower() == "true"
    X = pd.DataFrame([suitability.model_features(p) for p in profiles])[suitability.FEATURE_NAMES]
    grid = np.array([suitability.grid_class(p) for p in profiles])

    X_train, X_test, y_train, y_test, _, grid_test = train_test_split(
        X, y, grid, test_size=0.2, stratify=y, random_state=SEED
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv = {}
    for name, model in CANDIDATES.items():
        scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1_macro")
        cv[name] = {"macro_f1_mean": round(scores.mean(), 3), "macro_f1_std": round(scores.std(), 3)}
        print(f"{name:<20} CV macro-F1 {scores.mean():.3f} ± {scores.std():.3f}")
    best = max(cv, key=lambda n: cv[n]["macro_f1_mean"])
    model = CANDIDATES[best].fit(X_train, y_train)

    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    test = {
        "n": int(len(y_test)),
        "model": {
            "accuracy": round(accuracy_score(y_test, predicted), 3),
            "macro_f1": round(f1_score(y_test, predicted, average="macro"), 3),
            "within_one_class": round(float(np.mean(np.abs(predicted - y_test) <= 1)), 3),
            "too_risky_share": round(float(np.mean(predicted > y_test)), 3),
            "expected_calibration_error": round(expected_calibration_error(probabilities, y_test, model.classes_), 3),
        },
        "grid_baseline": {
            "accuracy": round(accuracy_score(y_test, grid_test), 3),
            "macro_f1": round(f1_score(y_test, grid_test, average="macro"), 3),
            "within_one_class": round(float(np.mean(np.abs(grid_test - y_test) <= 1)), 3),
            "too_risky_share": round(float(np.mean(grid_test > y_test)), 3),
        },
    }
    print(f"\nBest: {best}. Held-out test set ({test['n']} clients):")
    print(pd.DataFrame(test["model"] | {}, index=["model"]).T.join(pd.DataFrame(test["grid_baseline"], index=["grid"]).T).to_string())

    # Refit on all the data for serving; the test numbers above are the honest estimate.
    model = CANDIDATES[best].fit(X, y)
    version = f"{best}-{label_source}-labels-{date.today().isoformat()}"
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": suitability.FEATURE_NAMES, "version": version}, MODEL_FILE)
    card = {
        "version": version, "label_source": label_source, "n_clients": int(len(y)),
        "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "features": suitability.FEATURE_NAMES, "cross_validation": cv, "chosen": best, "held_out_test": test,
    }
    CARD_FILE.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {MODEL_FILE.relative_to(ROOT)} and {CARD_FILE.relative_to(ROOT)} (version {version}). Commit both.")
    if label_source != "advisor":
        sys.exit(0)


if __name__ == "__main__":
    main()
