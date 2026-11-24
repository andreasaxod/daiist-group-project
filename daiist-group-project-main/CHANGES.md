# What changed from the template

Delete this file before you tag `final` (or keep it as a record of the starting point).

## Removed
- **The placeholder logic in `pipeline.py`** (keyword "classifier" and canned replies): replaced by the real components.
- **`ROBOFLOW_API_KEY` from `.env.example`**: the product does not use Roboflow. (It stays in `render.yaml` and `verify.yml`, which the brief says not to edit; leave it empty on Render.)

## Added
| File | What it is |
|---|---|
| `suitability.py` | Domain rules: the questionnaire and its validation, the points-grid baseline, the fund lookup and the deterministic suitability check (the agent's tools) |
| `train.py` | Trains component A (3 scikit-learn models compared with 5-fold CV), writes the model and its card |
| `data/generate.py` | Synthetic clients to label, plus `eval/reference.csv` and `eval/shifted.csv` |
| `data/funds.csv` | The fund table (illustrative for now) |
| `data/clients_to_label.csv` | 150 synthetic clients, with an empty `advisor_label` column to fill in |
| `models/risk_model.joblib`, `models/model_card.json` | The trained model (on draft labels for now) and its metrics |
| `monitoring/reference_features.csv` | The reference features for the drift check |
| `eval/reference.csv`, `eval/shifted.csv` | 40 normal and 15 shifted (retiree) requests |
| `cache/` | Where the agent's and the judge's answers are cached (commit it) |
| `CHANGES.md` | This file |

## Changed
| File | Change |
|---|---|
| `pipeline.py` | Component A (risk model) → component B (Gemini agent with tool calling), review rule, drift features, out-of-scope refusal. Same `run()` format as the template |
| `app.py` | Advisor form (questionnaire + up to 4 funds), Build request / Assess, profile and verdict display. `serve()` and `/run` unchanged |
| `evaluate.py` | Component A vs grid, end to end vs baseline with bootstrap CIs, LLM judge + agreement with your labels (Cohen's kappa), cost and latency per component |
| `monitor.py` | Two more quality signals: share not suitable, share out of scope |
| `.env.example` | `GEMINI_API_KEY` (required), `GEMINI_MODEL` (optional) |
| `render.yaml` | Service name `suitability-copilot` |
| `pyproject.toml`, `uv.lock` | Added `scikit-learn` and `google-genai` |
| `eval/README.md` | The input and `expected` formats for this product |
| `REPORT.md` | Descriptive sections drafted; analysis sections left for the team |

Unchanged: `check.py`, `env.py`, `main.py`, `logbook.py`, `SETUP.md`, `README.md`, `CONTRIBUTIONS.md`, `.github/`.

## Still to do (the team's part)
1. **Label the 150 clients**: fill `advisor_label` (1-5) in `data/clients_to_label.csv`, then `uv run python train.py` and `uv run python main.py reference`.
2. **Replace the illustrative funds** in `data/funds.csv` with real ones from their KIDs (keep the ids, or change them in your cases too).
3. **Write `eval/cases.csv`**: 30+ cases (see `eval/README.md`).
4. Run `uv run python main.py evaluate` with your Gemini key, then **label 20+ statements** in `eval/human_labels.csv` and run it again.
5. **Confirm or change the TEAM DECISION values**: suitability limits (`suitability.py`), the grid, `REVIEW_BELOW_CONFIDENCE` (`pipeline.py`), the alert thresholds (`monitor.py`). Check the Gemini price constants in `pipeline.py`.
6. Fork, push, deploy on Render (SETUP.md), run `uv run python check.py`, fill REPORT.md and CONTRIBUTIONS.md.
7. Optional: the RL bonus (a bandit over statement styles, learning from 👍/👎).
