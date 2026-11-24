"""
The app: an App tab and a Monitoring tab. `uv run python app.py`

Keep the logic in pipeline.py (or your own modules); this file only wires it to
the interface and the logs. Keep `serve()` and the `/run` endpoint: check.py
calls `serve()` locally and `/run` on your deployed app, with one input from
eval/cases.csv (text, or a file path).

The App tab has a form for the advisor. "Build request" turns it into the JSON
request (the same format as the `input` column of eval/cases.csv), and "Assess"
sends that JSON to /run.
"""

import json
import os
import time

import gradio as gr

import env
import logbook
import monitor
import pipeline
import suitability

env.load_env()
LOG_TARGET = logbook.start()

TITLE = "Suitability Copilot"  # TODO: your product's name
DESCRIPTION = (
    "For private-banking advisors: predicts the client's risk profile from the questionnaire, "
    "checks the proposed portfolio against it, and drafts the MiFID II suitability statement for the advisor to sign. "
    "Synthetic clients only; no personal data."
)

FUND_CHOICES = [f"{fid} (SRI {f['sri']})" for fid, f in suitability.FUNDS.items()]
EXAMPLE = {
    "age": 47, "horizon_years": 12, "annual_income_k": 160, "liquid_wealth_k": 1200, "experience": 2,
    "loss_reaction": "hold", "objective": "growth", "max_drawdown_ok_pct": 25, "needs_liquidity_2y": False,
    "dependents": 2,
    "portfolio": [{"fund_id": "MSCI-WORLD", "weight_pct": 50}, {"fund_id": "CARM-SECU", "weight_pct": 30},
                  {"fund_id": "SP500-TECH", "weight_pct": 20}],
}


def serve(raw_input: str) -> dict:
    """Run one request through the pipeline, log it, and return the logged record."""
    started = time.perf_counter()
    result = pipeline.run(raw_input)
    return logbook.log_request(result | {"latency_s": round(time.perf_counter() - started, 3)})


# --- App tab -----------------------------------------------------------------

def build_request(age, horizon, income, wealth, experience, reaction, objective, drawdown, liquidity, dependents,
                  f1, w1, f2, w2, f3, w3, f4, w4):
    portfolio = [
        {"fund_id": f.split(" ")[0], "weight_pct": float(w)}
        for f, w in ((f1, w1), (f2, w2), (f3, w3), (f4, w4)) if f and w
    ]
    request = {
        "age": int(age), "horizon_years": int(horizon), "annual_income_k": float(income),
        "liquid_wealth_k": float(wealth), "experience": int(experience), "loss_reaction": reaction,
        "objective": objective, "max_drawdown_ok_pct": float(drawdown), "needs_liquidity_2y": bool(liquidity),
        "dependents": int(dependents), "portfolio": portfolio,
    }
    return json.dumps(request)


def on_run(raw_input):
    if raw_input is None or not str(raw_input).strip():
        raise gr.Error("Fill in the form and click 'Build request' first.")
    record = serve(raw_input)
    a, b = record["outputs"]["risk_model"], record["outputs"]["suitability_agent"]
    if a.get("refused"):
        return a, b["statement"], "⛔ **Out of scope.** " + a["reason"], record["id"]
    verdict = "✅ **Suitable**" if b["suitable"] else "❌ **Not suitable**: " + " ".join(b["issues"])
    status = "⚠️ **Sent to the advisor for review.**" if record["review"] else "✅ No review flag (the advisor still signs)."
    if b.get("error"):
        status += f"  LLM unavailable ({b['error']}): template statement."
    profile = f"Risk class **{a['risk_class']} ({a['profile_name']})**, confidence {a['confidence']:.0%} · points grid says {a['grid_baseline_class']}"
    return (
        a,
        b["statement"],
        f"{profile}\n\n{verdict}\n\n{status}  Latency {record['latency_s']} s, cost €{record['cost_eur']:.5f}.",
        record["id"],
    )


def on_feedback(request_id, value):
    if not request_id:
        gr.Warning("Run a request first.")
        return
    logbook.log_feedback(request_id, value)
    gr.Info("Thanks for the feedback.")


# --- Monitoring tab ----------------------------------------------------------

def on_refresh():
    logs = logbook.load()
    reference = monitor.load_reference()
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    if reference is None:
        alert_text = "No reference set yet: run `uv run python main.py reference` and commit `monitoring/`."
    elif fired:
        alert_text = "### 🚨 Alerts\n" + "\n".join(f"- {message}" for message in fired)
    else:
        alert_text = "### ✅ No alerts"
    if reference is not None and len(logs) >= monitor.MIN_WINDOW:
        drift_table = monitor.drift(reference, monitor.live_features(logs))
    else:
        drift_table = monitor.drift(reference, reference.head(0)) if reference is not None else None
    return alert_text, monitor.summary(logs), monitor.quality(logs), drift_table


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    if missing := env.missing():
        gr.Markdown(f"⚠️ **Missing environment variables:** {', '.join(missing)}. See SETUP.md.")

    with gr.Tab("App"):
        request_id = gr.State("")
        gr.Markdown("### 1. Client questionnaire")
        with gr.Row():
            age = gr.Number(EXAMPLE["age"], label="Age", precision=0)
            horizon = gr.Number(EXAMPLE["horizon_years"], label="Investment horizon (years)", precision=0)
            income = gr.Number(EXAMPLE["annual_income_k"], label="Annual income (k€)")
            wealth = gr.Number(EXAMPLE["liquid_wealth_k"], label="Liquid wealth (k€)")
            dependents = gr.Number(EXAMPLE["dependents"], label="Dependents", precision=0)
        with gr.Row():
            experience = gr.Dropdown([("None", 0), ("Basic", 1), ("Informed", 2), ("Expert", 3)], value=2, label="Investment experience")
            reaction = gr.Dropdown(suitability.LOSS_REACTIONS, value=EXAMPLE["loss_reaction"], label="Reaction to a 20% fall")
            objective = gr.Dropdown(suitability.OBJECTIVES, value=EXAMPLE["objective"], label="Objective")
            drawdown = gr.Number(EXAMPLE["max_drawdown_ok_pct"], label="Max. temporary loss accepted (%)")
            liquidity = gr.Checkbox(EXAMPLE["needs_liquidity_2y"], label="Needs a large part within 2 years")
        gr.Markdown("### 2. Proposed portfolio (weights add up to 100%)")
        fund_inputs = []
        with gr.Row():
            for i in range(4):
                line = EXAMPLE["portfolio"][i] if i < len(EXAMPLE["portfolio"]) else None
                with gr.Column(min_width=160):
                    default = next((c for c in FUND_CHOICES if line and c.startswith(line["fund_id"] + " ")), None)
                    fund_inputs.append(gr.Dropdown(FUND_CHOICES, value=default, label=f"Fund {i + 1}"))
                    fund_inputs.append(gr.Number(line["weight_pct"] if line else 0, label="Weight %"))
        build = gr.Button("Build request")
        raw_input = gr.Textbox(label="Request (JSON, the format of eval/cases.csv)", lines=3)
        run_button = gr.Button("Assess", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            out_b = gr.Textbox(label="Draft suitability statement (component B: agent)", lines=8)
            out_a = gr.JSON(label="Risk profile (component A: model)")
        with gr.Row():
            up = gr.Button("👍")
            down = gr.Button("👎")
        form = [age, horizon, income, wealth, experience, reaction, objective, drawdown, liquidity, dependents, *fund_inputs]
        build.click(build_request, form, raw_input, api_name=False)
        run_button.click(on_run, raw_input, [out_a, out_b, status, request_id], api_name="run")
        up.click(lambda rid: on_feedback(rid, +1), request_id, None, api_name=False)
        down.click(lambda rid: on_feedback(rid, -1), request_id, None, api_name=False)

    with gr.Tab("Monitoring"):
        refresh = gr.Button("Refresh")
        alert_box = gr.Markdown()
        summary_table = gr.Dataframe(label="Summary", interactive=False)
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", label="Quality over time", height=300)
        drift_table = gr.Dataframe(label=f"Drift: PSI of the last {monitor.WINDOW} requests against the reference set", interactive=False)
        gr.Markdown(f"<small>{LOG_TARGET}</small>")
        outputs = [alert_box, summary_table, quality_plot, drift_table]
        refresh.click(on_refresh, None, outputs, api_name=False)
        demo.load(on_refresh, None, outputs, api_name=False)


if __name__ == "__main__":
    # Render gives the port in $PORT; locally it is 7860.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
