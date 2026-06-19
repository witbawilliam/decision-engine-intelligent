import time
import uuid
import numpy as np
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import quote

API_BASE_URL = "http://api:8000"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_row(row: dict) -> dict:
    """Strip numpy scalar types before JSON serialisation — prevents 422s."""
    clean = {}
    for k, v in row.items():
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = None if (np.isnan(v) or np.isinf(v)) else float(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            v = None
        clean[k] = v
    return clean


def safe_request(method, url, **kwargs):
    try:
        return requests.request(method, url, timeout=60, **kwargs)
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def poll_job(job_id: str, max_wait_s: int = 300) -> dict | None:
    bar   = st.progress(0, text="Training in progress…")
    start = time.time()
    for _ in range(max_wait_s // 5):
        time.sleep(5)
        res = safe_request("GET", f"{API_BASE_URL}/v1/train/status/{job_id}")
        if res is None:
            continue
        if res.status_code == 404:
            st.error("Job not found on server.")
            bar.empty(); return None
        if res.status_code != 200:
            st.warning(f"Status check returned HTTP {res.status_code}, retrying…")
            continue
        data   = res.json()
        prog   = data.get("progress", 0)
        status = data.get("status", "running")
        bar.progress(int(prog), text=f"Training… ({status})")
        if status == "completed": bar.empty(); return data
        if status == "failed":    st.error("Training job failed."); bar.empty(); return None
        if time.time() - start >= max_wait_s:
            st.warning("Timed out waiting for training."); bar.empty(); return None
    bar.empty(); return None


def read_columns(file) -> list:
    try:
        file.seek(0)
        df = pd.read_parquet(file) if file.name.endswith("parquet") else pd.read_csv(file, nrows=5)
        return df.columns.tolist()
    except Exception:
        return []


def load_preview(file, nrows: int = 500) -> pd.DataFrame | None:
    try:
        file.seek(0)
        df = pd.read_parquet(file) if file.name.endswith("parquet") else pd.read_csv(file)
        return df.head(nrows)
    except Exception as e:
        st.error(f"Could not read file: {e}"); return None


def dispatch_training(endpoint: str, prob_type: str, target_col: str, time_col: str) -> None:
    normalized_prob = prob_type.lower().strip()
    payload = {
        "idempotency_key": f"{st.session_state['job_id']}_{normalized_prob}",
        "user_id":         "550e8400-e29b-41d4-a716-446655440000",
        "filename":        st.session_state.get("filename", ""),
        "s3_key":          st.session_state["job_id"],
        "problem_type":    normalized_prob,
        "target_column":   target_col,
    }
    if normalized_prob in ["forecasting", "temporal"]:
        if not time_col:
            st.error("Time Column selection is required for Forecasting."); return
        payload["time_column"]      = time_col
        payload["forecast_horizon"] = st.session_state.get("forecast_horizon", 30)

    train_res = safe_request("POST", f"{API_BASE_URL}{endpoint}", json=payload)
    if train_res is None or train_res.status_code not in (200, 202):
        st.error(f"Dispatch failed: {train_res.text if train_res else 'No Response'}"); return

    response_data = train_res.json()
    raw_job_id    = response_data.get("job_id", payload["idempotency_key"])

    # Store clean model_name separately from the raw S3 job_id
    model_name = response_data.get("model_name") or response_data.get("model_id") or None
    st.session_state["train_job_id"]     = raw_job_id
    st.session_state["train_model_name"] = model_name
    st.session_state["prob_type"]        = normalized_prob
    st.session_state["target_col"]       = target_col

    st.caption(f"Polling status for `{raw_job_id}` …")
    final = poll_job(quote(raw_job_id, safe=""))
    if final:
        resolved = (
            final.get("model_name")
            or final.get("model_id")
            or (final.get("result") or {}).get("model_name")
            or model_name
        )
        st.session_state["train_model_name"] = resolved
        st.session_state["train_status"]     = final
        st.success(f"Training complete! · `{normalized_prob}` · target: `{target_col}`")
        if resolved:
            st.info(f"Model registered as → `{resolved}`")
        else:
            st.warning("Training API did not return a `model_name`. Enter it manually in the override box.")


def show_metrics() -> None:
    if "train_status" not in st.session_state:
        return
    metrics = st.session_state["train_status"].get("metrics", {})
    prob    = st.session_state.get("prob_type", "")
    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    if   prob == "regression":     m1.metric("RMSE", metrics.get("rmse","—")); m2.metric("MAE",      metrics.get("mae","—"))
    elif prob == "classification": m1.metric("Accuracy", metrics.get("accuracy","—")); m2.metric("F1", metrics.get("f1","—"))
    elif prob == "forecasting":    m1.metric("MAPE", metrics.get("mape","—")); m2.metric("RMSE",     metrics.get("rmse","—"))
    m3.write(f"**Model:** `{st.session_state.get('train_model_name') or f'{prob}_v1'}`")


# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION RENDERERS
# Each engine gets its own dedicated rendering function so the logic is clean
# and reusable.
# ─────────────────────────────────────────────────────────────────────────────

def render_manifold(risk_score: float, manifold: dict | None, row_label: str):
    """
    Renders the Manifold Guard panel.
    Shows a gauge-style risk score, a colour-coded badge, and the features
    the guard was fitted on.
    """
    st.markdown("#### 🛡️ Manifold Guard — OOD Risk")

    # colour-coded risk band
    if risk_score >= 0.7:
        colour, label = "#e74c3c", "HIGH RISK — prediction may be unreliable"
    elif risk_score >= 0.35:
        colour, label = "#f39c12", "MODERATE RISK — mild extrapolation"
    else:
        colour, label = "#27ae60", "SAFE — within training manifold"

    st.markdown(
        f"""
        <div style="background:{colour};color:white;padding:10px 16px;
                    border-radius:8px;font-weight:bold;font-size:1.05rem;">
            Risk Score: {risk_score:.3f} &nbsp;|&nbsp; {label}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")  # spacer

    # gauge chart
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(risk_score, 3),
        title = {"text": "OOD Risk Score"},
        gauge = {
            "axis":  {"range": [0, 1]},
            "bar":   {"color": colour},
            "steps": [
                {"range": [0,    0.35], "color": "#d5f5e3"},
                {"range": [0.35, 0.70], "color": "#fdebd0"},
                {"range": [0.70, 1.0],  "color": "#fadbd8"},
            ],
            "threshold": {
                "line":  {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": risk_score,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=30, b=0, l=20, r=20))
    st.plotly_chart(fig, width="stretch")

    if manifold and manifold.get("numeric_columns"):
        st.caption(
            f"Guard fitted on {manifold['feature_count']} features: "
            + ", ".join(f"`{c}`" for c in manifold["numeric_columns"][:8])
            + ("…" if len(manifold["numeric_columns"]) > 8 else "")
        )


def render_sensitivity(sens: dict, row_label: str):
    """
    Renders the Sensitivity Analysis panel.
    Shows a horizontal bar chart of feature sensitivity scores and a table
    of raw values.
    SensitivityResult shape:
      baseline_prediction: float
      feature_rankings: [{feature, baseline_value, sensitivity_score,
                          max_prediction_shift}]
    """
    st.markdown("#### 📊 Sensitivity Analysis")
    st.caption(
        f"Baseline prediction: **{sens.get('baseline_prediction', '?'):.4f}**  "
        f"— how much each feature can shift the output by ±5% perturbation."
    )

    rankings = sens.get("feature_rankings", [])
    if not rankings:
        st.info("No numeric features available for sensitivity analysis.")
        return

    df_sens = pd.DataFrame(rankings).sort_values("sensitivity_score", ascending=False)

    # horizontal bar — top 15 features
    top = df_sens.head(15)
    fig = px.bar(
        top,
        x     = "sensitivity_score",
        y     = "feature",
        orientation = "h",
        color = "sensitivity_score",
        color_continuous_scale = "RdYlGn_r",
        labels = {"sensitivity_score": "Sensitivity (0–1)", "feature": "Feature"},
        title  = "Feature Sensitivity Ranking",
    )
    fig.update_layout(
        yaxis = {"autorange": "reversed"},
        coloraxis_showscale = False,
        height = max(260, 30 * len(top)),
        margin = dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, width="stretch")

    # detail table
    with st.expander("Full sensitivity table"):
        st.dataframe(
            df_sens[["feature", "baseline_value", "sensitivity_score", "max_prediction_shift"]]
            .rename(columns={
                "baseline_value":       "Baseline Value",
                "sensitivity_score":    "Score (0–1)",
                "max_prediction_shift": "Max Output Shift",
            })
            .reset_index(drop=True),
        )


def render_counterfactual(cf: dict, row_label: str):
    """
    Renders the Counterfactual + Optimisation panel.
    CounterfactualResult shape:
      lever_column, original_value, optimized_value, target_goal,
      achieved_prediction, risk_score, status, message
    """
    st.markdown("#### 🎯 Counterfactual & Optimisation")

    status  = cf.get("status", "UNKNOWN")
    message = cf.get("message", "")

    STATUS_COLOUR = {
        "SAFE":          ("#27ae60", "✅"),
        "MODERATE_RISK": ("#f39c12", "⚠️"),
        "HIGH_RISK":     ("#e74c3c", "🔴"),
        "UNREACHABLE":   ("#8e44ad", "🚫"),
        "FAILED":        ("#7f8c8d", "❌"),
    }
    colour, icon = STATUS_COLOUR.get(status, ("#7f8c8d", "❓"))

    st.markdown(
        f"""
        <div style="background:{colour};color:white;padding:10px 16px;
                    border-radius:8px;font-weight:bold;">
            {icon} {status} — {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    lever_col   = cf.get("lever_column",       "—")
    orig_val    = cf.get("original_value",     0.0)
    opt_val     = cf.get("optimized_value",    0.0)
    target      = cf.get("target_goal",        0.0)
    achieved    = cf.get("achieved_prediction",0.0)
    cf_risk     = cf.get("risk_score",         0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lever Feature",    lever_col)
    c2.metric("Original Value",   f"{orig_val:.4f}")
    c3.metric("Optimised Value",  f"{opt_val:.4f}",  delta=f"{opt_val - orig_val:+.4f}")
    c4.metric("Target / Achieved", f"{target:.4f} / {achieved:.4f}")

    # waterfall: original → optimised → effect on prediction
    fig = go.Figure(go.Waterfall(
        orientation = "v",
        measure     = ["absolute", "relative", "absolute"],
        x           = [f"{lever_col} (original)", "Optimisation delta", f"{lever_col} (optimised)"],
        y           = [orig_val, opt_val - orig_val, opt_val],
        connector   = {"line": {"color": "rgb(63, 63, 63)"}},
        decreasing  = {"marker": {"color": "#e74c3c"}},
        increasing  = {"marker": {"color": "#27ae60"}},
        totals      = {"marker": {"color": "#2980b9"}},
    ))
    fig.update_layout(
        title  = f"Lever Optimisation — {lever_col}",
        height = 300,
        margin = dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, width="stretch")

    # prediction comparison bar
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Target Goal",        x=["Goal"],     y=[target],   marker_color="#2980b9"))
    fig2.add_trace(go.Bar(name="Achieved Prediction",x=["Achieved"], y=[achieved], marker_color=colour))
    fig2.update_layout(
        title     = "Target vs Achieved Prediction",
        barmode   = "group",
        height    = 260,
        showlegend= True,
        margin    = dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig2, width="stretch")

    st.caption(f"Counterfactual manifold risk: **{cf_risk:.3f}**")


def render_explanation_panel(d: dict, i: int, infer_target: str, numeric_cols: list):
    """
    Top-level explanation renderer for one inference response.
    Called once per row. Renders manifold, sensitivity, and counterfactual
    panels inside a styled expander.
    """
    risk_score  = d.get("risk_score", 0.0)
    explanation = d.get("explanations") or {}
    sens        = explanation.get("sensitivity")
    cf          = explanation.get("counterfactual")
    manifold    = explanation.get("manifold")

    label = f"Row {i+1} · prediction={d.get('prediction','?')} · risk={risk_score:.2f}"
    with st.expander(f"🔍 Decision Intelligence — {label}", expanded=(i == 0)):

        tab_manifold, tab_sensitivity, tab_counterfactual = st.tabs([
            "🛡️ Manifold Guard",
            "📊 Sensitivity",
            "🎯 Counterfactual & Optimisation",
        ])

        with tab_manifold:
            render_manifold(risk_score, manifold, f"row {i+1}")

        with tab_sensitivity:
            if sens:
                render_sensitivity(sens, f"row {i+1}")
            else:
                st.info("Sensitivity data not available for this row.")

        with tab_counterfactual:
            if cf:
                render_counterfactual(cf, f"row {i+1}")
            else:
                st.info(
                    "No counterfactual was generated. To run optimisation, "
                    "fill in the **Counterfactual Controls** below the inference "
                    "settings and re-fetch."
                )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Decision Intelligence", layout="wide")
st.title("🤖 AutoML Decision Dashboard")


# ── DATA INGESTION ────────────────────────────────────────────────────────────
st.header("📂 Data Ingestion")

uploaded_file = st.file_uploader(
    "Upload Dataset (CSV, Parquet, Excel)", type=["csv", "parquet", "xlsx"]
)

if uploaded_file:
    file_size_mb = round(uploaded_file.size / (1024 * 1024), 2)
    if st.button("Ingest & Process Dataset"):
        res = safe_request(
            "POST", f"{API_BASE_URL}/datasets/upload",
            files=[("files", (uploaded_file.name, uploaded_file, uploaded_file.type))],
            data=[
                ("user_id",       "550e8400-e29b-41d4-a716-446655440000"),
                ("file_types",    uploaded_file.name.split(".")[-1]),
                ("file_sizes_mb", file_size_mb),
            ],
        )
        if res and res.status_code == 201:
            data = res.json()
            st.session_state["job_id"]     = data["job_id"]
            st.session_state["filename"]   = uploaded_file.name
            st.session_state["df_columns"] = read_columns(uploaded_file)
            st.success(f"Dataset uploaded! · Job ID: `{data['job_id']}`")
            c1, c2, c3, _ = st.columns(4)
            c1.metric("Rows",    data.get("rows",    "N/A"))
            c2.metric("Columns", data.get("columns", "N/A"))
            c3.info(f"Merged: {data.get('merged', False)}")
            if data.get("rows", 0) < 100:
                st.warning("Low sample size — training results may be unreliable.")
            if data.get("missing_count", 0) > 10:
                st.warning("High missing data — imputation will be applied.")
        elif res:
            st.error(f"Upload failed (HTTP {res.status_code}): {res.text}")


# ── TRAINING ──────────────────────────────────────────────────────────────────
st.divider()
st.header("⚙️ Model Execution & Evaluation")

if "job_id" not in st.session_state:
    st.info("Upload a dataset first to unlock training.")
else:
    columns = st.session_state.get("df_columns", [])

    if columns:
        target_col = st.selectbox(
            "Target Column",
            options=[" — select — "] + columns,
            index=0,
        )
        target_col = None if target_col == " — select — " else target_col
        if not target_col:
            st.warning("Choose your target column before running either pipeline.")
    else:
        target_col = st.text_input(
            "Target Column Name",
            value=st.session_state.get("target_col", ""),
            placeholder="e.g. price, churn, Class …",
        ) or None

    st.divider()
    tab_tabular, tab_forecast = st.tabs(["Tabular Pipeline", "Forecasting Pipeline"])

    with tab_tabular:
        tabular_prob = st.radio("Problem Type", ["regression", "classification"], horizontal=True, key="tabular_prob")
        if target_col:
            if st.button("▶ Run Tabular Pipeline", type="primary", key="btn_tabular"):
                st.session_state["prob_type"] = tabular_prob
                st.session_state["target_col"] = target_col
                dispatch_training("/v1/train/tabular", tabular_prob, target_col, None)
        else:
            st.warning("Select a target column above before running.")
        show_metrics()

    with tab_forecast:
        date_col        = st.selectbox("Date / Time Column", ["— select —"] + columns, key="forecast_date_col")
        forecast_horizon = st.number_input("Forecast Horizon (steps)", 1, 365, 30, key="forecast_horizon")
        if target_col and date_col != "— select —":
            if st.button("▶ Run Forecasting Pipeline", type="primary", key="btn_forecast"):
                st.session_state["prob_type"]  = "forecasting"
                st.session_state["target_col"] = target_col
                dispatch_training("/v1/train/forecast", "forecasting", target_col, date_col)
        else:
            st.warning("Select both a target and a date column before running.")
        show_metrics()


# ── INSIGHTS ──────────────────────────────────────────────────────────────────
st.divider()
st.header("📈 Insights & Visualization")

if uploaded_file:
    df_preview = load_preview(uploaded_file)
    if df_preview is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Feature Distribution")
            dist_col = st.selectbox("Column to plot", df_preview.columns.tolist(), key="dist_col")
            st.plotly_chart(px.histogram(df_preview, x=dist_col, nbins=30), width="stretch")
        with c2:
            st.write("### Target Trend")
            tc = st.session_state.get("target_col")
            default_trend = tc if tc in df_preview.columns else df_preview.columns[0]
            trend_col = st.selectbox(
                "Column to trend",
                df_preview.columns.tolist(),
                index=df_preview.columns.tolist().index(default_trend),
                key="trend_col",
            )
            st.plotly_chart(px.line(df_preview, y=trend_col), width="stretch")


# ── PREDICTION vs ACTUAL + DECISION INTELLIGENCE ──────────────────────────────
st.divider()
st.header("🎯 Prediction vs Actual & Decision Intelligence")

if "train_status" not in st.session_state:
    st.info("Run a training pipeline first to generate predictions.")
elif uploaded_file is None:
    st.info("Re-upload the dataset to generate predictions.")
else:
    df_pred = load_preview(uploaded_file)

    if df_pred is not None:
        pred_cols    = df_pred.columns.tolist()
        numeric_cols = df_pred.select_dtypes(include="number").columns.tolist()

        # ── Row 1: core inference controls ───────────────────────────────────
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            default_target = st.session_state.get("target_col", pred_cols[0])
            infer_target = st.selectbox(
                "Target Column (actuals)",
                pred_cols,
                index=pred_cols.index(default_target) if default_target in pred_cols else 0,
                key="infer_target",
            )
        with pc2:
            infer_pipeline = st.radio(
                "Pipeline Type", ["tabular", "forecasting"],
                index=1 if st.session_state.get("prob_type") == "forecasting" else 0,
                horizontal=True, key="infer_pipeline",
            )
        with pc3:
            infer_n = st.number_input(
                "Sample rows", min_value=1, max_value=len(df_pred),
                value=min(10, len(df_pred)), key="infer_n",
            )

        # ── Model name override ───────────────────────────────────────────────
        resolved_model = st.session_state.get("train_model_name")
        with st.expander("⚙️ Model name override (opens automatically if auto-detect failed)",
                         expanded=not bool(resolved_model)):
            manual = st.text_input(
                "Model registry name",
                value=resolved_model or "",
                placeholder="e.g. xgboost_regression",
                key="manual_model_override",
            )
            if manual.strip():
                resolved_model = manual.strip()

        if not resolved_model:
            prob           = st.session_state.get("prob_type", "regression")
            resolved_model = f"prophet_{prob}" if infer_pipeline == "forecasting" else f"xgboost_{prob}"
            st.warning(f"No model name found — using fallback `{resolved_model}`.")

        # ── Row 2: explanation toggle ─────────────────────────────────────────
        include_explanations = st.toggle(
            "🔍 Enable Decision Intelligence (Manifold · Sensitivity · Counterfactual · Optimisation)",
            value=False, key="include_explanations",
            help="Runs all four explanation engines per row. Slower but gives full insight.",
        )

        # ── Row 3: counterfactual controls (only shown when explanations on) ──
        lever_col = lever_min = lever_max = target_goal = None
        if include_explanations:
            st.markdown("**Counterfactual & Optimisation Controls**")
            st.caption(
                "Set a lever feature and a target prediction value. "
                "The optimiser (BinaryRefinementSearch) will find what value "
                "the lever needs to reach that target."
            )
            cf1, cf2, cf3, cf4 = st.columns(4)
            with cf1:
                lever_col = st.selectbox(
                    "Lever Feature",
                    options=["— none —"] + [c for c in numeric_cols if c != infer_target],
                    key="lever_col",
                    help="The feature the model should tweak to reach the target.",
                )
                lever_col = None if lever_col == "— none —" else lever_col
            with cf2:
                def _safe_median(df, col):
                    try:
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                            return float(df[col].median())
                    except Exception:
                        pass
                    return 0.0

                target_goal = st.number_input(
                    "Target Prediction Goal",
                    value=_safe_median(df_pred, infer_target),
                    key="target_goal",
                    help="The prediction value you want to achieve.",
                )
            with cf3:
                def _safe_min(df, col):
                    try:
                        if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                            return float(df[col].min())
                    except Exception:
                        pass
                    return 0.0

                lever_min = st.number_input(
                    "Lever Min Bound",
                    value=_safe_min(df_pred, lever_col),
                    key="lever_min",
                )
            with cf4:
                def _safe_max(df, col):
                    try:
                        if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                            return float(df[col].max())
                    except Exception:
                        pass
                    return 1.0

                lever_max = st.number_input(
                    "Lever Max Bound",
                    value=_safe_max(df_pred, lever_col),
                    key="lever_max",
                )

            if lever_col:
                st.info(
                    f"Optimiser will search `{lever_col}` in "
                    f"[{lever_min:.2f}, {lever_max:.2f}] to hit prediction = {target_goal:.2f}"
                )

        # ── FETCH 
        if st.button(" Fetch Predictions", type="primary"):

            sample = df_pred.head(int(infer_n))

            features_df = (
                sample
                .drop(columns=[infer_target], errors="ignore")
                .fillna(0)
            )

            rows = features_df.to_dict(orient="records")

            st.caption(
                f"Model: `{resolved_model}` | "
                f"Rows: {len(rows)}"
            )

            predictions = []
            actuals = []
            latencies = []
            explanations = []

            progress = st.progress(0)

            for idx, row in enumerate(rows):

                payload = {
                    "model_name":           resolved_model,
                    "features":             sanitize_row(row),
                    "include_explanations": include_explanations,
                    # Counterfactual lever controls — only sent when set
                    "lever_col":            lever_col   if lever_col   else None,
                    "target_goal":          float(target_goal) if lever_col and target_goal is not None else None,
                    "lever_min":            float(lever_min)   if lever_col and lever_min   is not None else None,
                    "lever_max":            float(lever_max)   if lever_col and lever_max   is not None else None,
                }

                response = safe_request(
                    "POST",
                    f"{API_BASE_URL}/v1/inference/predict",
                    json=payload,
                )

                if response and response.status_code == 200:

                    data = response.json()

                    predictions.append(data.get("prediction"))
                    latencies.append(data.get("latency_ms", 0))

                    actuals.append(
                        sample.iloc[idx].get(infer_target)
                    )

                    explanations.append(
                        data.get("explanations", {})
                    )

                else:
                    predictions.append(None)
                    actuals.append(None)
                    latencies.append(0)
                    explanations.append(None)

                progress.progress((idx + 1) / len(rows))

            progress.empty()

            st.success("Inference completed")

            summary_df = pd.DataFrame({
                "Actual": actuals,
                "Prediction": predictions,
                "Latency (ms)": latencies,
            })

            st.dataframe(summary_df, width="stretch")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Rows Processed",
                len(predictions)
            )

            col2.metric(
                "Average Latency",
                f"{np.mean(latencies):.2f} ms"
            )

            col3.metric(
                "Model",
                resolved_model
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    y=actuals,
                    mode="lines+markers",
                    name="Actual"
                )
            )

            fig.add_trace(
                go.Scatter(
                    y=predictions,
                    mode="lines+markers",
                    name="Prediction"
                )
            )

            fig.update_layout(
                title="Prediction vs Actual",
                height=400,
            )

            st.plotly_chart(fig, width="stretch")

            if include_explanations:

                st.markdown(" Decision Intelligence")

                # Build full response dicts so render_explanation_panel
                # can access both top-level risk_score and nested explanations
                full_responses = []
                for idx in range(len(predictions)):
                    full_responses.append({
                        "prediction":   predictions[idx],
                        "risk_score":   (explanations[idx] or {}).get("manifold", {}).get("risk_score", 0.0)
                                        if explanations[idx] else 0.0,
                        "explanations": explanations[idx],
                    })

                for idx, response_data in enumerate(full_responses):
                    if not response_data.get("explanations"):
                        continue
                    render_explanation_panel(
                        response_data,
                        idx,
                        infer_target,
                        numeric_cols,
                    )