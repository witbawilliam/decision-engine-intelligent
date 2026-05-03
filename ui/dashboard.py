import time
import uuid
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import quote

API_BASE_URL = "http://api:8000"

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def safe_request(method, url, **kwargs):
    try:
        res = requests.request(method, url, timeout=60, **kwargs)
        return res
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def poll_job(job_id: str, max_wait_s: int = 300) -> dict | None:
    bar   = st.progress(0, text="Training in progress…")
    start = time.time()

    for _ in range(max_wait_s // 5):
        time.sleep(5)
        # job_id is already URL-encoded (slashes → %2F) so the router
        # treats the entire S3 path as a single path parameter.
        res = safe_request("GET", f"{API_BASE_URL}/v1/train/status/{job_id}")

        if res is None:
            continue
        if res.status_code == 404:
            st.error("Job not found on server.")
            bar.empty()
            return None
        if res.status_code != 200:
            st.warning(f"Status check returned HTTP {res.status_code}, retrying…")
            continue

        data     = res.json()
        progress = data.get("progress", 0)
        status   = data.get("status", "running")
        bar.progress(int(progress), text=f"Training… ({status})")

        if status == "completed":
            bar.empty()
            return data
        if status == "failed":
            st.error("Training job failed on the server.")
            bar.empty()
            return None
        if time.time() - start >= max_wait_s:
            st.warning("Timed out waiting for training to complete.")
            bar.empty()
            return None

    bar.empty()
    return None


def read_columns(file) -> list:
    try:
        file.seek(0)
        df = (
            pd.read_parquet(file)
            if file.name.endswith("parquet")
            else pd.read_csv(file, nrows=5)
        )
        return df.columns.tolist()
    except Exception:
        return []


def load_preview(file, nrows: int = 500) -> pd.DataFrame | None:
    try:
        file.seek(0)
        df = (
            pd.read_parquet(file)
            if file.name.endswith("parquet")
            else pd.read_csv(file)
        )
        return df.head(nrows)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None


def dispatch_training(endpoint: str, prob_type: str, target_col: str) -> None:
    # 1. NORMALIZE: standardizes "Forecasting" or "Regression" to lowercase
    normalized_prob = prob_type.lower().strip()

    # 2. BASE PAYLOAD
    payload = {
        "idempotency_key": f"{st.session_state['job_id']}_{normalized_prob}",
        "user_id":         "550e8400-e29b-41d4-a716-446655440000",
        "filename":        st.session_state.get("filename", ""),
        "s3_key":          st.session_state["job_id"],
        "problem_type":    normalized_prob, 
        "target_column":   target_col,
    }

    # 3. FORECASTING/TEMPORAL LOGIC: Add mandatory time parameters
    # If the user chose forecasting/temporal, the backend MUST know the time axis.
    if normalized_prob in ["forecasting", "temporal"]:
        time_col = st.session_state.get("selected_time_col")
        
        if not time_col:
            st.error("Time Column selection is required for Forecasting.")
            return
            
        payload["time_column"] = time_col
        # Default horizon if not set (e.g., 30 steps ahead)
        payload["forecast_horizon"] = st.session_state.get("forecast_horizon", 30)

    # 4. DISPATCH
    print(f"DEBUG: Dispatching {normalized_prob} for {target_col}")
    train_res = safe_request("POST", f"{API_BASE_URL}{endpoint}", json=payload)

    if train_res is None or train_res.status_code not in (200, 202):
        err = train_res.text if train_res else "No Response"
        st.error(f"Dispatch failed: {err}")
        return

    # 5. STATE MANAGEMENT
    response_data = train_res.json()
    raw_job_id = response_data.get("job_id", payload["idempotency_key"])
    task_id    = quote(raw_job_id, safe="")   

    st.session_state["train_job_id"] = raw_job_id   
    st.session_state["prob_type"]    = normalized_prob
    st.session_state["target_col"]   = target_col

    # 6. POLLING
    st.caption(f"Polling status for `{raw_job_id}` …")
    final = poll_job(task_id)
    
    if final:
        st.session_state["train_status"] = final
        st.success(f"Training complete! · `{normalized_prob}` · target: `{target_col}`")


def show_metrics() -> None:
    if "train_status" not in st.session_state:
        return
    metrics = st.session_state["train_status"].get("metrics", {})
    prob    = st.session_state.get("prob_type", "")

    st.subheader("🎯 Model Performance")
    m1, m2, m3 = st.columns(3)
    if prob == "regression":
        m1.metric("RMSE", metrics.get("rmse", "—"))
        m2.metric("MAE",  metrics.get("mae",  "—"))
    elif prob == "classification":
        m1.metric("Accuracy", metrics.get("accuracy", "—"))
        m2.metric("F1-Score", metrics.get("f1",       "—"))
    elif prob == "forecasting":
        m1.metric("MAPE", metrics.get("mape", "—"))
        m2.metric("RMSE", metrics.get("rmse", "—"))
    m3.write(f"**Model:** `{prob}_v1`")


# ---------------------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Decision Intelligence", layout="wide")
st.title(" AutoML Decision Dashboard")

# ── 1. UPLOAD ────────────────────────────────────────────────────────────────
st.header("1. Data Ingestion")

uploaded_file = st.file_uploader(
    "Upload Dataset (CSV, Parquet, Excel)", type=["csv", "parquet", "xlsx"]
)

if uploaded_file:
    file_size_mb = round(uploaded_file.size / (1024 * 1024), 2)

    if st.button("Ingest & Process Dataset"):
        files_payload = [("files", (uploaded_file.name, uploaded_file, uploaded_file.type))]
        data_payload  = [
            ("user_id",       "550e8400-e29b-41d4-a716-446655440000"),
            ("file_types",    uploaded_file.name.split(".")[-1]),
            ("file_sizes_mb", file_size_mb),
        ]

        res = safe_request(
            "POST", f"{API_BASE_URL}/datasets/upload",
            files=files_payload, data=data_payload,
        )

        if res and res.status_code == 201:
            data = res.json()
            st.session_state["job_id"]     = data["job_id"]
            st.session_state["filename"]   = uploaded_file.name
            st.session_state["df_columns"] = read_columns(uploaded_file)

            st.success(f"Dataset uploaded!  ·  Job ID: `{data['job_id']}`")

            st.subheader("Dataset Ingestion Summary")
            c1, c2, c3, _ = st.columns(4)
            c1.metric("Rows",    data.get("rows",    "N/A"))
            c2.metric("Columns", data.get("columns", "N/A"))
            c3.info(f"Merged: {data.get('merged', False)}")

            st.write("###  Data Quality Alerts")
            if data.get("rows", 0) < 100:
                st.warning("Low sample size — training results may be unreliable.")
            if data.get("missing_count", 0) > 10:
                st.warning("High missing data — imputation will be applied.")
        elif res:
            st.error(f"Upload failed (HTTP {res.status_code}): {res.text}")


# ── 2. TRAINING ──────────────────────────────────────────────────────────────
st.divider()
st.header("2. Model Execution & Evaluation")

if "job_id" not in st.session_state:
    st.info("Upload a dataset first to unlock training.")
else:
    columns = st.session_state.get("df_columns", [])

    # ── Shared config (target column) ─────────────────────────────────────────
    st.subheader(" Shared Configuration")

    if columns:
        # "— select —" sentinel forces the user to make an explicit choice.
        # index=0 previously defaulted to the first column (e.g. "Time")
        # which is almost never the right target — the user must pick it.
        target_col = st.selectbox(
            "Target Column",
            options=[" select a target column "] + columns,
            index=0,
            help="The column your model will learn to predict. Applies to both pipelines.",
        )
        if target_col == " select a target column ":
            target_col = None
            st.warning("⬆ Choose your target column before running either pipeline.")
    else:
        target_col = st.text_input(
            "Target Column Name",
            value=st.session_state.get("target_col", ""),
            placeholder="e.g. Class, price, churn ...",
            help="Type the exact column name to predict.",
        ) or None
        if not target_col:
            st.warning(" Type your target column name before running either pipeline.")

    st.divider()

    # ── Two pipeline tabs ─────────────────────────────────────────────────────
    tab_tabular, tab_forecast = st.tabs(["Tabular Pipeline", "Forecasting Pipeline"])

    # ── TAB 1: Tabular (/v1/train/tabular) ───────────────────────────────────
    with tab_tabular:
        st.caption("Endpoint: `POST /v1/train/tabular`  ·  Accepts: regression, classification")

        col_a, col_b = st.columns(2)

        with col_a:
            tabular_prob = st.radio(
                "Problem Type",
                options=["regression", "classification"],
                horizontal=True,
                key="tabular_prob",
                help="regression → predicts a continuous number  |  classification → predicts a class label",
            )

        with col_b:
            with st.expander("Advanced options"):
                st.text_input(
                    "User ID",
                    value="550e8400-e29b-41d4-a716-446655440000",
                    key="tabular_user_id",
                )
                st.number_input(
                    "Inference preview rows",
                    min_value=10, max_value=1000, value=100, step=10,
                    key="tabular_sample_n",
                )

        if not target_col:
            st.warning("Select a target column above before running.")
        else:
            if st.button("▶ Run Tabular Pipeline", type="primary", key="btn_tabular"):
                # Write the widget value into session_state BEFORE dispatch
                # so session_state never holds a stale prob_type from a previous run.
                st.session_state["prob_type"]  = tabular_prob
                st.session_state["target_col"] = target_col
                dispatch_training("/v1/train/tabular", tabular_prob, target_col)

        # Confirmation banner — shows exactly what was sent to the API
        if st.session_state.get("prob_type") and "train_job_id" in st.session_state:
            st.info(
                f"Last run sent → `problem_type: {st.session_state['prob_type']}`  "
                f"·  `target_column: {st.session_state.get('target_col', '—')}`  "
                f"·  endpoint: `/v1/train/tabular`"
            )

        show_metrics()

    # ── TAB 2: Forecasting (/v1/train/forecast) ───────────────────────────────
    with tab_forecast:
        st.caption("Endpoint: `POST /v1/train/forecast`  ·  Accepts: forecasting only")

        col_c, col_d = st.columns(2)

        with col_c:
            st.radio(
                "Problem Type",
                options=["forecasting"],
                disabled=True,
                horizontal=True,
                key="forecast_prob",
                help="The forecasting endpoint only accepts problem_type='forecasting'.",
            )

            date_col = st.selectbox(
                "Date / Time Column",
                options=["— none —"] + columns,
                key="forecast_date_col",
                help="Column used as the time axis. Passed to the pipeline as metadata.",
            )

            forecast_horizon = st.number_input(
                "Forecast Horizon (steps)",
                min_value=1, max_value=365, value=30,
                key="forecast_horizon",
                help="How many future steps the model should predict.",
            )

        with col_d:
            with st.expander("Advanced options"):
                st.text_input(
                    "User ID",
                    value="550e8400-e29b-41d4-a716-446655440000",
                    key="forecast_user_id",
                )
                st.number_input(
                    "Inference preview rows",
                    min_value=10, max_value=1000, value=100, step=10,
                    key="forecast_sample_n",
                )

        if not target_col:
            st.warning("Select a target column above before running.")
        else:
            if st.button("▶ Run Forecasting Pipeline", type="primary", key="btn_forecast"):
                st.session_state["prob_type"]  = "forecasting"
                st.session_state["target_col"] = target_col
                dispatch_training("/v1/train/forecast", "forecasting", target_col)

        if st.session_state.get("prob_type") == "forecasting" and "train_job_id" in st.session_state:
            st.info(
                f"Last run sent → `problem_type: forecasting`  "
                f"·  `target_column: {st.session_state.get('target_col', '—')}`  "
                f"·  endpoint: `/v1/train/forecast`"
            )

        show_metrics()


# ── 3. INSIGHTS & VISUALIZATION ──────────────────────────────────────────────
st.divider()
st.header("3. Insights & Visualization")

if uploaded_file:
    df_preview = load_preview(uploaded_file)

    if df_preview is not None:
        c1, c2 = st.columns(2)

        with c1:
            st.write("### Feature Distribution")
            dist_col = st.selectbox(
                "Column to plot",
                options=df_preview.columns.tolist(),
                key="dist_col",
            )
            fig_dist = px.histogram(df_preview, x=dist_col, nbins=30)
            st.plotly_chart(fig_dist, width="stretch")

        with c2:
            st.write("### Target Trend")
            default_trend = (
                st.session_state.get("target_col")
                if st.session_state.get("target_col") in df_preview.columns
                else df_preview.columns[0]
            )
            trend_col = st.selectbox(
                "Column to trend",
                options=df_preview.columns.tolist(),
                index=df_preview.columns.tolist().index(default_trend),
                key="trend_col",
            )
            fig_trend = px.line(df_preview, y=trend_col)
            st.plotly_chart(fig_trend, width="stretch")


# ── 4. PREDICTION vs ACTUAL ───────────────────────────────────────────────────
st.divider()
st.header("4. Prediction vs Actual")

if "train_status" not in st.session_state:
    st.info("Run a training pipeline first to generate predictions.")
elif uploaded_file is None:
    st.info("Re-upload the dataset to generate predictions.")
else:
    df_pred = load_preview(uploaded_file)

    if df_pred is not None:
        pred_cols = df_pred.columns.tolist()

        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            default_target = st.session_state.get("target_col", pred_cols[0])
            infer_target = st.selectbox(
                "Target Column (actuals)",
                options=pred_cols,
                index=pred_cols.index(default_target) if default_target in pred_cols else 0,
                key="infer_target",
            )

        with pc2:
            infer_pipeline = st.radio(
                "Pipeline Type",
                options=["tabular", "forecasting"],
                index=1 if st.session_state.get("prob_type") == "forecasting" else 0,
                horizontal=True,
                key="infer_pipeline",
                help="Must match the pipeline used during training.",
            )

        with pc3:
            infer_n = st.number_input(
                "Sample rows",
                min_value=10, max_value=len(df_pred), value=min(100, len(df_pred)),
                key="infer_n",
            )

        if st.button("▶ Fetch Predictions", type="primary", key="btn_infer"):
            sample = df_pred.head(int(infer_n))

            # Read prob_type from the widget key (current selection),
            # NOT from session_state which may hold a value from a previous run.
            current_prob = st.session_state.get("prob_type", "regression")
            infer_payload = {
                "model_name":    f"{current_prob}_v1",
                "features":      sample.drop(columns=[infer_target], errors="ignore")
                                       .to_dict(orient="records"),
                "request_id":    str(uuid.uuid4()),
                "pipeline_type": infer_pipeline,
                "trace":         {"trace_id": str(uuid.uuid4())},
            }
            st.caption(f"Sending → `model_name: {current_prob}_v1`  ·  `pipeline_type: {infer_pipeline}`")

            with st.spinner("Calling /inference/predict …"):
                infer_res = safe_request(
                    "POST", f"{API_BASE_URL}/inference/predict", json=infer_payload
                )

            if infer_res and infer_res.status_code == 200:
                infer_data  = infer_res.json()
                predictions = infer_data.get("prediction", [])
                actuals     = (
                    sample[infer_target].tolist()
                    if infer_target in sample.columns else []
                )

                fig_pred = go.Figure()
                if actuals:
                    fig_pred.add_trace(go.Scatter(y=actuals, name="Actual"))
                if predictions:
                    fig_pred.add_trace(go.Scatter(
                        y=predictions, name="Predicted", line=dict(dash="dash")
                    ))
                fig_pred.update_layout(
                    xaxis_title="Sample index",
                    yaxis_title=infer_target,
                )
                st.plotly_chart(fig_pred, width="stretch")

                ia, ib = st.columns(2)
                ia.metric("Latency (ms)",  infer_data.get("latency_ms",    "—"))
                ib.metric("Model Version", infer_data.get("model_version", "—"))

            elif infer_res:
                st.error(f"Inference failed (HTTP {infer_res.status_code}): {infer_res.text}")