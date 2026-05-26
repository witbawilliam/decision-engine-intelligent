import time

import uuid

import streamlit as st

import requests

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from urllib.parse import quote



API_BASE_URL = "http://api:8000"





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

            st.error("Time Column selection is required for Forecasting.")

            return

        payload["time_column"] = time_col

        payload["forecast_horizon"] = st.session_state.get("forecast_horizon", 30)



    print(f"DEBUG: Dispatching {normalized_prob} for {target_col}")

    train_res = safe_request("POST", f"{API_BASE_URL}{endpoint}", json=payload)



    if train_res is None or train_res.status_code not in (200, 202):

        err = train_res.text if train_res else "No Response"

        st.error(f"Dispatch failed: {err}")

        return



    response_data = train_res.json()

    raw_job_id    = response_data.get("job_id", payload["idempotency_key"])

    task_id       = quote(raw_job_id, safe="")



    # ── FIX 1 ────────────────────────────────────────────────────────────────

    # The training API returns a clean `model_name` alongside the raw job_id.

    # Always prefer model_name for inference; fall back to job_id only if absent.

    # NEVER store the raw S3 path as the model name.

    model_name = response_data.get("model_name") or raw_job_id

    st.session_state["train_model_name"] = model_name   # ← used by inference

    st.session_state["train_job_id"] = response_data.get("model_name") or response_data.get("job_id")  # kept for status polling

    # ─────────────────────────────────────────────────────────────────────────



    st.session_state["prob_type"]  = normalized_prob

    st.session_state["target_col"] = target_col



    st.caption(f"Polling status for `{raw_job_id}` …")

    final = poll_job(task_id)



    if final:

        # ── FIX 2 ────────────────────────────────────────────────────────────

        # poll_job's completed payload may also carry model_name — keep in sync.

        if final.get("model_name"):

            st.session_state["train_model_name"] = final["model_name"]

        # ─────────────────────────────────────────────────────────────────────

        st.session_state["train_status"] = final

        st.success(f"Training complete! · `{normalized_prob}` · target: `{target_col}`")





def show_metrics() -> None:

    if "train_status" not in st.session_state:

        return

    metrics = st.session_state["train_status"].get("metrics", {})

    prob    = st.session_state.get("prob_type", "")



    st.subheader(" Model Performance")

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



    # ── FIX 3 ────────────────────────────────────────────────────────────────

    # Show the actual registry model name, not the raw job_id.

    display_name = st.session_state.get("train_model_name", f"{prob}_v1")

    m3.write(f"**Model:** `{display_name}`")

    # ─────────────────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────────────────────

# APP LAYOUT

# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Decision Intelligence", layout="wide")

st.title(" AutoML Decision Dashboard")



st.header(" Data Ingestion")



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





st.divider()

st.header(" Model Execution & Evaluation")



if "job_id" not in st.session_state:

    st.info("Upload a dataset first to unlock training.")

else:

    columns = st.session_state.get("df_columns", [])



    st.subheader(" Shared Configuration")



    if columns:

        target_col = st.selectbox(

            "Target Column",

            options=[" select a target column "] + columns,

            index=0,

            help="The column your model will learn to predict. Applies to both pipelines.",

        )

        if target_col == " select a target column ":

            target_col = None

            st.warning(" Choose your target column before running either pipeline.")

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



    tab_tabular, tab_forecast = st.tabs(["Tabular Pipeline", "Forecasting Pipeline"])



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

            if st.button(" Run Tabular Pipeline", type="primary", key="btn_tabular"):

                st.session_state["prob_type"]  = tabular_prob

                st.session_state["target_col"] = target_col

                dispatch_training("/v1/train/tabular", tabular_prob, target_col, None)



        if st.session_state.get("prob_type") and "train_job_id" in st.session_state:

            st.info(

                f"Last run sent → `problem_type: {st.session_state['prob_type']}`  "

                f"·  `target_column: {st.session_state.get('target_col', '—')}`  "

                f"·  endpoint: `/v1/train/tabular`"

            )



        show_metrics()



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

                options=["- select -"] + columns,

                key="forecast_date_col",

                help="Column used as the time axis.",

            )



            forecast_horizon = st.number_input(

                "Forecast Horizon (steps)",

                min_value=1, max_value=365, value=30,

                key="forecast_horizon",

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

        elif date_col == "- select -":

            st.warning("Select a Date / Time Column before running the forecasting pipeline.")

        else:

            if st.button("Run Forecasting Pipeline", type="primary", key="btn_forecast"):

                st.session_state["prob_type"]  = "forecasting"

                st.session_state["target_col"] = target_col

                dispatch_training("/v1/train/forecast", "forecasting", target_col, time_col=date_col)



        if st.session_state.get("prob_type") == "forecasting" and "train_job_id" in st.session_state:

            st.info(

                f"Last run sent → `problem_type: forecasting`  "

                f"·  `target_column: {st.session_state.get('target_col', '—')}`  "

                f"·  endpoint: `/v1/train/forecast`"

            )



        show_metrics()





# INSIGHTS & VISUALIZATION

st.divider()

st.header(" Insights & Visualization")



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





# PREDICTION vs ACTUAL

st.divider()

st.header(" Prediction vs Actual")



if "train_status" not in st.session_state:

    st.info("Run a training pipeline first to generate predictions.")

elif uploaded_file is None:

    st.info("Re-upload the dataset to generate predictions.")

else:

    df_pred = load_preview(uploaded_file)



    if df_pred is not None:

        pred_cols = df_pred.columns.tolist()



        pc1, pc2, pc3, pc4 = st.columns(4)



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

                min_value=1, max_value=len(df_pred), value=min(10, len(df_pred)),

                key="infer_n",

            )



        with pc4:

            # ── FIX 4 ────────────────────────────────────────────────────────

            # Let the user enable explanations (manifold, sensitivity,

            # counterfactuals) from the UI rather than hard-coding False.

            include_explanations = st.toggle(

                "Include Explanations",

                value=False,

                key="include_explanations",

                help="Runs manifold guard, sensitivity analysis, and counterfactuals. Slower.",

            )

            # ─────────────────────────────────────────────────────────────────



        if st.button(" Fetch Predictions", type="primary", key="btn_infer"):



            sample = df_pred.head(int(infer_n))



            # ── FIX 5 ────────────────────────────────────────────────────────

            # Use the clean model_name stored by dispatch_training, NOT the

            # raw S3 job_id.  Fall back gracefully if session is stale.

            model_to_call = st.session_state.get("train_model_name")



            if not model_to_call:

                # Absolute last resort — construct a sane default name

                prob = st.session_state.get("prob_type", "regression")

                train_status = st.session_state.get("train_status", {})

                model_to_call = train_status.get("model_name") or st.session_state.get("train_job_id")  

                st.warning(

                    f"No trained model name found in session — using fallback `{model_to_call}`. "

                    "Re-run training if this is unexpected."

                )

            # ─────────────────────────────────────────────────────────────────



            features_df = sample.drop(columns=[infer_target], errors="ignore").fillna(0)



            # ── FIX 6 ────────────────────────────────────────────────────────

            # Send all sampled rows as a list, not just row[0].

            # The service accepts a single feature dict; loop over rows so the

            # user sees predictions for every sample, not just the first one.

            # ─────────────────────────────────────────────────────────────────

            rows = features_df.to_dict(orient="records")



            st.caption(

                f"Sending → model: `{model_to_call}` | pipeline: `{infer_pipeline.lower()}` | rows: {len(rows)}"

            )



            predictions = []

            actuals     = []

            latencies   = []



            progress_bar = st.progress(0, text="Running inference…")



            for i, row in enumerate(rows):

                infer_payload = {

                    "model_name":          model_to_call,

                    "features":            row,

                    "request_id":          str(uuid.uuid4()),

                    "include_explanations": include_explanations,

                    "trace": {

                        "trace_id": str(uuid.uuid4())

                    },

                }



                infer_res = safe_request(

                    "POST",

                    f"{API_BASE_URL}/inference/predict",

                    json=infer_payload,

                )



                progress_bar.progress((i + 1) / len(rows), text=f"Row {i+1}/{len(rows)}")



                if infer_res and infer_res.status_code == 200:

                    d = infer_res.json()

                    predictions.append(d.get("prediction"))

                    latencies.append(d.get("latency_ms", 0))



                    # ── FIX 7 ────────────────────────────────────────────────

                    # Display explanation results (manifold risk, sensitivity,

                    # counterfactuals) when include_explanations is True.

                    if include_explanations and d.get("explanations"):

                        with st.expander(f"🔍 Explanations — row {i+1}"):

                            exp = d["explanations"]

                            if exp.get("sensitivity"):

                                st.write("**Sensitivity Analysis**")

                                st.json(exp["sensitivity"])

                            if exp.get("counterfactual"):

                                st.write("**Counterfactual**")

                                st.json(exp["counterfactual"])

                    if d.get("risk_score") is not None:

                        risk = d["risk_score"]

                        if risk > 0.7:

                            st.warning(f"Row {i+1}: High OOD risk score `{risk:.2f}` — prediction may be unreliable.")

                    # ─────────────────────────────────────────────────────────

                else:

                    predictions.append(None)

                    latencies.append(0)

                    if infer_res:

                        st.error(f"Row {i+1} failed (HTTP {infer_res.status_code}): {infer_res.text}")



                actuals.append(

                    sample[infer_target].iloc[i] if infer_target in sample.columns else None

                )



            progress_bar.empty()



            # ── FIX 8 ────────────────────────────────────────────────────────

            # Plot all rows, not just index 0.

            # ─────────────────────────────────────────────────────────────────

            fig = go.Figure()



            valid_actuals = [a for a in actuals if a is not None]

            valid_preds   = [p for p in predictions if p is not None]



            if valid_actuals:

                fig.add_trace(go.Scatter(

                    y=valid_actuals,

                    name="Actual",

                    mode="markers+lines",

                ))



            if valid_preds:

                fig.add_trace(go.Scatter(

                    y=valid_preds,

                    name="Predicted",

                    mode="markers+lines",

                    line=dict(dash="dash"),

                ))



            fig.update_layout(

                title=f"Prediction vs Actual ({len(rows)} rows)",

                xaxis_title="Sample Index",

                yaxis_title=infer_target,

            )



            st.plotly_chart(fig, use_container_width=True)



            # Summary metrics

            col1, col2, col3 = st.columns(3)

            col1.metric("Predictions returned", sum(1 for p in predictions if p is not None))

            col2.metric("Avg Latency (ms)", round(sum(latencies) / len(latencies), 1) if latencies else "—")

            col3.metric("Model", model_to_call)



            if all(p is not None for p in predictions):

                st.success(f"All {len(rows)} predictions completed successfully!")

            else:

                failed = sum(1 for p in predictions if p is None)

                st.warning(f"{failed}/{len(rows)} predictions failed.")