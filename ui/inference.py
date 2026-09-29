"""
ui/forecast_inference_page.py

Standalone Streamlit page for CSV-driven forecast inference.

Flow:
    1. Upload a CSV containing a column of product identifiers.
    2. The file is pushed to S3 via the existing POST /datasets/upload
       endpoint (its response `job_id` IS the S3 key).
    3. POST /v1/forecast/batch-from-csv downloads that CSV, forecasts each
       product's already-trained, production-stage model, and returns
       results synchronously.

This is intentionally a separate file rather than a patch to the large
existing ui/dashboard.py -- run it standalone during review, then either
fold it in as a new st.tabs() section of dashboard.py or run it as its
own Streamlit service. See the accompanying explanation for both options.

Run standalone:
    streamlit run ui/forecast_inference_page.py --server.port=8502
"""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

_DEFAULT_USER_ID = "550e8400-e29b-41d4-a716-446655440000"


def safe_request(method: str, url: str, **kwargs):
    """Safely make an HTTP request to the backend API."""
    try:
        return requests.request(method, url, timeout=120, **kwargs)
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


st.set_page_config(page_title="Forecast Inference", layout="wide")
st.title("🔮 Batch Forecast Inference")

st.caption(
    "Upload a CSV listing the products you want forecasts for. The file is "
    "stored in S3, then the API downloads it and runs forecast inference "
    "for each listed product, using that product's already-trained, "
    "production-stage model."
)

# ═════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD PRODUCT LIST CSV
# ═════════════════════════════════════════════════════════════════════════

st.header("1. Upload product list")

uploaded_file = st.file_uploader(
    "CSV containing a product identifier column",
    type=["csv"],
    key="forecast_csv_uploader",
)

if uploaded_file:
    try:
        preview_df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        preview_df = None

    if preview_df is not None:
        st.dataframe(preview_df.head(10))

        product_id_column = st.selectbox(
            "Which column identifies the product?",
            options=preview_df.columns.tolist(),
            key="forecast_product_id_column",
        )

        n_unique = preview_df[product_id_column].nunique()
        st.caption(f"{n_unique} unique value(s) detected in `{product_id_column}`.")

        if st.button("⬆ Upload to S3", type="primary", key="btn_upload_forecast_csv"):
            file_size_mb = round(uploaded_file.size / (1024 * 1024), 2)

            res = safe_request(
                "POST",
                f"{API_BASE_URL}/datasets/upload",
                files=[("files", (uploaded_file.name, uploaded_file, "text/csv"))],
                data=[
                    ("user_id", _DEFAULT_USER_ID),
                    ("file_types", "csv"),
                    ("file_sizes_mb", file_size_mb),
                ],
            )

            if res and res.status_code == 201:
                data = res.json()
                # UploadResponse.job_id IS the S3 key the file was written to.
                st.session_state["forecast_s3_key"] = data["job_id"]
                # NOTE: do NOT also write st.session_state["forecast_product_id_column"]
                # here -- that key already belongs to the selectbox widget above
                # (key="forecast_product_id_column"). Streamlit forbids manually
                # overwriting a session_state slot that a widget owns; the widget
                # already keeps its current value there on every rerun for free.
                st.session_state.pop("forecast_results", None)  # clear stale results
                st.success(f"Uploaded. S3 key: `{data['job_id']}`")
            elif res:
                st.error(f"Upload failed (HTTP {res.status_code}): {res.text}")

st.divider()

# ═════════════════════════════════════════════════════════════════════════
# STEP 2 — RUN FORECAST INFERENCE
# ═════════════════════════════════════════════════════════════════════════

st.header("2. Run forecast inference")

if "forecast_s3_key" not in st.session_state:
    st.info("Upload a product list CSV above first.")
else:
    st.write(f"**S3 key:** `{st.session_state['forecast_s3_key']}`")
    st.write(f"**Product ID column:** `{st.session_state['forecast_product_id_column']}`")

    periods = st.number_input(
        "Forecast horizon (periods)",
        min_value=1,
        max_value=365,
        value=30,
        key="forecast_periods",
    )

    if st.button("▶ Run Forecast Inference", type="primary", key="btn_run_forecast"):
        with st.spinner("Running forecast inference for each product…"):
            res = safe_request(
                "POST",
                f"{API_BASE_URL}/v1/forecast/batch-from-csv",
                json={
                    "s3_key": st.session_state["forecast_s3_key"],
                    "product_id_column": st.session_state["forecast_product_id_column"],
                    "periods": int(periods),
                },
            )

        if res is not None:
            if res.status_code != 200:
                st.error(f"Forecast run failed (HTTP {res.status_code}): {res.text}")
            else:
                st.session_state["forecast_results"] = res.json()

st.divider()

# ═════════════════════════════════════════════════════════════════════════
# STEP 3 — RESULTS
# ═════════════════════════════════════════════════════════════════════════

if "forecast_results" in st.session_state:
    st.header("3. Results")

    data = st.session_state["forecast_results"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total products", data["total_products"])
    c2.metric("Succeeded", data["succeeded"])
    c3.metric("Failed", data["failed"])

    if data["failed"] > 0:
        st.warning(
            f"{data['failed']} product(s) failed to forecast — expand each "
            "below for the specific error (e.g. no production model registered "
            "for that product_id)."
        )

    for result in data["results"]:
        icon = "✅" if result["status"] == "success" else "❌"
        with st.expander(f"{icon} {result['product_id']} — {result['status']}"):
            if result["status"] == "success":
                st.write(
                    f"Model: `{result['model_name']}` "
                    f"v{result['model_version']} · "
                    f"{result['periods']} period(s)"
                )
                forecast_df = pd.DataFrame(result["forecast"])
                st.dataframe(forecast_df)

                # Best-effort chart -- Prophet-style forecast frames expose
                # 'yhat'; skip the chart gracefully if the shape differs.
                if "yhat" in forecast_df.columns and len(forecast_df.columns) > 0:
                    try:
                        st.line_chart(
                            forecast_df.set_index(forecast_df.columns[0])["yhat"]
                        )
                    except Exception:
                        pass
            else:
                st.error(result["error"])