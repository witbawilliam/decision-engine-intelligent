import time
import numpy as np
import streamlit as st
import requests
import pandas as pd


API_BASE_URL = "http://api:8000"


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def safe_request(method, url, **kwargs):
    """
    Safely make an HTTP request to the backend API.
    """
    try:
        return requests.request(
            method,
            url,
            timeout=60,
            **kwargs,
        )
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def read_columns(file) -> list:
    """
    Read dataset columns without loading the entire dataset into memory.
    """
    try:
        file.seek(0)

        if file.name.lower().endswith(".parquet"):
            df = pd.read_parquet(file)

        elif file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(file, nrows=5)

        else:
            df = pd.read_csv(
                file,
                nrows=5,
            )

        return df.columns.tolist()

    except Exception:
        return []


def load_preview(
    file,
    nrows: int = 500,
) -> pd.DataFrame | None:
    """
    Load a small preview of the uploaded dataset.
    """
    try:
        file.seek(0)

        if file.name.lower().endswith(".parquet"):
            df = pd.read_parquet(file)

        elif file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(file)

        else:
            df = pd.read_csv(file)

        return df.head(nrows)

    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING DISPATCH
# ═════════════════════════════════════════════════════════════════════════════

def dispatch_training(
    endpoint: str,
    prob_type: str,
    target_col: str,
    time_col: str | None = None,
    product_col: str | None = None,
) -> None:
    """
    Submit a training job to the backend.

    Tabular:
        target_column only

    Forecasting:
        target_column
        time_column
        product_col
        forecast_horizon
    """

    normalized_prob = prob_type.lower().strip()

    # ─────────────────────────────────────────────────────────────────────────
    # BASE PAYLOAD
    # ─────────────────────────────────────────────────────────────────────────

    payload = {
        "idempotency_key": (
            f"{st.session_state['job_id']}_{normalized_prob}"
        ),
        "user_id": (
            "550e8400-e29b-41d4-a716-446655440000"
        ),
        "filename": (
            st.session_state.get(
                "filename",
                "",
            )
        ),
        "s3_key": (
            st.session_state["job_id"]
        ),
        "problem_type": normalized_prob,
        "target_column": target_col,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # FORECASTING-ONLY FIELDS
    # ─────────────────────────────────────────────────────────────────────────

    if normalized_prob == "forecasting":

        if not time_col:
            st.error(
                "Date / Time Column is required for forecasting."
            )
            return

        if not product_col:
            st.error(
                "Product Column is required for "
                "multi-product forecasting."
            )
            return

        payload["time_column"] = time_col

        payload["product_col"] = product_col

        payload["forecast_horizon"] = (
            st.session_state.get(
                "forecast_horizon",
                7,
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DEBUG INFORMATION
    # ─────────────────────────────────────────────────────────────────────────

    with st.expander(
        "🔧 Training Request",
        expanded=False,
    ):
        st.json(payload)

    # ─────────────────────────────────────────────────────────────────────────
    # SEND REQUEST
    # ─────────────────────────────────────────────────────────────────────────

    train_res = safe_request(
        "POST",
        f"{API_BASE_URL}{endpoint}",
        json=payload,
    )

    if train_res is None:
        st.error(
            "Dispatch failed: no response from training API."
        )
        return

    if train_res.status_code not in (200, 202):
        st.error(
            f"Dispatch failed "
            f"(HTTP {train_res.status_code}): "
            f"{train_res.text}"
        )
        return

    # ─────────────────────────────────────────────────────────────────────────
    # RESPONSE
    # ─────────────────────────────────────────────────────────────────────────

    try:
        response_data = train_res.json()

    except Exception:
        st.error(
            "Training API returned an invalid JSON response."
        )
        return

    raw_job_id = response_data.get(
        "job_id",
        payload["idempotency_key"],
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE TRAINING STATE
    # ─────────────────────────────────────────────────────────────────────────

    st.session_state["train_job_id"] = raw_job_id

    st.session_state["prob_type"] = normalized_prob

    st.session_state["target_col"] = target_col

    # ─────────────────────────────────────────────────────────────────────────
    # FORECASTING STATE
    # ─────────────────────────────────────────────────────────────────────────

    if normalized_prob == "forecasting":

        st.session_state["trained_product_col"] = (
            product_col
        )

        st.session_state["trained_date_col"] = (
            time_col
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TABULAR STATE
    # ─────────────────────────────────────────────────────────────────────────

    else:

        # Very important:
        # Remove old forecasting information.
        #
        # This prevents something like:
        #
        # Regression
        # Target: sales
        # Product Column: marketing_spend
        #
        # from appearing after a previous forecasting run.

        st.session_state.pop(
            "trained_product_col",
            None,
        )

        st.session_state.pop(
            "trained_date_col",
            None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SUCCESS
    # ─────────────────────────────────────────────────────────────────────────

    st.success(
        f"Training job submitted · "
        f"`{normalized_prob}` · "
        f"target: `{target_col}`"
    )

    st.info(
        f"Job ID: `{raw_job_id}`"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Model Training",
    layout="wide",
)

st.title(
    "🤖 AutoML Model Training Dashboard"
)


# ═════════════════════════════════════════════════════════════════════════════
# DATA INGESTION
# ═════════════════════════════════════════════════════════════════════════════

st.header(
    "📂 Data Ingestion"
)

uploaded_file = st.file_uploader(
    "Upload Dataset (CSV, Parquet, Excel)",
    type=[
        "csv",
        "parquet",
        "xlsx",
    ],
)


if uploaded_file:

    file_size_mb = round(
        uploaded_file.size / (1024 * 1024),
        2,
    )

    if st.button(
        "Ingest & Process Dataset",
        type="primary",
        key="btn_ingest",
    ):

        res = safe_request(
            "POST",
            f"{API_BASE_URL}/datasets/upload",
            files=[
                (
                    "files",
                    (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type,
                    ),
                )
            ],
            data=[
                (
                    "user_id",
                    "550e8400-e29b-41d4-a716-446655440000",
                ),
                (
                    "file_types",
                    uploaded_file.name.split(".")[-1],
                ),
                (
                    "file_sizes_mb",
                    file_size_mb,
                ),
            ],
        )

        if res and res.status_code == 201:

            data = res.json()

            # ─────────────────────────────────────────────────────────────────
            # SAVE DATASET STATE
            # ─────────────────────────────────────────────────────────────────

            st.session_state["job_id"] = (
                data["job_id"]
            )

            st.session_state["filename"] = (
                uploaded_file.name
            )

            st.session_state["df_columns"] = (
                read_columns(uploaded_file)
            )

            # New dataset means previous training state
            # should not remain active.

            st.session_state.pop(
                "train_job_id",
                None,
            )

            st.session_state.pop(
                "prob_type",
                None,
            )

            st.session_state.pop(
                "target_col",
                None,
            )

            st.session_state.pop(
                "trained_product_col",
                None,
            )

            st.session_state.pop(
                "trained_date_col",
                None,
            )

            st.success(
                f"Dataset uploaded! · "
                f"Job ID: `{data['job_id']}`"
            )

            # ─────────────────────────────────────────────────────────────────
            # DATASET INFORMATION
            # ─────────────────────────────────────────────────────────────────

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Rows",
                data.get(
                    "rows",
                    "N/A",
                ),
            )

            c2.metric(
                "Columns",
                data.get(
                    "columns",
                    "N/A",
                ),
            )

            c3.write(
                f"**Merged:** "
                f"{data.get('merged', False)}"
            )

            if data.get(
                "rows",
                0,
            ) < 100:

                st.warning(
                    "Low sample size — training results "
                    "may be unreliable."
                )

            if data.get(
                "missing_count",
                0,
            ) > 10:

                st.warning(
                    "High missing data — imputation "
                    "will be applied."
                )

        elif res:

            st.error(
                f"Upload failed "
                f"(HTTP {res.status_code}): "
                f"{res.text}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════

st.divider()

st.header(
    "⚙️ Model Training"
)


if "job_id" not in st.session_state:

    st.info(
        "Upload a dataset first to unlock training."
    )

else:

    columns = st.session_state.get(
        "df_columns",
        [],
    )

    # ═════════════════════════════════════════════════════════════════════════
    # TARGET COLUMN
    # ═════════════════════════════════════════════════════════════════════════

    if columns:

        target_col = st.selectbox(
            "Target Column",
            options=[
                " — select — "
            ] + columns,
            index=0,
            key="training_target_col",
        )

        target_col = (
            None
            if target_col == " — select — "
            else target_col
        )

        if not target_col:

            st.warning(
                "Choose your target column "
                "before running a pipeline."
            )

    else:

        target_col = st.text_input(
            "Target Column Name",
            value="",
            placeholder=(
                "e.g. sales, price, churn, Class …"
            ),
            key="training_target_text",
        ) or None

    st.divider()

    # ═════════════════════════════════════════════════════════════════════════
    # PIPELINE TABS
    # ═════════════════════════════════════════════════════════════════════════

    tab_tabular, tab_forecast = st.tabs(
        [
            "Tabular Training",
            "Multi-Product Forecasting",
        ]
    )

    # ═════════════════════════════════════════════════════════════════════════
    # TABULAR TRAINING
    # ═════════════════════════════════════════════════════════════════════════

    with tab_tabular:

        st.markdown(
            "#### Tabular Model Training"
        )

        st.caption(
            "Train a regression or classification "
            "model on the uploaded dataset."
        )

        tabular_prob = st.radio(
            "Problem Type",
            [
                "regression",
                "classification",
            ],
            horizontal=True,
            key="tabular_prob",
        )

        if target_col:

            if st.button(
                "▶ Train Tabular Model",
                type="primary",
                key="btn_tabular",
            ):

                # ─────────────────────────────────────────────────────────────
                # CLEAR OLD FORECASTING STATE
                # ─────────────────────────────────────────────────────────────

                st.session_state.pop(
                    "trained_product_col",
                    None,
                )

                st.session_state.pop(
                    "trained_date_col",
                    None,
                )

                # ─────────────────────────────────────────────────────────────
                # SUBMIT TABULAR TRAINING
                # ─────────────────────────────────────────────────────────────

                dispatch_training(
                    endpoint="/v1/train/tabular",
                    prob_type=tabular_prob,
                    target_col=target_col,
                )

        else:

            st.warning(
                "Select a target column above "
                "before training."
            )

    # ═════════════════════════════════════════════════════════════════════════
    # MULTI-PRODUCT FORECASTING
    # ═════════════════════════════════════════════════════════════════════════

    with tab_forecast:

        st.markdown(
            "#### Multi-Product Sales Forecasting"
        )

        st.caption(
            "Train a separate forecasting model "
            "for each product in the dataset."
        )

        fc1, fc2, fc3 = st.columns(3)

        # ─────────────────────────────────────────────────────────────────────
        # DATE COLUMN
        # ─────────────────────────────────────────────────────────────────────

        with fc1:

            date_col = st.selectbox(
                "Date / Time Column",
                [
                    "— select —"
                ] + columns,
                key="forecast_date_col",
            )

        # ─────────────────────────────────────────────────────────────────────
        # PRODUCT COLUMN
        # ─────────────────────────────────────────────────────────────────────

        with fc2:

            product_col = st.selectbox(
                "Product Column",
                [
                    "— select —"
                ] + columns,
                key="forecast_product_col",
                help=(
                    "Column containing the product "
                    "identifier, SKU, or product name."
                ),
            )

        # ─────────────────────────────────────────────────────────────────────
        # FORECAST HORIZON
        # ─────────────────────────────────────────────────────────────────────

        with fc3:

            st.number_input(
                "Forecast Horizon (steps)",
                min_value=1,
                max_value=365,
                value=30,
                key="forecast_horizon",
            )

        # ─────────────────────────────────────────────────────────────────────
        # NORMALIZE SELECTED VALUES
        # ─────────────────────────────────────────────────────────────────────

        selected_date_col = (
            None
            if date_col == "— select —"
            else date_col
        )

        selected_product_col = (
            None
            if product_col == "— select —"
            else product_col
        )

        # ─────────────────────────────────────────────────────────────────────
        # FORECASTING VALIDATION
        # ─────────────────────────────────────────────────────────────────────

        if (
            target_col
            and selected_date_col
            and selected_product_col
        ):

            # Target, date, and product must be different.

            if len(
                {
                    target_col,
                    selected_date_col,
                    selected_product_col,
                }
            ) < 3:

                st.error(
                    "Target, Date/Time, and Product "
                    "columns must all be different."
                )

            else:

                # ─────────────────────────────────────────────────────────────
                # PRODUCT COUNT
                # ─────────────────────────────────────────────────────────────

                try:

                    preview = load_preview(
                        uploaded_file,
                        nrows=500,
                    )

                    if (
                        preview is not None
                        and selected_product_col
                        in preview.columns
                    ):

                        n_products = (
                            preview[
                                selected_product_col
                            ]
                            .dropna()
                            .nunique()
                        )

                        st.info(
                            f"Product column: "
                            f"`{selected_product_col}` · "
                            f"{n_products} unique "
                            f"product(s) in preview."
                        )

                except Exception:

                    pass

                # ─────────────────────────────────────────────────────────────
                # FORECASTING BUTTON
                # ─────────────────────────────────────────────────────────────

                if st.button(
                    "▶ Train Multi-Product "
                    "Forecasting Models",
                    type="primary",
                    key="btn_forecast",
                ):

                    dispatch_training(
                        endpoint="/v1/train/forecast",
                        prob_type="forecasting",
                        target_col=target_col,
                        time_col=selected_date_col,
                        product_col=selected_product_col,
                    )

        else:

            st.warning(
                "Select a target column, date/time "
                "column, and product column before "
                "running multi-product forecasting."
            )


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING JOB INFORMATION
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.get("train_job_id"):

    st.divider()

    st.subheader(
        "Training Job"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # JOB ID
    # ─────────────────────────────────────────────────────────────────────────

    st.write(
        f"**Job ID:** "
        f"`{st.session_state['train_job_id']}`"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PROBLEM TYPE
    # ─────────────────────────────────────────────────────────────────────────

    current_prob = st.session_state.get(
        "prob_type",
        "N/A",
    )

    st.write(
        f"**Problem Type:** "
        f"`{current_prob}`"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TARGET
    # ─────────────────────────────────────────────────────────────────────────

    st.write(
        f"**Target:** "
        f"`{st.session_state.get('target_col', 'N/A')}`"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # FORECASTING-ONLY INFORMATION
    # ═════════════════════════════════════════════════════════════════════════

    if current_prob == "forecasting":

        st.write(
            f"**Date / Time Column:** "
            f"`{st.session_state.get('trained_date_col', 'N/A')}`"
        )

        st.write(
            f"**Product Column:** "
            f"`{st.session_state.get('trained_product_col', 'N/A')}`"
        )

        st.write(
            f"**Forecast Horizon:** "
            f"`{st.session_state.get('forecast_horizon', 30)}`"
        )