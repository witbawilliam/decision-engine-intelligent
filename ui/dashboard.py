import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://api:8000"


# -----------------------------
# HEALTH CHECK
# -----------------------------
@st.cache_data(ttl=10)
def check_health():
    try:
        res = requests.get(f"{API_BASE_URL}/health/live", timeout=2)
        return res.status_code == 200
    except:
        return False


def safe_request(method, url, **kwargs):
    try:
        return requests.request(method, url, timeout=60, **kwargs)
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


# -----------------------------
# DECISION ENGINE
# -----------------------------
def generate_recommendation(prediction, explanations):
    recommendations = []

    if isinstance(prediction, (int, float)):
        if prediction < 0:
            recommendations.append("⚠️ Negative trend detected → Investigate business decline")
        elif prediction < 50:
            recommendations.append("📉 Low performance → Increase marketing or optimize pricing")
        else:
            recommendations.append("📈 Strong performance → Scale operations or increase stock")

    risk = explanations.get("manifold_risk")
    if risk and risk > 0.7:
        recommendations.append("⚠️ High model uncertainty → Data may be out-of-distribution")

    if not recommendations:
        recommendations.append("No critical action required")

    return recommendations


# -----------------------------
# FILE TYPE NORMALIZER
# -----------------------------
def get_file_type(filename):
    if filename.endswith(".csv"):
        return "csv"
    elif filename.endswith(".parquet"):
        return "parquet"
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        return "excel"
    return "csv"


# -----------------------------
# UI START
# -----------------------------
st.title("AI Decision Intelligence Platform")

if not check_health():
    st.error("Backend is not alive")
    st.stop()

st.success("System is live")


# -----------------------------
# FILE UPLOAD
# -----------------------------
st.subheader("Upload Dataset")

file = st.file_uploader("Upload CSV / Excel / Parquet")

if file:

    file_size_mb = round(file.size / (1024 * 1024), 2)

    if file_size_mb > 200:
        st.error("File too large. Max 50MB")
        st.stop()

    st.info(f"File size: {file_size_mb} MB")

    # -----------------------------
    # Preview dataset
    # -----------------------------
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            df = None

        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
    except:
        st.warning("Could not preview dataset")

    # -----------------------------
    # UPLOAD DATASET
    # -----------------------------
    st.info("Uploading dataset...")

    file_type = get_file_type(file.name)

    files_payload = [
        ("files", (file.name, file, file.type))
    ]

    data_payload = [
        ("user_id", "user_123"),
        ("file_types", file_type),
        ("file_sizes_mb", str(file_size_mb)),
    ]

    upload_res = safe_request(
        "POST",
        f"{API_BASE_URL}/datasets/upload",
        files=files_payload,
        data=data_payload
    )

    # -----------------------------
    # ERROR HANDLING
    # -----------------------------
    if not upload_res or upload_res.status_code != 201:
        st.error("Upload failed")

        if upload_res is not None:
            st.write("Status Code:", upload_res.status_code)
            try:
                st.json(upload_res.json())
            except:
                st.write(upload_res.text)

        st.stop()

    response_data = upload_res.json()
    job_id = response_data["job_id"]

    st.success("Upload successful")
    st.write("Job ID:", job_id)


    # -----------------------------
    # RUN INFERENCE
    # -----------------------------
    st.subheader("Run Inference")

    if st.button("Run AI Prediction"):
        st.info("Running inference...")

        payload = {
            "features": {},
            "model_name": "default_model",
            "pipeline_type": "tabular",
            "trace": {"trace_id": "streamlit-ui"}
        }

        inference_res = safe_request(
            "POST",
            f"{API_BASE_URL}/inference/predict",
            json=payload
        )

        if not inference_res or inference_res.status_code != 200:
            st.error("Inference failed")
            st.stop()

        data = inference_res.json()

        # -----------------------------
        # RESULTS
        # -----------------------------
        st.subheader("Prediction Results")

        prediction = data.get("prediction")
        st.metric("Prediction", prediction)

        st.write("Model:", data.get("model_version"))
        st.write("Latency (ms):", data.get("latency_ms"))

        st.divider()

        # -----------------------------
        # EXPLANATIONS
        # -----------------------------
        st.subheader("AI Explanations")

        explanations = data.get("explanations", {})

        risk = explanations.get("manifold_risk")
        if risk is not None:
            st.metric("OOD Risk Score", round(risk, 3))

        sensitivity = explanations.get("sensitivity")
        if sensitivity:
            st.bar_chart(sensitivity)

        counterfactual = explanations.get("counterfactual")
        if counterfactual:
            st.json(counterfactual)

        st.divider()

        # -----------------------------
        # DECISION INTELLIGENCE
        # -----------------------------
        st.subheader("🎯 Recommendations")

        recommendations = generate_recommendation(prediction, explanations)

        for rec in recommendations:
            st.write("-", rec)

        st.divider()

        # -----------------------------
        # FEEDBACK LOOP
        # -----------------------------
        st.subheader("Feedback")

        feedback = st.selectbox("Was this useful?", ["Yes", "No"])

        if st.button("Submit Feedback"):
            safe_request(
                "POST",
                f"{API_BASE_URL}/feedback",
                json={"job_id": job_id, "feedback": feedback}
            )
            st.success("Feedback submitted")