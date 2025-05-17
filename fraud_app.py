import streamlit as st
import pandas as pd
import pickle

# === Load Models ===
with open("iso_model.pkl", "rb") as f2:
    iso = pickle.load(f2)

with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features"]

# === Streamlit UI Setup ===
st.set_page_config(page_title="Fraud Checker", page_icon="🛡️", layout="centered")
st.title("🛡️ Fraud Checker")

st.markdown("""
Welcome to **Fraud Checker** — an AI-powered tool to detect potentially fraudulent financial transactions.

Choose how you’d like to check:

- **Quick Check**: Based on Amount + Time
- **Advanced**: For users with full V1–V28 feature data
""")

# === Tabs for Beginner vs Advanced ===
tab1, tab2 = st.tabs([" Quick Check (Time + Amount)", "Advanced (Full Features)"])

# --------------------------------------------------------------------
# TAB 1: Quick Manual + CSV Upload
# --------------------------------------------------------------------
with tab1:
    st.subheader("🔍 Manually Check One Transaction")

    with st.form("manual_check"):
        amount = st.number_input("💵 Transaction Amount ($)", min_value=0.01, value=100.00)
        time = st.number_input("⏱️ Time (in seconds since first transaction)", min_value=0, max_value=200000, value=100000, step=1)
        submitted = st.form_submit_button("🔍 Analyze Transaction")

    if submitted:
        df_manual = pd.DataFrame([[time, amount]], columns=["Time", "Amount"])
        anomaly = iso.predict(df_manual)[0]
        score = iso.decision_function(df_manual)[0]

        if anomaly == -1:
            st.error(f"⚠️ Suspicious Transaction Detected!\n\nAnomaly Score: {round(-score, 3)}")
        else:
            st.success("✅ Transaction looks normal based on amount and time.")

    st.markdown("---")
    st.subheader("📁 Upload a CSV to Scan Multiple Transactions")

    uploaded_file = st.file_uploader("Upload a CSV with 'Time' and 'Amount' columns", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "Time" in df.columns and "Amount" in df.columns:
            preds = iso.predict(df[["Time", "Amount"]])
            df["Result"] = ["⚠️ Fraud" if p == -1 else "✅ Safe" for p in preds]
            st.success(f"Analyzed {len(df)} transactions")
            st.dataframe(df[["Time", "Amount", "Result"]])
        else:
            st.error("❌ Your CSV must have 'Time' and 'Amount' columns.")

# --------------------------------------------------------------------
# TAB 2: Advanced (Full Features V1–V28)
# --------------------------------------------------------------------
with tab2:
    st.subheader("🧠 Paste Full Feature Inputs (V1–V28 + Time & Amount)")

    v_string = st.text_area("Paste 28 comma-separated values for V1 to V28", height=150)
    amount_2 = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.00, key="adv_amt")
    time_2 = st.number_input("Time (sec)", min_value=0, max_value=200000, value=100000, key="adv_time")

    if st.button("Analyze Advanced Transaction"):
        try:
            v_inputs = [float(x.strip()) for x in v_string.split(",")]
            assert len(v_inputs) == 28

            input_data = pd.DataFrame([[time_2] + v_inputs + [amount_2]], columns=feature_names)
            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0][prediction]

            if prediction == 1:
                st.error(f"⚠️ Fraud Detected! Confidence: {round(confidence * 100, 2)}%")
            else:
                st.success(f"✅ Transaction is likely safe. Confidence: {round(confidence * 100, 2)}%")

        except:
            st.error("❌ Please paste exactly 28 valid numbers, separated by commas.")
