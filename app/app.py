

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
ARTIFACTS_DIR = os.path.join(ROOT, "src", "artifacts")
DATA_DIR = os.path.join(ROOT, "data", "processed")
SHAP_DIR = os.path.join(ROOT, "outputs", "shap")


st.set_page_config(page_title="CKD Prediction — XAI (SHAP)", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    category_maps = joblib.load(os.path.join(ARTIFACTS_DIR, "category_maps.joblib"))
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    feature_names = X_train.columns.tolist()
    return model, scaler, category_maps, X_train, feature_names

model, scaler, category_maps, X_train, FEATURE_NAMES = load_artifacts()


DROPDOWN_MAPS = {
    "rbc": {"Normal": 1, "Abnormal": 0},
    "pc": {"Normal": 0, "Abnormal": 1},
    "pcc": {"Not present": 0, "Present": 1},
    "ba": {"Not present": 0, "Present": 1},
    "htn": {"No": 0, "Yes": 1},
    "dm": {"No": 0, "Yes": 1},
    "cad": {"No": 0, "Yes": 1},
    "appet": {"Good": 1, "Poor": 0},
    "pe": {"No": 0, "Yes": 1},
    "ane": {"No": 0, "Yes": 1},
    
}


SAMPLE_HEALTHY = {
    "age": 35, "bp": 118, "sg": 1.025, "al": 0, "su": 0,
    "bgr": 90, "bu": 20, "sc": 0.8, "sod": 140, "pot": 4.5,
    "hemo": 15.0, "pcv": 45, "wc": 7500, "rc": 5.0, "rbc": 1,
    "pc": 0, "pcc": 0, "ba": 0, "htn": 0, "dm": 0, "cad": 0,
    "appet": 1, "pe": 0, "ane": 0
}

SAMPLE_CKD = {
    "age": 65, "bp": 150, "sg": 1.005, "al": 3, "su": 1,
    "bgr": 180, "bu": 120, "sc": 4.5, "sod": 130, "pot": 5.2,
    "hemo": 8.0, "pcv": 25, "wc": 12000, "rc": 3.0, "rbc": 0,
    "pc": 1, "pcc": 1, "ba": 1, "htn": 1, "dm": 1, "cad": 1,
    "appet": 0, "pe": 1, "ane": 1
}


st.sidebar.title("Patient Input")
st.sidebar.markdown("**Quick samples**")
if st.sidebar.button("Load healthy sample"):
    for k, v in SAMPLE_HEALTHY.items():
        st.session_state[k] = v
if st.sidebar.button("Load CKD-positive sample"):
    for k, v in SAMPLE_CKD.items():
        st.session_state[k] = v

st.sidebar.markdown("---")
st.sidebar.markdown("## Patient Profile")
age = st.sidebar.slider("Age", min_value=1, max_value=100, value=int(st.session_state.get("age", 35)))
bp = st.sidebar.slider("Blood Pressure (mmHg)", min_value=50, max_value=220, value=int(st.session_state.get("bp", 118)))
appet_choice = st.sidebar.selectbox("Appetite", ["Good", "Poor"], index=0 if st.session_state.get("appet", 1) == 1 else 1)

st.sidebar.markdown("---")
st.sidebar.markdown("## Kidney Function")
sg = st.sidebar.slider("Specific Gravity (sg)", min_value=1.000, max_value=1.030, value=float(st.session_state.get("sg", 1.025)), step=0.001, format="%.3f")
al = st.sidebar.slider("Albumin (al) [urine]", min_value=0, max_value=5, value=int(st.session_state.get("al", 0)))
su = st.sidebar.slider("Sugar (su) [urine]", min_value=0, max_value=5, value=int(st.session_state.get("su", 0)))
sc = st.sidebar.slider("Serum creatinine (sc)", min_value=0.5, max_value=15.0, value=float(st.session_state.get("sc", 1.0)), step=0.1)
bu = st.sidebar.slider("Blood urea (bu)", min_value=5, max_value=300, value=int(st.session_state.get("bu", 20)))
pcv = st.sidebar.slider("PCV (%)", min_value=10, max_value=60, value=int(st.session_state.get("pcv", 45)))
hemo = st.sidebar.slider("Hemoglobin (hemo)", min_value=5.0, max_value=20.0, value=float(st.session_state.get("hemo", 15.0)), step=0.1)
rc = st.sidebar.slider("RBC count (rc)", min_value=1.0, max_value=6.0, value=float(st.session_state.get("rc", 5.0)), step=0.1)
rbc_choice = st.sidebar.selectbox("RBC (microscopy)", ["Normal", "Abnormal"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("## Electrolytes")
sod = st.sidebar.slider("Sodium (sod)", min_value=100, max_value=160, value=int(st.session_state.get("sod", 140)))
pot = st.sidebar.slider("Potassium (pot)", min_value=2.0, max_value=8.0, value=float(st.session_state.get("pot", 4.5)), step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("## Comorbidities")
dm_choice = st.sidebar.selectbox("Diabetes (dm)", ["No", "Yes"], index=0)
htn_choice = st.sidebar.selectbox("Hypertension (htn)", ["No", "Yes"], index=0)
cad_choice = st.sidebar.selectbox("Coronary artery disease (cad)", ["No", "Yes"], index=0)
ane_choice = st.sidebar.selectbox("Anemia (ane)", ["No", "Yes"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("## Urine Examination")
pc_choice = st.sidebar.selectbox("Pus cell (pc)", ["Normal", "Abnormal"], index=0)
pcc_choice = st.sidebar.selectbox("Pus cell clumps (pcc)", ["Not present", "Present"], index=0)
ba_choice = st.sidebar.selectbox("Bacteria (ba)", ["Not present", "Present"], index=0)
pe_choice = st.sidebar.selectbox("Pedal edema (pe)", ["No", "Yes"], index=0)

st.sidebar.markdown("---")

wc = st.sidebar.number_input("White blood cell count (wc)", value=int(st.session_state.get("wc", 7500)), step=100)
bgr = st.sidebar.number_input("Blood glucose random (bgr)", value=int(st.session_state.get("bgr", 90)), step=1)


input_raw = {
    "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
    "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
    "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc,
    
    "rbc": None, "pc": None, "pcc": None, "ba": None,
    "htn": None, "dm": None, "cad": None, "appet": None,
    "pe": None, "ane": None
}


input_raw["rbc"] = DROPDOWN_MAPS["rbc"][rbc_choice]
input_raw["pc"] = DROPDOWN_MAPS["pc"][pc_choice]
input_raw["pcc"] = DROPDOWN_MAPS["pcc"][pcc_choice]
input_raw["ba"] = DROPDOWN_MAPS["ba"][ba_choice]
input_raw["htn"] = DROPDOWN_MAPS["htn"][htn_choice]
input_raw["dm"] = DROPDOWN_MAPS["dm"][dm_choice]
input_raw["cad"] = DROPDOWN_MAPS["cad"][cad_choice]
input_raw["appet"] = DROPDOWN_MAPS["appet"][appet_choice]
input_raw["pe"] = DROPDOWN_MAPS["pe"][pe_choice]
input_raw["ane"] = DROPDOWN_MAPS["ane"][ane_choice]


ordered_input = [input_raw[f] for f in FEATURE_NAMES]
input_df = pd.DataFrame([ordered_input], columns=FEATURE_NAMES)


st.title("Explainable CKD Prediction (SHAP)")
tabs = st.tabs(["Prediction", "Local SHAP", "Global SHAP"])


with tabs[0]:
    st.subheader("Input preview")
    st.dataframe(input_df)

   
    invalid = False
    if sc < 0.3 or sc > 20:
        st.warning("Serum creatinine value unusual. Check units.")
        invalid = True
    if bp > 240:
        st.warning("Extremely high blood pressure. Verify input.")
        invalid = True
    if age > 120:
        st.warning("Age seems unrealistic. Verify input.")
        invalid = True

    
    X_scaled = scaler.transform(input_df)
    prob = model.predict_proba(X_scaled)[:, 1][0]
    pred = int(prob >= 0.5)

    
    if pred == 1:
        label = "CKD Positive"
        st.markdown("### 🔴 CKD Positive")
    else:
        label = "CKD Negative"
        st.markdown("### 🟢 CKD Negative")

    
    if prob < 0.3:
        st.success(f"Low Risk — Probability: {prob:.4f}")
    elif prob < 0.7:
        st.warning(f"Moderate Risk — Probability: {prob:.4f}")
    else:
        st.error(f"High Risk — Probability: {prob:.4f}")

    st.write(f"Model raw probability: **{prob:.4f}**")
    st.write(f"Model decision: **{label}**")

   
    download_df = input_df.copy()
    download_df["probability"] = prob
    download_df["prediction"] = pred
    csv = download_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download result CSV", csv, "ckd_prediction.csv", "text/csv")


with tabs[1]:
    st.subheader("Local SHAP Explanation (Waterfall)")

    with st.spinner("Computing SHAP explanation..."):
        #
        try:
            explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_scaled)
            sv = shap_vals if not isinstance(shap_vals, list) else shap_vals[1]
            ev = explainer.expected_value
            base_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
        except Exception:
            bg = X_train.sample(min(50, len(X_train)), random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            shap_vals = explainer.shap_values(X_scaled, nsamples=200)
            sv = shap_vals[1]
            ev = explainer.expected_value
            base_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev

        scaled_row = X_scaled[0]
        exp = shap.Explanation(values=sv[0], base_values=base_value, data=scaled_row, feature_names=FEATURE_NAMES)
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig)

    st.markdown("**Interpretation tips:**")
    st.markdown("- Red / positive bars push the model toward CKD Positive.  Blue / negative bars push toward CKD Negative.")
    st.markdown("- The base value is the expected model output; the waterfall shows how features add/subtract to reach the final score.")


with tabs[2]:
    st.subheader("🌍 Global SHAP (Summary & Importance)")

    col1, col2 = st.columns(2)

    with col1:
        summary_path = os.path.join(SHAP_DIR, "shap_summary.png")
        if os.path.exists(summary_path):
            st.image(summary_path, use_container_width=True, caption="SHAP Summary")
        else:
            st.info("Global SHAP summary not found. Run explainability.py to generate plots.")

    with col2:
        bar_path = os.path.join(SHAP_DIR, "shap_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, use_container_width=True, caption="SHAP Feature Importance (Bar Chart)")
        else:
            st.info("Global SHAP bar plot not found. Run explainability.py to generate plots.")


st.markdown("---")
st.markdown("**Notes:** This app uses the trained Logistic Regression model and StandardScaler saved in the project artifacts. Input values should be raw clinical measurements (not scaled). SHAP explanations use the scaled feature space to remain consistent with the model.")
