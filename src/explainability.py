# explainability.py
import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")
os.makedirs("outputs/shap", exist_ok=True)

def load_artifacts():
    model = joblib.load("models/best_model.joblib")
    scaler = joblib.load("src/artifacts/scaler.joblib")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    return model, scaler, X_train, X_test

def choose_explainer(model, background):
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
        method = "linear"
    elif isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
        explainer = shap.TreeExplainer(model)
        method = "tree"
    else:
        explainer = shap.KernelExplainer(model.predict_proba, background)
        method = "kernel"
    return explainer, method

def get_background_sample(X_train, size=50):
    if len(X_train) > size:
        return X_train.sample(size, random_state=42)
    return X_train

def compute_shap_explanations(explainer, method, X_test):
    if method == "linear":
        shap_vals = explainer.shap_values(X_test)
    elif method == "tree":
        shap_vals = explainer.shap_values(X_test)
    else:
        shap_vals = explainer.shap_values(X_test, nsamples=200)
    if isinstance(shap_vals, list) or (isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3):
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals[1]
    else:
        sv = np.array(shap_vals)
    expected_value = None
    try:
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            expected_value = ev[1] if len(ev) > 1 else ev[0]
        else:
            expected_value = float(ev)
    except Exception:
        try:
            expected_value = float(explainer.expected_value())
        except Exception:
            expected_value = 0.0
    return sv, expected_value

def plot_summary(shap_values, X_test):
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap/shap_summary.png", dpi=300)
    plt.close()

def plot_bar(shap_values, X_test):
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap/shap_bar.png", dpi=300)
    plt.close()

def plot_waterfall(shap_values, base_value, X_test, index=0):
    sample = X_test.iloc[index]
    vals = shap_values[index]
    exp = shap.Explanation(values=vals, base_values=base_value, data=sample, feature_names=X_test.columns)
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    plt.savefig(f"outputs/shap/waterfall_sample_{index}.png", dpi=300)
    plt.close()

def main():
    print("Loading artifacts...")
    model, scaler, X_train, X_test = load_artifacts()
    print("Preparing background sample...")
    background = get_background_sample(X_train)
    print("Selecting explainer based on model type...")
    explainer, method = choose_explainer(model, background)
    print(f"Explainer method: {method}")
    print("Computing SHAP explanations...")
    shap_values, base_value = compute_shap_explanations(explainer, method, X_test)
    print("Generating SHAP summary plot...")
    plot_summary(shap_values, X_test)
    print("Generating SHAP bar plot...")
    plot_bar(shap_values, X_test)
    print("Generating SHAP waterfall for sample index 0...")
    plot_waterfall(shap_values, base_value, X_test, index=0)
    print("SHAP artifacts saved to outputs/shap/")

if __name__ == "__main__":
    main()
