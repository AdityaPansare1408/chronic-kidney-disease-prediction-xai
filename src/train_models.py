# train_models.py
import os
import argparse
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
import joblib
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import SMOTE
    HAVE_SMOTE = True
except ImportError:
    HAVE_SMOTE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def ensure_dirs(models_out: str, plots_out: str):
    os.makedirs(models_out, exist_ok=True)
    os.makedirs(plots_out, exist_ok=True)

def load_processed(input_dir: str):
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"))
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]
    return X_train, X_test, y_train, y_test

def get_models(random_state=42) -> Dict[str, Any]:
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state
        ),

        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state
        ),

        "DecisionTree": DecisionTreeClassifier(
            random_state=random_state
        ),

        "SVM": SVC(
            probability=True,
            random_state=random_state
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
    }

    return models

def evaluate_model_cv(model, X, y, cv_splits=5, use_smote=False):

    skf = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42
    )

    aucs = []
    f1s = []
    precisions = []
    recalls = []

    for train_idx, val_idx in skf.split(X, y):

        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if use_smote and HAVE_SMOTE:

            sm = SMOTE(random_state=42)

            X_tr_res, y_tr_res = sm.fit_resample(
                X_tr,
                y_tr
            )

        else:

            X_tr_res = X_tr
            y_tr_res = y_tr

        model.fit(X_tr_res, y_tr_res)

        if hasattr(model, "predict_proba"):

            y_proba = model.predict_proba(X_val)[:, 1]

        else:

            y_scores = model.decision_function(X_val)

            y_proba = (
                y_scores - y_scores.min()
            ) / (
                y_scores.max() - y_scores.min()
            )

        y_pred = model.predict(X_val)

        aucs.append(
            roc_auc_score(y_val, y_proba)
        )

        f1s.append(
            f1_score(y_val, y_pred)
        )

        precisions.append(
            precision_score(y_val, y_pred)
        )

        recalls.append(
            recall_score(y_val, y_pred)
        )

    return {
        "roc_auc_mean": np.mean(aucs),
        "roc_auc_std": np.std(aucs),
        "f1_mean": np.mean(f1s),
        "precision_mean": np.mean(precisions),
        "recall_mean": np.mean(recalls),
    }

def plot_model_comparison(metrics_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(9, 6))
    metrics_df = metrics_df.sort_values(by="roc_auc_mean", ascending=False)
    bar_width = 0.35
    x = np.arange(len(metrics_df))
    plt.bar(x - bar_width/2, metrics_df["roc_auc_mean"], width=bar_width, label="ROC-AUC")
    plt.bar(x + bar_width/2, metrics_df["f1_mean"], width=bar_width, label="F1-Score")
    plt.ylabel("Score")
    plt.title("Model Comparison: ROC-AUC vs F1")
    plt.xticks(x, metrics_df["model"], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc_curves(models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, out_path: str):
    plt.figure(figsize=(8, 6))
    for name, model in models_dict.items():
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                y_proba = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                continue
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        except Exception as e:
            logger.warning(f"ROC curve failed for {name}: {e}")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (All Models)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, out_path: str, labels=[0,1]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label", title="Confusion Matrix")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close()

def main(input_dir: str, models_out: str, plots_out: str, use_smote: bool = True):
    ensure_dirs(models_out, plots_out)
    print("Models output:", os.path.abspath(models_out))
    print("Plots output:", os.path.abspath(plots_out))
    X_train, X_test, y_train, y_test = load_processed(input_dir)
    models = get_models()
    results = []
    trained_models = {}

    logger.info(f"Training models: {list(models.keys())}")

    for name, estimator in models.items():
        logger.info(f"Evaluating (CV) {name}...")
        model_clone = clone(estimator)
        metrics = evaluate_model_cv(model_clone, X_train, y_train, cv_splits=5, use_smote=use_smote)
        metrics['model'] = name
        results.append(metrics)

        if use_smote and HAVE_SMOTE:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train

        logger.info(f"Fitting {name} on full training data...")
        model_clone.fit(X_res, y_res)
        trained_models[name] = model_clone
        joblib.dump(model_clone, os.path.join(models_out, f"{name}.joblib"))

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(plots_out, "metrics_summary.csv"), index=False)
    plot_model_comparison(metrics_df, os.path.join(plots_out, "model_comparison.png"))
    best_row = metrics_df.sort_values(by="roc_auc_mean", ascending=False).iloc[0]
    best_name = best_row['model']
    best_model = trained_models[best_name]
    logger.info(f"Best model: {best_name} (ROC-AUC = {best_row['roc_auc_mean']:.4f})")
    joblib.dump(best_model, os.path.join(models_out, "best_model.joblib"))
    plot_roc_curves(trained_models, X_test, y_test, os.path.join(plots_out, "roc_curves.png"))
    y_pred_best = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_best, os.path.join(plots_out, "confusion_matrix_best.png"))
    test_metrics = {
        'model': best_name,
        'test_accuracy': accuracy_score(y_test, y_pred_best),
        'test_precision': precision_score(y_test, y_pred_best),
        'test_recall': recall_score(y_test, y_pred_best),
        'test_f1': f1_score(y_test, y_pred_best),
        'test_roc_auc': roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]) if hasattr(best_model, "predict_proba") else None
    }
    pd.DataFrame([test_metrics]).to_csv(os.path.join(plots_out, "best_model_test_metrics.csv"), index=False)
    joblib.dump(trained_models, os.path.join(models_out, "trained_models.joblib"))
    logger.info("Training and evaluation complete. Artifacts saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple models for CKD project")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser.add_argument(
    "--input_dir",
    type=str,
    default=os.path.join(BASE_DIR, "data", "processed")
    )
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser.add_argument(
    "--models_out",
    type=str,
    default=os.path.join(BASE_DIR, "models")
    )

    parser.add_argument(
    "--plots_out",
    type=str,
    default=os.path.join(BASE_DIR, "outputs", "plots")
    )
    parser.add_argument("--no-smote", dest='no_smote', action='store_true')
    args = parser.parse_args()

    main(input_dir=args.input_dir, models_out=args.models_out, plots_out=args.plots_out, use_smote=not args.no_smote)
