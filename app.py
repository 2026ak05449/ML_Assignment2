"""Streamlit app: evaluate the saved classifiers on an uploaded test CSV.

Run locally with:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"
CLASS_NAMES = ["No purchase", "Purchase"]

st.set_page_config(page_title="Shopper Intent · Model Lab", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
    .hero {padding: 1.1rem 1.4rem; border-radius: 14px;
           background: linear-gradient(120deg, #0f4c5c 0%, #2c7a7b 55%, #e9c46a 140%); color: #fff;}
    .hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
    .hero p {margin: .35rem 0 0; opacity: .9;}
    div[data-testid="stMetric"] {background: rgba(44,122,123,.08); border: 1px solid rgba(44,122,123,.25);
                                 border-radius: 12px; padding: .6rem .8rem;}
    </style>
    <div class="hero">
      <h1>🛒 Shopper Intent · Model Lab</h1>
      <p>Will an online shopping session end in a purchase? Upload test data and compare five
      pre-trained classifiers on the UCI <i>Online Shoppers Purchasing Intention</i> dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- loading ----------
@st.cache_resource
def load_models():
    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in info["models"].items()}
    return info, models


def to_binary(series):
    mapped = series.astype(str).str.strip().str.lower().map({"true": 1, "1": 1, "false": 0, "0": 0})
    if mapped.isna().any():
        raise ValueError(f"Column '{series.name}' must contain only True/False or 1/0 values.")
    return mapped.astype(int)


def prepare(df, info):
    """Check columns and coerce types so the frame matches what the pipelines were trained on."""
    features = info["numeric_features"] + info["categorical_features"]
    missing = [c for c in features + [info["target"]] if c not in df.columns]
    if missing:
        raise ValueError("Uploaded CSV is missing columns: " + ", ".join(missing))
    df = df.copy()
    df["Weekend"] = to_binary(df["Weekend"])
    y = to_binary(df[info["target"]])
    for col in info["categorical_features"]:
        df[col] = df[col].astype(str)
    return df[features], y


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y, y_pred, scores):
    auc = roc_auc_score(y, scores) if y.nunique() == 2 else np.nan
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


try:
    info, models = load_models()
except FileNotFoundError:
    st.error("Saved models not found in `model/`. Run the training notebook first.")
    st.stop()

# ---------- sidebar ----------
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + Revenue)", type=["csv"])
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None)
    if SAMPLE_CSV.exists():
        st.download_button("⬇️ Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))
    st.caption("Models were trained offline; nothing is retrained here.")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source = f"uploaded file **{uploaded.name}**"
elif use_sample and SAMPLE_CSV.exists():
    raw = pd.read_csv(SAMPLE_CSV)
    source = "bundled **test_data.csv**"
else:
    st.info("👈 Upload a test CSV or tick *Use bundled test_data.csv* to begin.")
    st.stop()

try:
    X, y = prepare(raw, info)
except ValueError as err:
    st.error(str(err))
    st.stop()

st.write(f"Evaluating on {source} — **{len(X):,} rows**, purchase rate **{y.mean():.1%}**")

tab_eval, tab_compare, tab_data = st.tabs(["📊 Selected model", "🏁 Compare all models", "🔎 Data & predictions"])

@st.cache_data(show_spinner="Scoring the five models on this data…")
def score_all(X, _models):
    """Predict once per uploaded file; switching models or tabs then reuses these results."""
    return {name: (mdl.predict(X), positive_scores(mdl, X)) for name, mdl in _models.items()}


all_scores = score_all(X, models)

# ---------- selected model ----------
y_pred, scores = all_scores[model_name]
m = compute_metrics(y, y_pred, scores)

with tab_eval:
    st.subheader(model_name)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 14})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, width="content")
        plt.close(fig)
    with right:
        st.markdown("**ROC curve**")
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, scores)
            ax.plot(fpr, tpr, color="#2c7a7b", lw=2, label=f"AUC = {m['AUC']:.3f}")
            ax.legend(loc="lower right")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        st.pyplot(fig, width="content")
        plt.close(fig)

    st.markdown("**Classification report**")
    report = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), width="stretch")

# ---------- all models ----------
with tab_compare:
    st.subheader("All five models on this data")
    rows = {}
    for name, (pred, sc) in all_scores.items():
        rows[name] = compute_metrics(y, pred, sc)
    table = pd.DataFrame(rows).T
    table.index.name = "ML Model Name"
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#b7e4c7"),
                 width="stretch")

    metric_pick = st.radio("Metric to chart", list(table.columns), index=4, horizontal=True)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    order = table[metric_pick].sort_values()
    ax.barh(order.index, order.values, color="#2c7a7b")
    for i, v in enumerate(order.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(metric_pick)
    st.pyplot(fig)
    plt.close(fig)
    best = table["F1"].idxmax()
    st.success(f"Best F1 on this data: **{best}** ({table.loc[best, 'F1']:.4f})")

# ---------- data ----------
with tab_data:
    preview = raw.copy()
    preview["Predicted"] = np.where(y_pred == 1, "Purchase", "No purchase")
    preview["Correct"] = y_pred == y.values
    st.dataframe(preview.head(200), width="stretch")
    st.download_button("⬇️ Download predictions", preview.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
                       mime="text/csv")
