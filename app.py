import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from ucimlrepo import fetch_ucirepo

st.set_page_config(page_title="Session2Sale · Conversion Intelligence", page_icon="🛍️", layout="wide")

# A restrained, editorial dashboard style: warm background, dark ink, and one muted accent.
st.markdown("""
<style>
:root { --ink: #202923; --muted: #68736b; --line: #e3e7df; --paper: #fbfbf8; --accent: #587663; }
.stApp { background: var(--paper); color: var(--ink); }
.block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #f1f3ed; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--ink); }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.035em; }
h1 { font-weight: 680; }
h2, h3 { font-weight: 620; }
p, li, label { color: #39443d; }
.small-kicker { color: var(--accent); text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 750; margin-bottom: .45rem; }
.app-header { padding: 0 0 1.3rem 0; border-bottom: 1px solid var(--line); margin-bottom: 1.6rem; }
.app-header h1 { font-size: 2.45rem; line-height: 1.08; margin: 0 0 .45rem 0; }
.app-subtitle { color: var(--muted); font-size: 1rem; max-width: 760px; margin: 0; }
.header-meta { color: var(--muted); font-size: .78rem; letter-spacing: .02em; padding-top: .35rem; }
.section-note { color: var(--muted); font-size: .93rem; margin-top: -.25rem; margin-bottom: 1rem; }
.panel { background: #ffffff; border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; box-shadow: none; }
div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .8rem; }
div[data-testid="stMetricValue"] { color: var(--ink); font-weight: 650; }
.stButton > button[kind="primary"], .stDownloadButton > button { border-radius: 8px; font-weight: 600; }
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); }
.stButton > button[kind="primary"]:hover { background: #466451; border-color: #466451; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
hr { border-color: var(--line); }
div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; }
footer { visibility: hidden; }
@media (max-width: 700px) {
  .block-container { padding-top: 1rem; }
  .app-header h1 { font-size: 2rem; }
}
</style>
<div class="app-header">
  <div class="small-kicker">E-commerce · Machine learning prototype</div>
  <h1>Session2Sale<span style="color:#587663">.</span></h1>
  <p class="app-subtitle">A small decision-support tool for understanding online shopping sessions and estimating purchase intent.</p>
  <div class="header-meta">UCI Online Shoppers dataset &nbsp;·&nbsp; Random Forest + Logistic Regression</div>
</div>
""", unsafe_allow_html=True)

FEATURE_HELP = {
    "Administrative": "Administrative pages viewed during the session.",
    "Administrative_Duration": "Time spent on administrative pages (seconds).",
    "Informational": "Informational pages viewed.",
    "Informational_Duration": "Time spent on informational pages (seconds).",
    "ProductRelated": "Product-related pages viewed.",
    "ProductRelated_Duration": "Time spent on product pages (seconds).",
    "BounceRates": "Average bounce rate of pages visited.",
    "ExitRates": "Average exit rate of pages visited.",
    "PageValues": "Estimated value of pages visited before transaction.",
    "SpecialDay": "Closeness to a special shopping day (0 to 1).",
    "Month": "Month in which the session occurred.",
    "OperatingSystems": "Operating system category.",
    "Browser": "Browser category.",
    "Region": "Geographic region category.",
    "TrafficType": "Traffic source category.",
    "VisitorType": "Returning or new visitor.",
    "Weekend": "Whether the session happened on a weekend."
}

@st.cache_resource(show_spinner="Loading UCI dataset and training models (first run may take a minute)...")
def build_models():
    # UCI Online Shoppers Purchasing Intention Dataset (ID 468)
    dataset = fetch_ucirepo(id=468)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy().iloc[:, 0]
    # Normalize occasional source-column spelling variations for consistent app controls.
    rename = {
        "ProductRelated": "ProductRelated",
        "ProductRelated_Duration": "ProductRelated_Duration",
    }
    X = X.rename(columns=rename)
    y = y.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Random Forest": Pipeline([("prep", prep), ("model", RandomForestClassifier(
            n_estimators=220, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ))]),
        "Logistic Regression": Pipeline([("prep", prep), ("model", LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ))])
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        trained[name] = {
            "pipeline": model,
            "metrics": {
                "PR-AUC": float(average_precision_score(y_test, prob)),
                "ROC-AUC": float(roc_auc_score(y_test, prob)),
                "Precision": float(precision_score(y_test, pred, zero_division=0)),
                "Recall": float(recall_score(y_test, pred, zero_division=0)),
                "F1": float(f1_score(y_test, pred, zero_division=0)),
            },
            "y_test": y_test,
            "pred": pred,
            "prob": prob
        }
    return X, y, X_train, X_test, y_train, y_test, trained, cat_cols, num_cols

try:
    X, y, X_train, X_test, y_train, y_test, trained, cat_cols, num_cols = build_models()
except Exception as e:
    st.error("The app could not download the public UCI dataset. Check internet access or try again in a moment.")
    st.code(str(e))
    st.markdown("Dataset source: [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)")
    st.stop()

with st.sidebar:
    st.header("Workspace")
    page = st.radio("Choose a view", ["Predict a session", "Batch scoring", "Model evaluation", "Dataset explorer"])
    model_name = st.selectbox("Prediction model", list(trained.keys()), index=0)
    st.caption("Model outputs are estimates, not guarantees. The dataset is observational and does not establish causation.")
    st.divider()
    st.markdown("**Dataset**  \nUCI · 12,330 sessions  \nTarget: `Revenue`")

model = trained[model_name]["pipeline"]

if page == "Predict a session":
    st.markdown('<div class="small-kicker">01 / Individual analysis</div>', unsafe_allow_html=True)
    st.title("Assess a shopping session")
    st.markdown('<p class="section-note">Adjust the session details below. The selected model returns an estimated purchase probability, not a guarantee.</p>', unsafe_allow_html=True)
    defaults = X.median(numeric_only=True).to_dict()
    cat_defaults = {}
    for c in cat_cols:
        modes = X[c].mode(dropna=True)
        cat_defaults[c] = modes.iloc[0] if len(modes) else ""
    input_data = {}
    page_features = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
                     "ProductRelated", "ProductRelated_Duration", "PageValues", "BounceRates", "ExitRates"]
    session_features = ["SpecialDay", "Month", "Weekend", "VisitorType"]
    technical_features = [c for c in X.columns if c not in page_features + session_features]

    def render_feature(col):
        if col in cat_cols:
            choices = sorted(X[col].dropna().astype(str).unique().tolist())
            default = str(cat_defaults.get(col, choices[0] if choices else ""))
            idx = choices.index(default) if default in choices else 0
            input_data[col] = st.selectbox(col, choices, index=idx, help=FEATURE_HELP.get(col, "Dataset feature"))
        elif pd.api.types.is_integer_dtype(X[col]) or col in ["Administrative", "Informational", "ProductRelated", "OperatingSystems", "Browser", "Region", "TrafficType"]:
            lo = int(max(0, np.floor(X[col].min())))
            hi = int(np.ceil(X[col].max()))
            val = int(np.clip(round(float(defaults.get(col, lo))), lo, hi))
            input_data[col] = st.number_input(col, min_value=lo, max_value=hi, value=val, step=1, help=FEATURE_HELP.get(col, "Dataset feature"))
        else:
            lo, hi = float(X[col].min()), float(X[col].max())
            val = float(defaults.get(col, (lo+hi)/2))
            input_data[col] = st.number_input(col, min_value=lo, max_value=max(hi, lo+0.000001), value=float(np.clip(val, lo, hi)), format="%.4f", help=FEATURE_HELP.get(col, "Dataset feature"))

    with st.form("session_prediction_form"):
        st.markdown("#### Page engagement")
        st.caption("How much of the site did this visitor explore?")
        cols = st.columns(3)
        for i, col in enumerate([c for c in page_features if c in X.columns]):
            with cols[i % 3]:
                render_feature(col)

        st.divider()
        st.markdown("#### Session context")
        st.caption("Visitor type, timing and technical context.")
        cols = st.columns(3)
        for i, col in enumerate([c for c in session_features + technical_features if c in X.columns]):
            with cols[i % 3]:
                render_feature(col)

        submitted = st.form_submit_button("Analyse session", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{c: input_data[c] for c in X.columns}])
        probability = float(model.predict_proba(row)[0, 1])
        predicted = int(probability >= 0.5)
        st.markdown("---")
        st.markdown('<div class="small-kicker">Model output</div>', unsafe_allow_html=True)
        st.subheader("Prediction summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated purchase chance", f"{probability:.1%}")
        c2.metric("Classification at 0.50", "Likely to purchase" if predicted else "Unlikely to purchase")
        c3.metric("Model selected", model_name)
        st.progress(probability, text=f"Estimated probability: {probability:.1%}")
        if probability >= 0.70:
            st.success("High estimated intent. A business could review this session for a helpful, low-friction next step.")
        elif probability >= 0.35:
            st.warning("Moderate estimated intent. Review product engagement and possible checkout friction.")
        else:
            st.info("Lower estimated intent. Consider reviewing product discovery and the relevance of the visit.")
        st.caption("Intent bands are illustrative heuristics. This prototype does not send messages or prove that an intervention will increase sales.")

elif page == "Batch scoring":
    st.markdown('<div class="small-kicker">02 / Batch workflow</div>', unsafe_allow_html=True)
    st.title("Score multiple sessions")
    st.markdown('<p class="section-note">Upload a CSV containing the same input columns as the dataset. A template is available below; an optional Revenue column is ignored.</p>', unsafe_allow_html=True)
    template = X.head(0).copy()
    st.download_button("Download input CSV template", template.to_csv(index=False).encode("utf-8"), "session2sale_template.csv", "text/csv")
    upload = st.file_uploader("Upload session CSV", type=["csv"])
    if upload:
        try:
            batch = pd.read_csv(upload)
            missing = [c for c in X.columns if c not in batch.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                features = batch[X.columns].copy()
                probabilities = model.predict_proba(features)[:, 1]
                result = batch.copy()
                result["purchase_probability"] = probabilities.round(4)
                result["predicted_purchase"] = np.where(probabilities >= 0.5, "Yes", "No")
                result["intent_band"] = pd.cut(probabilities, [-0.001, 0.35, 0.70, 1.0], labels=["Low", "Moderate", "High"])
                st.dataframe(result.head(100), use_container_width=True)
                st.metric("Sessions scored", len(result))
                st.download_button("Download scored results", result.to_csv(index=False).encode("utf-8"), "session2sale_scored.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not score this file: {e}")

elif page == "Model evaluation":
    st.markdown('<div class="small-kicker">03 / Model quality</div>', unsafe_allow_html=True)
    st.title("Evaluation, not just a score")
    st.markdown('<p class="section-note">Both models are evaluated on the same held-out test set using metrics suited to an imbalanced classification problem.</p>', unsafe_allow_html=True)
    metric_rows = []
    for name, item in trained.items():
        metric_rows.append({"Model": name, **{k: round(v, 4) for k, v in item["metrics"].items()}})
    st.dataframe(pd.DataFrame(metric_rows).set_index("Model"), use_container_width=True)
    selected = trained[model_name]
    a, b = st.columns(2)
    with a:
        st.markdown("**Confusion matrix (threshold = 0.50)**")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(selected["y_test"], selected["pred"], display_labels=["No purchase", "Purchase"], ax=ax, colorbar=False)
        plt.xticks(rotation=15)
        st.pyplot(fig, clear_figure=True)
    with b:
        st.markdown("**Precision–recall trade-off**")
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(selected["y_test"], selected["prob"])
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall curve")
        ax.grid(alpha=.25)
        st.pyplot(fig, clear_figure=True)
    st.info("Because purchases are the minority class, accuracy alone can be misleading. Compare PR-AUC, precision, recall and F1, and choose a threshold based on the business cost of missed purchases versus unnecessary outreach.")

elif page == "Dataset explorer":
    st.markdown('<div class="small-kicker">04 / Data inspection</div>', unsafe_allow_html=True)
    st.title("Know the data behind the model")
    st.markdown('<p class="section-note">Explore the target distribution, sample rows and relationships among numerical features.</p>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Sessions", f"{len(X):,}")
    b.metric("Features", X.shape[1])
    c.metric("Purchase rate", f"{y.mean():.1%}")
    st.markdown("#### Target distribution")
    st.caption("The outcome is imbalanced: purchase sessions are less common than non-purchase sessions.")
    counts = y.map({0: "No purchase", 1: "Purchase"}).value_counts()
    st.bar_chart(counts)
    st.markdown("**Feature sample**")
    st.dataframe(X.head(15), use_container_width=True)
    st.markdown("**Numeric feature correlations**")
    corr = X.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=70, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    fig.colorbar(im, ax=ax, fraction=.025)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

st.divider()
st.caption("Session2Sale · Student ML prototype | Dataset: Sakar & Kastro, UCI Online Shoppers Purchasing Intention (2018), DOI: 10.24432/C5F88Q. Observational data; predictions are estimates, not guarantees.")
