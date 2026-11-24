"""
Streamlit demo: AI-attribution classifier for workforce events.

Run from the project root (after executing both notebooks):
    streamlit run app.py

The app loads models/final_pipeline.joblib (saved by Part2_Modelling_and_Evaluation.ipynb),
builds the same features as training with src/attribution_features.py, and shows the predicted
attribution category with the model's class confidences.
"""
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from attribution_features import (EXECUTION_OPTIONS, JOB_FAMILIES, KIND_OPTIONS,  # noqa: E402
                                  engineer_features)

MODEL_PATH = ROOT / "models" / "final_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
CLASSES = ["explicit", "mixed", "blamed"]
CLASS_COLORS = {"explicit": "#2a78d6", "mixed": "#eb6834", "blamed": "#1baf7a"}
CLASS_MEANING = {
    "explicit": "The employer itself declared the cut AI-related.",
    "mixed": "The employer cited AI among other reasons.",
    "blamed": "The employer named no AI reason; press, unions or analysts linked it to AI.",
}
EXAMPLES = {
    "(empty)": {},
    "CEO statement about AI replacing roles": {
        "company": "Example Software Inc", "sector": "Software / SaaS", "country": "United States",
        "claim": "In a blog post the CEO wrote that AI agents now handle most customer support tickets, so the "
                 "company is eliminating 400 support roles: \"AI lets us do this work with far fewer people.\"",
        "source_title": "Example Software blog: Building an AI-first company", "headcount": 400,
        "pct": "about 8% of staff", "families": ["Customer service and support"]},
    "Results announcement: cost cuts incl. AI": {
        "company": "Example Bank", "sector": "Banking", "country": "United Kingdom",
        "claim": "The bank said the cuts are part of a cost-reduction and restructuring programme to improve "
                 "efficiency, citing simplification, automation and the use of AI among the drivers.",
        "source_title": "Example Bank Q2 results announcement", "headcount": 1200, "pct": "",
        "families": ["Operations and back office", "Corporate functions"]},
    "Union / press links cuts to AI": {
        "company": "Example Retail", "sector": "Retail", "country": "Germany",
        "claim": "According to the union, the job losses are linked to the rollout of automated systems; people "
                 "familiar with the plan told reporters AI tools will take over the work.",
        "source_title": "Local newspaper report on retail job cuts", "headcount": 300, "pct": "", "families": []},
}

st.set_page_config(page_title="AI Attribution Classifier", page_icon="🧭", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    return model, meta


st.title("AI-attribution classifier for workforce events")
st.caption("DSTT term project · Ayesha Naeem (26L-8007) · Muhammad Mazhar (26L-8027)")
st.warning("**Disclaimer.** The model predicts which *attribution category* the AI Layoffs Register would most "
           "likely assign, based on how an event is reported. It is **not** evidence that AI caused any job losses.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `Part1_Data_Preparation_and_EDA.ipynb` and then "
             "`Part2_Modelling_and_Evaluation.ipynb` first; Part 2 saves `models/final_pipeline.joblib`.")
    st.stop()

model, meta = load_model()

with st.expander("About the model"):
    if meta:
        st.markdown(
            f"- **Model:** {meta.get('model')} on {meta.get('feature_configuration', '').lower()} features "
            f"(text + context), trained on {meta.get('trained_on_events')} events\n"
            f"- **Cross-validated macro-F1:** {meta.get('cv_macro_f1')}\n"
            f"- **Held-out test macro-F1:** {meta.get('test_metrics_train_only_model', {}).get('macro_f1')} "
            f"(95% CI {meta.get('test_macro_f1_95ci')}); majority-class reference: "
            f"{meta.get('majority_baseline_test_macro_f1')}\n"
            "- Confidences are the forest's raw class probabilities: useful as a *relative* signal, "
            "not calibrated probabilities.")
    for c in CLASSES:
        st.markdown(f"- **{c}**: {CLASS_MEANING[c]}")

example_name = st.selectbox("Load an example (optional)", list(EXAMPLES))
ex = EXAMPLES[example_name]

with st.form("event"):
    claim = st.text_area("Claim / description of the event *", value=ex.get("claim", ""), height=150,
                         help="What was said about the cut and by whom. Quotes help.")
    source_title = st.text_input("Source title", value=ex.get("source_title", ""),
                                 help="e.g. 'Company X Q3 results', 'Reuters: ...', 'CEO blog post'")
    c1, c2 = st.columns(2)
    company = c1.text_input("Company name", value=ex.get("company", ""),
                            help="Used only to mask the name in the text, as in training.")
    event_date = c2.date_input("Event date", value=date.today())
    sector = c1.text_input("Sector", value=ex.get("sector", ""), help="Free text, e.g. 'Banking', 'Software / SaaS'")
    country = c2.text_input("Country", value=ex.get("country", ""), help="e.g. 'United States', 'Global'")
    kind = c1.selectbox("Event kind", KIND_OPTIONS)
    execution = c2.selectbox("Execution status", EXECUTION_OPTIONS)
    headcount = c1.number_input("Roles affected (0 = unknown)", min_value=0, step=10,
                                value=int(ex.get("headcount", 0)))
    pct = c2.text_input("% of workforce (free text, optional)", value=ex.get("pct", ""),
                        help="e.g. 'about 5%', '10-15%', 'a third'")
    families = st.multiselect("Job families affected (optional)", JOB_FAMILIES, default=ex.get("families", []))
    submitted = st.form_submit_button("Classify", type="primary")

if submitted:
    if len(claim.split()) < 5:
        st.error("Please enter a claim of at least a few words; the text is the most important input.")
        st.stop()
    event = pd.DataFrame([{
        "company": company, "date": pd.Timestamp(event_date), "claim": claim, "source_title": source_title,
        "sector": sector, "country": country or "global", "kind": kind, "execution": execution,
        "headcount": headcount if headcount > 0 else None, "pct_of_workforce": pct,
        "job_families": "; ".join(families),
    }])
    X = engineer_features(event)
    proba = pd.Series(model.predict_proba(X)[0], index=model.classes_).reindex(CLASSES)
    pred = proba.idxmax()

    st.subheader(f"Predicted category: :{'blue' if pred == 'explicit' else 'orange' if pred == 'mixed' else 'green'}[{pred}]")
    st.write(CLASS_MEANING[pred])
    margin = proba.sort_values(ascending=False)
    if margin.iloc[0] - margin.iloc[1] < 0.10:
        st.info(f"Low margin: the model is almost undecided between **{margin.index[0]}** and "
                f"**{margin.index[1]}**. Treat this as uncertain and review it manually.")

    st.markdown("**Model confidence per category**")
    for c in CLASSES:
        st.progress(float(proba[c]), text=f"{c}: {proba[c]:.0%}")

    with st.expander("Features the model saw"):
        st.write("Cleaned, company-masked text:", X.loc[0, "text_clean"])
        st.dataframe(X[["sector_group", "region", "kind", "execution", "log_headcount", "pct_workforce",
                        "ai_term_count", "business_reason_count", "third_party_count", "ai_negation_count",
                        "source_is_company", "source_is_filing"]].T.rename(columns={0: "value"})
                     .astype(str))   # mixed types in one column -> show as text
