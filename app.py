import html
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

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
CLASSES = ["explicit", "mixed", "blamed"]
CLASS_STYLE = {   # same class colours as the notebooks: (main, soft background)
    "explicit": ("#2a78d6", "#eaf2fc"),
    "mixed": ("#eb6834", "#fdf0ea"),
    "blamed": ("#1baf7a", "#e8f7f1"),
}
CLASS_ICON = {"explicit": "🗣️", "mixed": "⚖️", "blamed": "📰"}
CLASS_MEANING = {
    "explicit": "The employer itself declared the cut AI-related.",
    "mixed": "The employer cited AI among other reasons.",
    "blamed": "The employer named no AI reason; press, unions or analysts linked it to AI.",
}
EXAMPLES = {
    "✍️  Write my own event": {},
    "🗣️  CEO statement about AI replacing roles": {
        "company": "Example Software Inc", "sector": "Software / SaaS", "country": "United States",
        "claim": "In a blog post the CEO wrote that AI agents now handle most customer support tickets, so the "
                 "company is eliminating 400 support roles: \"AI lets us do this work with far fewer people.\"",
        "source_title": "Example Software blog: Building an AI-first company", "headcount": 400,
        "pct": "about 8% of staff", "families": ["Customer service and support"]},
    "⚖️  Results announcement: cost cuts incl. AI": {
        "company": "Example Bank", "sector": "Banking", "country": "United Kingdom",
        "claim": "The bank said the cuts are part of a cost-reduction and restructuring programme to improve "
                 "efficiency, citing simplification, automation and the use of AI among the drivers.",
        "source_title": "Example Bank Q2 results announcement", "headcount": 1200, "pct": "",
        "families": ["Operations and back office", "Corporate functions"]},
    "📰  Union / press links cuts to AI": {
        "company": "Example Retail", "sector": "Retail", "country": "Germany",
        "claim": "According to the union, the job losses are linked to the rollout of automated systems; people "
                 "familiar with the plan told reporters AI tools will take over the work.",
        "source_title": "Local newspaper report on retail job cuts", "headcount": 300, "pct": "", "families": []},
}

# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Attribution Classifier", page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, .stSelectbox { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1280px; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { display: none; }
[data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: .55rem; }
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stDateInput input,
[data-baseweb="select"] > div, .stDateInput [data-baseweb="input"], .stDateInput [data-baseweb="input"] > div,
.stMultiSelect [data-baseweb="select"] > div { background: #f4f3ef !important; }

/* header */
.hero { background: linear-gradient(120deg, #1f3a68 0%, #2a78d6 55%, #1baf7a 100%);
        border-radius: 16px; padding: 18px 26px; color: #fff; margin-bottom: 14px;
        display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; }
.hero h1 { font-size: 1.55rem; font-weight: 700; margin: 0; color: #fff; padding: 0; }
.hero p  { margin: 2px 0 0 0; opacity: .9; font-size: .9rem; }
.hero .chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
        padding: 5px 12px; font-size: .8rem; white-space: nowrap; }
.chip b { font-weight: 700; }

/* panels */
.panel-title { font-weight: 700; font-size: 1.02rem; color: #1b1b1b; margin: 0 0 6px 0; }
.panel-sub { color: #6b6a65; font-size: .82rem; margin: -4px 0 10px 0; }
[data-testid="stForm"] { background: #ffffff; border: 1px solid #e6e4df; border-radius: 14px;
                         padding: 18px 20px 8px 20px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
.stTextArea textarea { font-size: .9rem; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    width: 100%; border-radius: 10px; font-weight: 600; padding: .55rem 0;
    background: linear-gradient(90deg, #2a78d6, #1f5fb0); border: 0; color: #fff; }
div[data-testid="stFormSubmitButton"] > button:hover { filter: brightness(1.08); color: #fff; }

/* result side */
.placeholder { border: 2px dashed #d9d7d1; border-radius: 14px; padding: 46px 24px; text-align: center;
               color: #8a8983; background: #fbfbfa; }
.placeholder .big { font-size: 2.4rem; }
.result-card { border-radius: 14px; padding: 18px 20px; margin-bottom: 14px; border: 1px solid; }
.result-label { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: #6b6a65; font-weight: 600; }
.result-class { font-size: 2rem; font-weight: 700; margin: 2px 0 4px 0; line-height: 1.1; }
.result-meaning { color: #3d3c39; font-size: .92rem; }
.conf-box { background: #fff; border: 1px solid #e6e4df; border-radius: 14px; padding: 14px 18px 6px 18px; margin-bottom: 12px; }
.conf-row { margin-bottom: 11px; }
.conf-head { display: flex; justify-content: space-between; font-size: .88rem; margin-bottom: 4px; color: #1b1b1b; }
.conf-head span:last-child { font-weight: 700; }
.bar-bg { background: #efeee9; border-radius: 999px; height: 10px; overflow: hidden; }
.bar-fg { height: 100%; border-radius: 999px; }
.note { border-radius: 10px; padding: 9px 13px; font-size: .84rem; margin-bottom: 10px; }
.note-warn { background: #fff6e0; border: 1px solid #f2d48a; color: #6b4e00; }
.note-ok   { background: #eaf6ee; border: 1px solid #b6dfc3; color: #1d5a31; }
.disclaimer { font-size: .78rem; color: #6b6a65; border-top: 1px solid #e6e4df; padding-top: 8px; margin-top: 4px; }
.signal-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.signal { background: #f6f5f2; border-radius: 10px; padding: 8px 10px; }
.signal .v { font-weight: 700; font-size: 1.05rem; color: #1b1b1b; }
.signal .k { font-size: .72rem; color: #6b6a65; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    return model, meta


if not MODEL_PATH.exists():
    st.error("Model not found. Run `Part1_Data_Preparation_and_EDA.ipynb` and then "
             "`Part2_Modelling_and_Evaluation.ipynb` first; Part 2 saves `models/final_pipeline.joblib`.")
    st.stop()

model, meta = load_model()
test_f1 = meta.get("test_metrics_train_only_model", {}).get("macro_f1")
test_f1 = f"{test_f1:.2f}" if test_f1 is not None else "–"
cv_f1 = f"{meta['cv_macro_f1']:.2f}" if "cv_macro_f1" in meta else "–"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div>
    <h1>🧭 AI-Attribution Classifier</h1>
    <p>How was the AI link of a layoff established? &nbsp;·&nbsp; DSTT term project · Ayesha Naeem &amp; Muhammad Mazhar</p>
  </div>
  <div class="chips">
    <span class="chip">Model: <b>{meta.get('model', 'Random Forest')}</b></span>
    <span class="chip">CV macro-F1: <b>{cv_f1}</b></span>
    <span class="chip">Test macro-F1: <b>{test_f1 if test_f1 is not None else '–'}</b></span>
    <span class="chip">Trained on <b>{meta.get('trained_on_events', 320)}</b> events</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: background information (kept out of the main view)
with st.sidebar:
    st.markdown("### ℹ️ About")
    for c in CLASSES:
        st.markdown(f"**{CLASS_ICON[c]} {c}** — {CLASS_MEANING[c]}")
    st.markdown("---")
    if meta:
        st.markdown(
            f"**Model:** {meta.get('model')} on text + context features  \n"
            f"**Held-out test macro-F1:** {test_f1} (95% CI {meta.get('test_macro_f1_95ci')})  \n"
            f"**Majority-class reference:** {meta.get('majority_baseline_test_macro_f1')}")
    st.caption("Confidences are the forest's raw class probabilities: a relative signal, not calibrated "
               "probabilities. Predictions are not evidence that AI caused any job losses.")

# ---------------------------------------------------------------------------
# Layout: input (left) | result (right)
# ---------------------------------------------------------------------------
left, right = st.columns([1.35, 1], gap="large")

with left:
    st.markdown('<div class="panel-title">📝 Describe the workforce event</div>', unsafe_allow_html=True)
    example_name = st.selectbox("Start from an example", list(EXAMPLES), label_visibility="collapsed")
    ex = EXAMPLES[example_name]

    with st.form("event", border=False):
        claim = st.text_area("Claim / description *", value=ex.get("claim", ""), height=100,
                             placeholder="What was said about the cut, and by whom? Quotes help.")
        s1, s2 = st.columns([2, 1])
        source_title = s1.text_input("Source title", value=ex.get("source_title", ""),
                                     placeholder="e.g. 'Company X Q3 results', 'CEO blog post'")
        company = s2.text_input("Company", value=ex.get("company", ""), placeholder="optional",
                                help="Only used to mask the name in the text, as in training.")
        c1, c2, c3 = st.columns(3)
        sector = c1.text_input("Sector", value=ex.get("sector", ""), placeholder="e.g. Banking")
        country = c2.text_input("Country", value=ex.get("country", ""), placeholder="e.g. United States")
        event_date = c3.date_input("Date", value=date.today())
        c4, c5, c6, c7 = st.columns(4)
        kind = c4.selectbox("Event kind", KIND_OPTIONS)
        execution = c5.selectbox("Execution", EXECUTION_OPTIONS)
        headcount = c6.number_input("Roles affected", min_value=0, step=10, value=int(ex.get("headcount", 0)),
                                    help="0 = unknown")
        pct = c7.text_input("% of workforce", value=ex.get("pct", ""), placeholder="e.g. 5%")
        families = st.multiselect("Job families affected", JOB_FAMILIES, default=ex.get("families", []),
                                  placeholder="Optional")
        submitted = st.form_submit_button("🔍  Classify event", use_container_width=True)

if submitted:
    if len(claim.split()) < 5:
        st.session_state["result"] = {"error": "Please enter a claim of at least a few words; "
                                               "the text is the most important input."}
    else:
        event = pd.DataFrame([{
            "company": company, "date": pd.Timestamp(event_date), "claim": claim, "source_title": source_title,
            "sector": sector, "country": country or "global", "kind": kind, "execution": execution,
            "headcount": headcount if headcount > 0 else None, "pct_of_workforce": pct,
            "job_families": "; ".join(families),
        }])
        X = engineer_features(event)
        proba = pd.Series(model.predict_proba(X)[0], index=model.classes_).reindex(CLASSES)
        st.session_state["result"] = {"proba": proba.to_dict(), "features": X.iloc[0].to_dict()}


def confidence_bars(proba):
    rows = []
    for c in sorted(CLASSES, key=lambda k: -proba[k]):
        color = CLASS_STYLE[c][0]
        rows.append(f"""
        <div class="conf-row">
          <div class="conf-head"><span>{CLASS_ICON[c]} {c}</span><span>{proba[c]:.0%}</span></div>
          <div class="bar-bg"><div class="bar-fg" style="width:{proba[c] * 100:.1f}%; background:{color};"></div></div>
        </div>""")
    return '<div class="conf-box"><div class="result-label" style="margin-bottom:8px;">Model confidence</div>' \
           + "".join(rows) + "</div>"


with right:
    st.markdown('<div class="panel-title">📊 Prediction</div>', unsafe_allow_html=True)
    result = st.session_state.get("result")

    if not result:
        st.markdown("""
        <div class="placeholder">
          <div class="big">🧭</div>
          <div style="font-weight:600; color:#3d3c39; margin-top:6px;">No prediction yet</div>
          <div style="font-size:.88rem; margin-top:4px;">Pick an example or write an event on the left,<br>
          then press <b>Classify event</b>. The result appears here.</div>
        </div>""", unsafe_allow_html=True)
    elif "error" in result:
        st.markdown(f'<div class="note note-warn">⚠️ {result["error"]}</div>', unsafe_allow_html=True)
    else:
        proba = result["proba"]
        feats = result["features"]
        pred = max(proba, key=proba.get)
        main, soft = CLASS_STYLE[pred]
        st.markdown(f"""
        <div class="result-card" style="background:{soft}; border-color:{main}55; border-left:6px solid {main};">
          <div class="result-label">Predicted attribution</div>
          <div class="result-class" style="color:{main};">{CLASS_ICON[pred]} {pred.capitalize()}</div>
          <div class="result-meaning">{CLASS_MEANING[pred]}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(confidence_bars(proba), unsafe_allow_html=True)

        ranked = sorted(proba.values(), reverse=True)
        if ranked[0] - ranked[1] < 0.10:
            second = sorted(proba, key=proba.get, reverse=True)[1]
            st.markdown(f'<div class="note note-warn">⚠️ <b>Low margin</b> – the model is almost undecided between '
                        f'<b>{pred}</b> and <b>{second}</b>. Treat this as uncertain and review it manually.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="note note-ok">✅ Clear margin between the top two categories.</div>',
                        unsafe_allow_html=True)

        signals = [("AI terms", feats["ai_term_count"]), ("Business reasons", feats["business_reason_count"]),
                   ("Third-party words", feats["third_party_count"]), ("AI denials", feats["ai_negation_count"]),
                   ("Company source", "yes" if feats["source_is_company"] else "no"),
                   ("Sector group", str(feats["sector_group"]).replace("_", " "))]
        cells = "".join(f'<div class="signal"><div class="v">{html.escape(str(v))}</div><div class="k">{k}</div></div>'
                        for k, v in signals)
        with st.expander("🔎 Text signals the model saw"):
            st.markdown(f'<div class="signal-grid">{cells}</div>', unsafe_allow_html=True)
            st.caption("Cleaned, company-masked text: " + feats["text_clean"])

    st.markdown('<div class="disclaimer">⚖️ <b>Disclaimer:</b> the model predicts which attribution category the '
                'AI Layoffs Register would most likely assign, based on how an event is reported. It is '
                '<b>not</b> evidence that AI caused any job losses.</div>', unsafe_allow_html=True)
