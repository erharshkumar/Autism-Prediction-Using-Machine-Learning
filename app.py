import streamlit as st
import pandas as pd
import numpy as np
import pickle, json
from pathlib import Path

st.set_page_config(
    page_title="Autism Screening Assistant",
    page_icon="🧠",
    layout="wide",
)

# ── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("autism_pipeline_tier4.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("autism_threshold_tier4.json") as f:
        cfg = json.load(f)
    return pipeline, cfg["optimal_threshold"], cfg["model"]

pipeline, threshold, model_name = load_artifacts()

# ── AQ-10 question text ──────────────────────────────────────────────────────
AQ10_QUESTIONS = {
    "A1_Score" : "I often notice small sounds when others do not",
    "A2_Score" : "I usually concentrate more on the whole picture rather than small details",
    "A3_Score" : "I find it easy to do more than one thing at once",
    "A4_Score" : "If there is an interruption, I can switch back to what I was doing very quickly",
    "A5_Score" : "I find it easy to read between the lines when someone is talking to me",
    "A6_Score" : "I know how to tell if someone listening to me is getting bored",
    "A7_Score" : "When I am reading a story I find it difficult to work out the characters intentions",
    "A8_Score" : "I like to collect information about categories of things",
    "A9_Score" : "I find it easy to work out what someone is thinking just by looking at their face",
    "A10_Score": "I find it difficult to work out people intentions",
}

COUNTRIES = [
    'Afghanistan','Angola','Argentina','Armenia','Aruba','Australia','Austria',
    'Azerbaijan','Bahamas','Bangladesh','Belgium','Bolivia','Brazil','Burundi',
    'Canada','China','Cyprus','Czech Republic','Egypt','Ethiopia','France',
    'Germany','Iceland','India','Iran','Iraq','Ireland','Italy','Japan','Jordan',
    'Kazakhstan','Malaysia','Mexico','Netherlands','New Zealand','Nicaragua',
    'Niger','Oman','Pakistan','Romania','Russia','Saudi Arabia','Serbia',
    'Sierra Leone','South Africa','Spain','Sri Lanka','Sweden','Tonga',
    'Ukraine','United Arab Emirates','United Kingdom','United States','Vietnam',
]
ETHNICITIES = [
    'Asian','Black','Hispanic','Latino','Middle Eastern',
    'Others','Pasifika','South Asian','Turkish','White-European',
]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🧠 Autism Screening Assistant")
st.caption(
    f"**Model:** {model_name} stacking ensemble  |  "
    f"**Threshold:** {threshold:.3f} (sensitivity ≥ 0.90)  |  "
    "**⚠ For research use only — not a clinical diagnosis**"
)
st.divider()

# ── Input form ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("AQ-10 Questionnaire")
    st.caption("Select 1 = Applies / Definitely agree  |  0 = Does not apply / Definitely disagree")
    a_scores = {}
    for key, question in AQ10_QUESTIONS.items():
        a_scores[key] = st.radio(
            f"**{key.replace('_Score','')}**: {question}",
            options=[0, 1],
            format_func=lambda x: "1 — Applies" if x else "0 — Does not apply",
            horizontal=True,
            key=key,
        )

with col2:
    st.subheader("Demographics")
    age    = st.slider("Age", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Gender", ["f","m"],
                          format_func=lambda x: "Female" if x=="f" else "Male")
    ethnicity = st.selectbox("Ethnicity", ETHNICITIES)
    country   = st.selectbox("Country of Residence", COUNTRIES,
                              index=COUNTRIES.index("United Kingdom"))

with col3:
    st.subheader("Medical History")
    jaundice        = st.selectbox("Born with jaundice?", ["no","yes"])
    austim          = st.selectbox("Family member with autism?", ["no","yes"])
    used_app_before = st.selectbox("Used ASD screening app before?", ["no","yes"])
    relation        = st.selectbox("Relation to respondent", ["Self","Others"])

st.divider()

# ── Prediction ───────────────────────────────────────────────────────────────
if st.button("🔍  Run Screening", type="primary", use_container_width=True):
    input_row = {
        **a_scores,
        "age"            : age,
        "gender"         : gender,
        "ethnicity"      : ethnicity,
        "jaundice"       : jaundice,
        "austim"         : austim,
        "country_of_res" : country,
        "used_app_before": used_app_before,
        "relation"       : relation,
    }
    input_df = pd.DataFrame([input_row])

    with st.spinner("Running model..."):
        prob = pipeline.predict_proba(input_df)[0, 1]
        pred = int(prob >= threshold)

    st.divider()
    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        if pred == 1:
            st.error(f"### ⚠  Screen Positive for ASD\nP(ASD) = **{prob:.1%}**")
            st.warning(
                "A positive screen indicates the person should be referred "
                "for a comprehensive diagnostic evaluation by a licensed clinician. "
                "**This is NOT a diagnosis.**"
            )
        else:
            st.success(f"### ✓  Screen Negative for ASD\nP(ASD) = **{prob:.1%}**")
            st.info(
                "A negative screen does not rule out ASD. "
                "If there are clinical concerns, seek professional evaluation."
            )

    with gauge_col:
        st.subheader("Probability Gauge")
        fig, ax = plt.subplots(figsize=(4, 1.2))
        bar_color = "#E24B4A" if prob >= threshold else "#1baf7a"
        ax.barh(["P(ASD)"], [prob], color=bar_color, height=0.5)
        ax.axvline(threshold, color="#eda100", lw=2, linestyle="--",
                   label=f"Threshold ({threshold:.2f})")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.legend(fontsize=8)
        ax.set_title(f"{prob:.1%}", fontsize=14, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # AQ-10 score summary
    aq_total = sum(a_scores.values())
    st.divider()
    st.subheader("AQ-10 Score Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total AQ-10 Score", f"{aq_total} / 10",
              help="Score ≥ 6 is the standard AQ-10 positive screen cutoff")
    c2.metric("P(ASD)", f"{prob:.1%}")
    c3.metric("Decision", "Refer for evaluation" if pred else "No referral indicated")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Disclaimer:** This tool is for research and educational purposes only. "
    "It is not a medical device and must not be used as a substitute for professional "
    "clinical assessment. | Built by Harsh Kumar · harshkumar.in"
)
