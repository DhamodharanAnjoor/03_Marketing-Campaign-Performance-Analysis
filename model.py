import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Campaign Profit Predictor", layout="centered")

# ── Load Model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("profit_loss_classifier.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ profit_loss_classifier.pkl not found — run pipeline.py first!")
    st.stop()

# ── Encoding Maps ────────────────────────────────────────
brand_map    = {"Nykaa":0, "Purplle":1, "Tira":2}
camp_map     = {"Email":0, "Influencer":1, "Paid Ads":2, "SEO":3, "Social Media":4}
audience_map = {"College Students":0, "Premium Shoppers":1,
                "Tier 2 City Customers":2, "Working Women":3, "Youth":4}
lang_map     = {"Bengali":0, "English":1, "Hindi":2, "Tamil":3}

# ── Header ───────────────────────────────────────────────
st.title("Campaign Profit Predictor")
st.caption("Nykaa · Purplle · Tira")
st.markdown("Enter your campaign details and click **Predict** to see if it will be a **Profit ✅** or **Loss ❌**.")
st.divider()

# ── Inputs: Row 1 — Dropdowns ────────────────────────────
c1, c2, c3, c4 = st.columns(4)
brand     = c1.selectbox("Brand",            list(brand_map))
camp_type = c2.selectbox("Campaign Type",    list(camp_map))
audience  = c3.selectbox("Target Audience",  list(audience_map))
language  = c4.selectbox("Language",         list(lang_map))

# ── Inputs: Row 2 — Numbers ──────────────────────────────
st.markdown("### Campaign Metrics")
left, right = st.columns(2)

with left:
    duration    = st.number_input("Duration (days)",   min_value=1,   value=15)
    impressions = st.number_input("Impressions",       min_value=0,   value=50000,   step=1000)
    clicks      = st.number_input("Clicks",            min_value=0,   value=3000,    step=100)
    leads       = st.number_input("Leads",             min_value=0,   value=500,     step=50)
    conversions = st.number_input("Conversions",       min_value=0,   value=200,     step=10)

with right:
    revenue          = st.number_input("Revenue (₹)",          min_value=0.0, value=500000.0, step=10000.0)
    acquisition_cost = st.number_input("Acquisition Cost (₹)", min_value=0.0, value=200.0,    step=10.0)
    engagement_score = st.number_input("Engagement Score",     min_value=0.0, value=13.0,     step=0.5)
    channels         = st.multiselect("Channels Used",
                           ['Email','Facebook','Google','Instagram','WhatsApp','YouTube'],
                           default=['Instagram','Facebook'])

st.divider()

# ── Predict ──────────────────────────────────────────────
if st.button("Predict Profit / Loss", use_container_width=True, type="primary"):

    input_df = pd.DataFrame([[
        brand_map[brand], camp_map[camp_type], audience_map[audience], lang_map[language],
        duration, impressions, clicks, leads, conversions,
        revenue, acquisition_cost, engagement_score,
        1 if "Email"     in channels else 0,
        1 if "Facebook"  in channels else 0,
        1 if "Google"    in channels else 0,
        1 if "Instagram" in channels else 0,
        1 if "WhatsApp"  in channels else 0,
        1 if "YouTube"   in channels else 0,
    ]], columns=[
        "Campaign_Name","Campaign_Type","Target_Audience","Language",
        "Duration","Impressions","Clicks","Leads","Conversions",
        "Revenue","Acquisition_Cost","Engagement_Score",
        "Email","Facebook","Google","Instagram","WhatsApp","YouTube"
    ])

    pred  = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.markdown("### 📌 Result")
    if pred == 1:
        conf = proba[1] * 100
        st.success(f"✅  PROFIT  —  Confidence: {conf:.1f}%")
    else:
        conf = proba[0] * 100
        st.error(f"❌  LOSS  —  Confidence: {conf:.1f}%")

    st.progress(int(conf))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Brand",         brand)
    m2.metric("Campaign Type", camp_type)
    m3.metric("Revenue",       f"₹{revenue:,.0f}")
    m4.metric("Channels",      len(channels))

# ── Footer ───────────────────────────────────────────────
st.divider()
st.caption("Marketing Campaign Performance Prediction  |  RandomForestClassifier  |  Accuracy: 96.26%")
