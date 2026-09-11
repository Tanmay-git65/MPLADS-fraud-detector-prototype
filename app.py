import os
import streamlit as st
import pandas as pd
import risk_scoring
from detectors import cost_anomaly, duplicate_works, payment_structuring, execution_risk, image_forensics, vendor_network, rules_engine
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(page_title="MAPLADS — Anomaly Detection Engine", layout="wide")

@st.cache_data
def get_risk_data():
    data = risk_scoring.load_data(DATA_DIR)
    works, payments, photos, mps, ias = data["works"], data["payments"], data["photos"], data["mps"], data["ias"]

    # 1. Cost Anomaly
    cost_scored, cost_importances = cost_anomaly.run(works)
    # 2. Duplicate Works
    dup_scored = duplicate_works.run(works)
    # 3. Payment Structuring
    struct_scored, ia_benford = payment_structuring.run(payments, works)
    # 4. Execution Risk
    exec_scored = execution_risk.run(works, payments)
    # 5. Image Forensics
    img_scored = image_forensics.run(photos, works)
    # 6. Vendor Network
    net_scored, ia_network = vendor_network.run(payments, works, ias)
    # 7. Rules Engine
    rule_scored, entitlement_breaches = rules_engine.run(works, payments, mps)

    # Base DF with works details
    df = works[["work_id", "mp_id", "state", "district", "ia_id", "category",
                "sanctioned_amount", "status", "sanction_date", "expected_duration_days", "scale"]].copy()
    
    # Calculate additional fields for UI display
    paid = payments.groupby("work_id")["amount"].sum().rename("total_paid")
    df = df.merge(paid, on="work_id", how="left")
    df["total_paid"] = df["total_paid"].fillna(0)

    df = df.merge(cost_scored, on="work_id", how="left")
    df = df.merge(dup_scored, on="work_id", how="left")
    df = df.merge(struct_scored, on="work_id", how="left")
    df = df.merge(exec_scored, on="work_id", how="left")
    df = df.merge(img_scored, on="work_id", how="left")
    df = df.merge(net_scored, on="work_id", how="left")
    df = df.merge(rule_scored, on="work_id", how="left")
    
    score_cols = ["rule_score", "cost_anomaly_score", "dup_score", "structuring_score",
                  "image_forensics_score", "network_score", "execution_risk_score"]
    reason_map = {
        "rule_score": "rule_reason",
        "cost_anomaly_score": "cost_anomaly_reason",
        "dup_score": "dup_reason",
        "structuring_score": "structuring_reason",
        "image_forensics_score": "image_forensics_reason",
        "network_score": "network_reason",
        "execution_risk_score": "execution_risk_reason",
    }
    
    for c in score_cols:
        df[c] = df[c].fillna(0.0)
    for c in reason_map.values():
        df[c] = df[c].fillna("")

    df_scored = risk_scoring.compute_composite_risk(df, score_cols, reason_map)
    return df_scored, data

df_scored, raw_data = get_risk_data()

st.title("MAPLADS — Anomaly Detection Engine")
st.markdown("##### Technical validation interface • Demonstration using synthetic data")
st.info("Prototype demonstration using synthetic data")

# --- SECTION 1: SELECT WORK ---
st.header("1. Select Work")

# Let's sort the work_ids so high risk ones can be easily spotted or selected
df_scored = df_scored.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)

work_list = df_scored.apply(lambda r: f"{r['work_id']} - Risk: {r['risk_tier']} ({r['composite_risk_score']})", axis=1).tolist()
selected_str = st.selectbox("Select Work ID to inspect:", work_list)
selected_work_id = selected_str.split(" ")[0]

work_row = df_scored[df_scored["work_id"] == selected_work_id].iloc[0]

st.subheader("Work Details")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Work ID", work_row["work_id"])
col2.metric("Work Category", work_row["category"])
col3.metric("District", work_row["district"])
col4.metric("Status", work_row["status"])

col5, col6, col7, col8 = st.columns(4)
col5.metric("Sanctioned Amount", f"₹{work_row['sanctioned_amount']:,.0f}")
col6.metric("Total Paid", f"₹{work_row['total_paid']:,.0f}")
col7.metric("Sanction Date", str(work_row["sanction_date"]))
col8.metric("Expected Duration", f"{work_row['expected_duration_days']} days")


# --- SECTION 2: OVERALL RISK ---
st.header("2. Overall Risk")

score = work_row["composite_risk_score"]
tier = work_row["risk_tier"]
num_flags = work_row.get("corroboration_count", 0) + 1 if score > 0 else 0 
# Wait, corroboration_count is additional flags. Total flags = base (if any fired) + corroboration_count
# Let's compute exact flagged detectors:
score_cols = ["cost_anomaly_score", "dup_score", "structuring_score", 
              "image_forensics_score", "network_score", "rule_score", "execution_risk_score"]
firing_threshold = risk_scoring.FIRING_THRESHOLD
flagged_detectors = [c for c in score_cols if work_row[c] >= firing_threshold]

risk_color = "red" if tier == "Critical" else ("orange" if tier == "High" else ("#e6c200" if tier == "Medium" else "green"))

st.markdown(f"""
<div style="padding: 20px; border-radius: 10px; border: 2px solid {risk_color}; text-align: center; background-color: rgba(255,255,255,0.05);">
    <h2 style="margin:0;">OVERALL RISK: {score} / 100</h2>
    <h3 style="color: {risk_color}; margin: 5px 0;">{tier.upper()} RISK</h3>
    <p style="margin:0;">{len(flagged_detectors)} of 7 detectors flagged</p>
</div>
""", unsafe_allow_html=True)


# --- SECTION 3: 7 DETECTOR BREAKDOWN ---
st.header("3. Anomaly Breakdown")

def format_detector(name, score_val, reason):
    flagged = score_val >= firing_threshold
    status = "🔴 FLAGGED" if score_val >= 0.7 else ("🟡 FLAGGED" if flagged else "🟢 NORMAL")
    return f"**{name}**: {score_val*100:.0f}/100 - {status}"

col1, col2 = st.columns(2)
with col1:
    st.markdown(format_detector("Cost Anomaly", work_row["cost_anomaly_score"], work_row["cost_anomaly_reason"]))
    st.markdown(format_detector("Duplicate Work", work_row["dup_score"], work_row["dup_reason"]))
    st.markdown(format_detector("Payment Structuring", work_row["structuring_score"], work_row["structuring_reason"]))
    st.markdown(format_detector("Execution Risk (Delay)", work_row["execution_risk_score"], work_row["execution_risk_reason"]))
with col2:
    st.markdown(format_detector("Image Forensics", work_row["image_forensics_score"], work_row["image_forensics_reason"]))
    st.markdown(format_detector("Vendor Network", work_row["network_score"], work_row["network_reason"]))
    st.markdown(format_detector("Compliance / Rules", work_row["rule_score"], work_row["rule_reason"]))

# --- SECTION 4: DETECTOR EXPLANATION ---
st.header("4. Detector Explanations")
tabs = st.tabs(["Cost Anomaly", "Duplicate Work", "Payment Structuring", "Execution Risk", "Image Forensics", "Vendor Network", "Compliance Rules"])

with tabs[0]:
    st.subheader("Cost Anomaly Detector")
    st.markdown("**What does this detector look at?**")
    st.write(f"- Sanctioned Cost: ₹{work_row['sanctioned_amount']:,.0f}")
    st.write(f"- Category: {work_row['category']}")
    st.write(f"- Scale: {work_row.get('scale', 'N/A')}")
    st.write(f"- District: {work_row['district']}")
    
    st.markdown("**How did it reach the result?**")
    if pd.notna(work_row.get("peer_zscore")):
        st.write(f"- Peer Group Deviation (Z-Score): {work_row['peer_zscore']:.2f}")
    if pd.notna(work_row.get("ml_relative_residual")):
        st.write(f"- ML Model Relative Residual: {work_row['ml_relative_residual']*100:.1f}%")
        
    st.markdown("**Result**")
    st.write(f"Score: {work_row['cost_anomaly_score']*100:.0f}/100")
    st.markdown("**Human-Readable Explanation:**")
    st.info(work_row["cost_anomaly_reason"] if work_row["cost_anomaly_reason"] else "Normal. Cost is within expected ranges.")

with tabs[1]:
    st.subheader("Duplicate Work Detector")
    st.markdown("**What does this detector look at?**")
    st.write("Compares work description text and geographical coordinates (latitude/longitude) against other works in the same district.")
    
    st.markdown("**Result**")
    st.write(f"Score: {work_row['dup_score']*100:.0f}/100")
    st.markdown("**Human-Readable Explanation:**")
    st.info(work_row["dup_reason"] if work_row["dup_reason"] else "Normal. No duplicate works found nearby.")

with tabs[2]:
    st.subheader("Payment Structuring Detector")
    st.markdown("**What does this detector look at?**")
    st.write("Looks for multiple payments made just under the ₹1,000,000 approval threshold within a short time window.")
    
    st.markdown("**Result**")
    st.write(f"Score: {work_row['structuring_score']*100:.0f}/100")
    st.markdown("**Human-Readable Explanation:**")
    st.info(work_row["structuring_reason"] if work_row["structuring_reason"] else "Normal. No suspicious payment structuring detected.")

with tabs[3]:
    st.subheader("Execution Risk (Delay/Velocity) Detector")
    st.markdown("**What does this detector look at?**")
    st.write(f"- Status: {work_row['status']}")
    st.write(f"- Expected Duration: {work_row['expected_duration_days']} days")
    st.write(f"- Sanctioned Amount: ₹{work_row['sanctioned_amount']:,.0f}")
    st.write(f"- Disbursed Amount: ₹{work_row['total_paid']:,.0f}")
    
    st.markdown("**How did it reach the result?**")
    if pd.notna(work_row.get("overdue_ratio")):
        st.write(f"- Overdue Ratio: {work_row['overdue_ratio']:.2f}x expected duration")
    if pd.notna(work_row.get("disbursed_fraction")):
        st.write(f"- Disbursed Fraction: {work_row['disbursed_fraction']*100:.1f}%")
        
    st.markdown("**Result**")
    st.write(f"Score: {work_row['execution_risk_score']*100:.0f}/100")
    st.markdown("**Human-Readable Explanation:**")
    st.info(work_row["execution_risk_reason"] if work_row["execution_risk_reason"] else "Normal. Execution timeline and payments are aligned.")

with tabs[4]:
    st.subheader("Image Forensics Detector")
    st.markdown("**Result**")
    st.write(f"Score: {work_row['image_forensics_score']*100:.0f}/100")
    st.markdown("**Human-Readable Explanation:**")
    st.info(work_row["image_forensics_reason"] if work_row.get("image_forensics_reason") else "Normal. Images look authentic and match work type.")

with tabs[5]:
    st.subheader("Vendor Network Detector")
    st.markdown("**Result**")
    st.write(f"Score: {work_row['network_score']*100:.0f}/100")
    st.markdown("**Human-Readable Explanation:**")
    st.info(work_row["network_reason"] if work_row.get("network_reason") else "Normal. Vendor network behavior is standard.")

with tabs[6]:
    st.subheader("Compliance Rules Engine")
    st.markdown("**Result**")
    st.write(f"Score: {work_row['rule_score']*100:.0f}/100")
    st.markdown("**Human-Readable Explanation:**")
    st.info(work_row["rule_reason"] if work_row.get("rule_reason") else "Normal. Complies with core MPLADS guidelines.")

# --- SECTION 5: MULTI-DETECTOR REASONING ---
st.header("5. Multi-Detector Reasoning")
st.write("How independent anomaly signals contribute to the final assessment:")

cols = st.columns(len(score_cols))
for i, col_name in enumerate(score_cols):
    score_val = work_row[col_name]
    with cols[i]:
        st.write(f"**{col_name.replace('_score', '').replace('_', ' ').title()}**")
        if score_val >= firing_threshold:
            st.error(f"{score_val*100:.0f} 🔴")
        elif score_val > 0:
            st.warning(f"{score_val*100:.0f} 🟡")
        else:
            st.success(f"{score_val*100:.0f} 🟢")

st.markdown(f"**⬇️ RISK AGGREGATION**")
st.info(f"**{score} / 100** — **{tier.upper()} RISK** (Requires Investigation)")
