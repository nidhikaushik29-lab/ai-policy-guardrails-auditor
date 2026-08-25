"""
AI Policy & Guardrails Auditor — Dashboard

Streamlit dashboard that visualizes the governance audit report.
Reads from out/report.json produced by auditor.py.

Run locally:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPORT_PATH = Path(__file__).parent / "out" / "report.json"


st.set_page_config(
    page_title="AI Policy & Guardrails Auditor",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("AI Policy & Guardrails Auditor")
st.caption(
    "Automated governance review for LLM-based agents. "
    "Five specialist agents run in parallel via LangGraph, "
    "auditing configurations against an enterprise AI usage policy."
)

# ---------------------------------------------------------------------------
# Load report
# ---------------------------------------------------------------------------

if not REPORT_PATH.exists():
    st.error(
        f"No report found at `{REPORT_PATH}`. "
        "Run `python auditor.py` first to generate one."
    )
    st.stop()

report = json.loads(REPORT_PATH.read_text())
findings = report.get("findings", [])
counts = report.get("counts", {})

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Overall risk score", f"{report.get('risk_score', 0)}/100")
k2.metric("Total findings", report.get("total_findings", 0))
k3.metric("Critical", counts.get("critical", 0))
k4.metric("High", counts.get("high", 0))
k5.metric("Medium", counts.get("medium", 0))
k6.metric("Configs audited", len(report.get("configs_audited", [])))

st.divider()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

if not findings:
    st.success("No findings above confidence threshold. Configs look compliant.")
    st.stop()

df = pd.DataFrame(findings)

# normalize columns that may be missing across agents
for col in [
    "finding_id", "severity", "policy_clause_id", "source_config",
    "agent", "evidence", "recommendation", "confidence",
    "data_type", "action_risk",
]:
    if col not in df.columns:
        df[col] = None

with st.sidebar:
    st.header("Filters")

    severity_sel = st.multiselect(
        "Severity",
        options=["critical", "high", "medium", "low"],
        default=["critical", "high", "medium", "low"],
    )
    agent_sel = st.multiselect(
        "Agent",
        options=sorted(df["agent"].dropna().unique().tolist()),
        default=sorted(df["agent"].dropna().unique().tolist()),
    )
    config_sel = st.multiselect(
        "Config",
        options=sorted(df["source_config"].dropna().unique().tolist()),
        default=sorted(df["source_config"].dropna().unique().tolist()),
    )
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.6, 0.05)

filtered = df[
    df["severity"].isin(severity_sel)
    & df["agent"].isin(agent_sel)
    & df["source_config"].isin(config_sel)
    & (df["confidence"].fillna(0).astype(float) >= min_conf)
].copy()

# ---------------------------------------------------------------------------
# Charts row
# ---------------------------------------------------------------------------

c1, c2 = st.columns(2)

with c1:
    st.subheader("Findings by severity")
    sev_counts = filtered["severity"].value_counts().reindex(
        ["critical", "high", "medium", "low"], fill_value=0
    )
    st.bar_chart(sev_counts)

with c2:
    st.subheader("Findings by config")
    cfg_counts = filtered["source_config"].value_counts()
    st.bar_chart(cfg_counts)

# ---------------------------------------------------------------------------
# Findings table
# ---------------------------------------------------------------------------

st.subheader(f"Findings ({len(filtered)})")

table = filtered[
    [
        "severity", "source_config", "policy_clause_id",
        "agent", "recommendation", "confidence",
    ]
].rename(
    columns={
        "source_config": "config",
        "policy_clause_id": "clause",
        "recommendation": "recommendation",
    }
)
st.dataframe(table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Details drill-down
# ---------------------------------------------------------------------------

st.subheader("Finding details")

for _, row in filtered.iterrows():
    with st.expander(
        f"[{str(row['severity']).upper()}] {row['source_config']} — "
        f"clause {row['policy_clause_id']} — {row['agent']}"
    ):
        st.markdown(f"**Recommendation:** {row['recommendation']}")
        st.markdown(f"**Confidence:** {row['confidence']}")
        if row.get("data_type"):
            st.markdown(f"**Data type:** {row['data_type']}")
        if row.get("action_risk"):
            st.markdown(f"**Action risk:** {row['action_risk']}")
        st.markdown("**Evidence:**")
        st.code(row.get("evidence") or "", language="yaml")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Built with the OpenAI Agents SDK and LangGraph. "
    "Design-time governance auditing — not a substitute for legal review."
)
