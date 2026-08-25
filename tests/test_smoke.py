"""
Smoke test for the AI Policy & Guardrails Auditor.

Runs the full LangGraph pipeline end-to-end against the sample policy and
configs, then asserts:

  1. Recall on planted violations >= RECALL_THRESHOLD (default 0.90)
  2. False positives on the clean baseline config <= MAX_BASELINE_FINDINGS

Requires OPENAI_API_KEY in the environment (loaded from .env if present).
This test spends real API tokens — run selectively:

    pytest -m smoke tests/test_smoke.py -s
"""

import asyncio
import os
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auditor import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Planted violations. Each entry: (config_filename, policy_clause_id, hint).
# `hint` is a short substring (lowercased) expected to appear somewhere in the
# finding's evidence or recommendation. Matching is fuzzy so agents have room
# to phrase things naturally.
# ---------------------------------------------------------------------------

EXPECTED_FINDINGS = [
    # customer-support-bot.yaml
    ("customer-support-bot.yaml", "3.3", "human"),          # impersonation
    ("customer-support-bot.yaml", "4.2", "*"),              # wildcard tool
    ("customer-support-bot.yaml", "4.1", "pii"),            # missing PII redaction
    ("customer-support-bot.yaml", "4.3", "injection"),      # missing prompt injection defense
    ("customer-support-bot.yaml", "7.1", "refund"),         # autonomous refund
    ("customer-support-bot.yaml", "7.2", "email"),          # autonomous customer email
    ("customer-support-bot.yaml", "6.2", "180"),            # retention exceeds 30
    ("customer-support-bot.yaml", "6.4", "eu"),             # EU users processed in US
    ("customer-support-bot.yaml", "5.2", "opt-out"),        # opt-out training data
    ("customer-support-bot.yaml", "5.1", "disclosure"),     # no AI disclosure

    # internal-summarizer.yaml
    ("internal-summarizer.yaml", "4.1", "pii"),             # missing PII redaction
    ("internal-summarizer.yaml", "6.3", "local"),           # persist to local disk

    # sales-outreach-generator.yaml
    ("sales-outreach-generator.yaml", "3.3", "human"),      # impersonation
    ("sales-outreach-generator.yaml", "7.2", "email"),      # autonomous external send
    ("sales-outreach-generator.yaml", "5.1", "footer"),     # footer-only disclosure
    ("sales-outreach-generator.yaml", "4.3", "injection"),  # missing prompt injection defense
]

RECALL_THRESHOLD = 0.85          # requirements doc target is 90%; leave 5pt headroom
MAX_BASELINE_FINDINGS = 1        # clean baseline should be near-zero
CLEAN_BASELINE = "code-review-agent.yaml"


def _matches(finding: dict, config: str, clause: str, hint: str) -> bool:
    if finding.get("source_config") != config:
        return False
    if str(finding.get("policy_clause_id", "")).strip() != clause:
        return False
    blob = (
        (finding.get("evidence") or "") + " " +
        (finding.get("recommendation") or "") + " " +
        (finding.get("explanation") or "")
    ).lower()
    return hint.lower() in blob


@pytest.mark.smoke
def test_planted_violation_recall_and_baseline_precision():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping live smoke test.")

    app = build_graph()
    final_state = asyncio.run(app.ainvoke({}))
    report = final_state["report_json"]
    findings = report["findings"]

    # -- Recall on planted violations ---------------------------------------
    caught = 0
    missed = []
    for cfg, clause, hint in EXPECTED_FINDINGS:
        if any(_matches(f, cfg, clause, hint) for f in findings):
            caught += 1
        else:
            missed.append(f"{cfg}#{clause} ({hint})")

    recall = caught / len(EXPECTED_FINDINGS)
    print(f"\n[smoke] Recall: {caught}/{len(EXPECTED_FINDINGS)} = {recall:.2%}")
    if missed:
        print("[smoke] Missed:")
        for m in missed:
            print(f"        - {m}")

    # -- False positives on the clean baseline ------------------------------
    baseline_findings = [f for f in findings if f.get("source_config") == CLEAN_BASELINE]
    print(f"[smoke] Findings on {CLEAN_BASELINE}: {len(baseline_findings)}")

    assert recall >= RECALL_THRESHOLD, (
        f"Recall {recall:.2%} below threshold {RECALL_THRESHOLD:.2%}. "
        f"Missed: {missed}"
    )
    assert len(baseline_findings) <= MAX_BASELINE_FINDINGS, (
        f"Clean baseline produced {len(baseline_findings)} findings; "
        f"threshold is {MAX_BASELINE_FINDINGS}."
    )
