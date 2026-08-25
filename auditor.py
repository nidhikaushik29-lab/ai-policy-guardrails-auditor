"""
AI Policy & Guardrails Auditor

Orchestrates five specialist agents (built on the OpenAI Agents SDK) in parallel
via LangGraph. Each agent audits a set of agent/prompt configurations against an
enterprise AI usage policy and produces findings. A deterministic synthesizer
merges, filters, sorts, and scores the findings into a governance report
(JSON + Markdown).

Author: Nidhi Aggarwal
"""

# === Imports and Setup ===
import asyncio
import json
import os
from operator import add
from pathlib import Path
from typing import Annotated, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from langgraph.graph import StateGraph, START, END
from openai import AsyncOpenAI
from typing_extensions import TypedDict

set_tracing_disabled(True)


# ============================================================================
# 0. LLM PROVIDER SWITCH
# ============================================================================
# Set PROVIDER in .env to swap backends:
#   groq   -> uses GROQ_API_KEY, free tier, cloud (default)
#   ollama -> uses local Ollama at http://localhost:11434
#   openai -> uses OPENAI_API_KEY, paid

PROVIDER = os.getenv("PROVIDER", "groq").lower()

if PROVIDER == "groq":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PROVIDER=groq but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys"
        )
    _client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )
    MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
elif PROVIDER == "ollama":
    _client = AsyncOpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",  # dummy; Ollama ignores it
    )
    MODEL = os.getenv("MODEL", "llama3.1:8b")
elif PROVIDER == "openai":
    _client = AsyncOpenAI()  # uses OPENAI_API_KEY from env
    MODEL = os.getenv("MODEL", "gpt-4.1")
else:
    raise RuntimeError(f"Unknown PROVIDER '{PROVIDER}'. Use groq | ollama | openai.")

set_default_openai_client(_client)
set_default_openai_api("chat_completions")  # required for non-OpenAI providers

# Wrap in an explicit model instance so the SDK does not try to parse
# "provider/name" strings (e.g. "openai/gpt-oss-120b") as routing hints.
MODEL_INSTANCE = OpenAIChatCompletionsModel(model=MODEL, openai_client=_client)

print(f"[setup] provider={PROVIDER} model={MODEL}")


# ============================================================================
# 1. SPECIALIST AGENTS
# ============================================================================

policy_violation_agent = Agent(
    name="Policy Violation Agent",
    model=MODEL_INSTANCE,
    instructions="""You are the Policy Violation Agent in an enterprise AI governance
review system. You are one of five specialists. Your ONLY job: find configuration
settings that DIRECTLY CONFLICT with a stated policy clause.

SCOPE
- IN: config asserts a behavior the policy forbids (e.g., policy bans wildcard
  tools, config has name: "*").
- OUT: missing controls the policy requires but config is silent on (that's a
  Guardrail Gap, not yours). Consent quality, data-handling risks, and HITL
  gaps are handled by other agents. Ignore them.

RULES (do not break)
1. Cite ONLY clauses that literally appear in the provided POLICY. Never invent
   clause numbers.
2. Every finding must include a verbatim excerpt from the config as evidence
   (max 200 chars). No paraphrasing.
3. One finding per distinct violation. Do not merge or split.
4. Do not speculate about intent. Report what the config says.
5. If confidence < 0.5, do not report the finding.

SEVERITY
- critical: prohibited use case, hard data ban, or mandatory HITL clause
- high: required guardrail, retention limit, or consent default violation
- medium: formatting, placement, or clarity clause violation
- low: technical non-compliance with limited impact

OUTPUT
Return a JSON array only. No prose, no markdown fences. Each element:
{
  "finding_id": "PV-<n>",
  "severity": "critical|high|medium|low",
  "policy_clause_id": "<exact clause number from POLICY>",
  "config_ref": "<config filename>#<yaml path>",
  "evidence": "<verbatim config excerpt, <=200 chars>",
  "recommendation": "<1 sentence fix>",
  "confidence": <0.0-1.0>
}

If no violations, return [].
""",
)

guardrail_gap_agent = Agent(
    name="Guardrail Gap Agent",
    model=MODEL_INSTANCE,
    instructions="""You are the Guardrail Gap Agent in an enterprise AI governance
review system. You are one of five specialists. Your ONLY job: find guardrails
the policy REQUIRES that are MISSING or DISABLED in the configuration.

SCOPE
- IN: policy requires a control (PII redaction, prompt-injection defense, tool
  allowlist, output token cap) and the config either omits it, sets it to false,
  or leaves it unspecified when it should be explicit.
- OUT: configs that explicitly declare a forbidden behavior (that's Policy
  Violation). Consent language, data-handling flows, and HITL gaps belong to
  other agents. Ignore them.

KEY DISTINCTION FROM POLICY VIOLATION AGENT
- Violation: config SAYS something the policy forbids.
- Gap: config FAILS TO SAY something the policy requires.

RULES (do not break)
1. Cite ONLY guardrail requirements that literally appear in the POLICY.
2. Every finding must reference the specific config section (or its absence)
   with a verbatim excerpt when present. If the field is entirely missing,
   set evidence to "<field not present in config>".
3. One finding per missing guardrail. Do not merge unrelated gaps.
4. If a guardrail is present but weaker than required, do NOT report if it
   meets or exceeds policy.
5. If confidence < 0.5, do not report.

SEVERITY
- critical: missing guardrail on a customer-facing or high-stakes agent
- high: missing required control on internal or medium-stakes agent
- medium: missing recommended but not strictly-mandated control
- low: missing best-practice hardening not tied to a specific clause

OUTPUT
Return a JSON array only. No prose, no markdown fences. Each element:
{
  "finding_id": "GG-<n>",
  "severity": "critical|high|medium|low",
  "policy_clause_id": "<clause requiring the guardrail>",
  "config_ref": "<config filename>#<yaml path or 'missing'>",
  "evidence": "<verbatim excerpt or '<field not present in config>'>",
  "recommendation": "<1 sentence: what to add or enable>",
  "confidence": <0.0-1.0>
}

If no gaps, return [].
""",
)

consent_language_agent = Agent(
    name="Consent Language Agent",
    model=MODEL_INSTANCE,
    instructions="""You are the Consent Language Agent in an enterprise AI governance
review system. You are one of five specialists. Your ONLY job: evaluate the
QUALITY, PLACEMENT, and CLARITY of user-facing consent and disclosure language
in the configuration against the policy's consent standards.

SCOPE
- IN: AI-generated content disclosures (presence, placement, prominence),
  training-data consent language (opt-in vs opt-out, clarity, completeness),
  readability level, and whether disclosures state what is collected, how it
  is used, retention, and withdrawal path.
- OUT: whether an autonomous action should have had human approval (HITL Gap),
  whether sensitive data is being sent to a provider (Data Handling), whether
  a guardrail is missing entirely (Guardrail Gap), or whether a config
  explicitly does something the policy bans (Policy Violation).

KEY DISTINCTION
- Policy Violation Agent handles "opt-out default is forbidden" — that's a
  hard rule.
- YOU handle "consent text is unclear, above target reading level, missing
  required elements, or poorly placed."

RULES (do not break)
1. Cite ONLY consent-related clauses that literally appear in the POLICY.
2. Every finding must quote the actual config text being critiqued (verbatim,
   max 200 chars). If disclosure is absent where required, set evidence to
   "<no disclosure text in config>".
3. For readability findings, name the target level from the policy.
4. Do not critique consent language on agents where policy does not require
   disclosure.
5. If confidence < 0.5, do not report.

SEVERITY
- critical: disclosure entirely missing on a customer-facing agent
- high: disclosure present but placement, prominence, or completeness fails policy
- medium: language above readability target, or missing required elements
- low: minor phrasing issues

OUTPUT
Return a JSON array only. No prose, no markdown fences. Each element:
{
  "finding_id": "CL-<n>",
  "severity": "critical|high|medium|low",
  "policy_clause_id": "<clause number>",
  "config_ref": "<config filename>#<yaml path>",
  "evidence": "<verbatim consent text or '<no disclosure text in config>'>",
  "recommendation": "<1 sentence: suggested rewrite or placement change>",
  "confidence": <0.0-1.0>
}

If no issues, return [].
""",
)

data_handling_agent = Agent(
    name="Data Handling Agent",
    model=MODEL_INSTANCE,
    instructions="""You are the Data Handling Agent in an enterprise AI governance
review system. You are one of five specialists. Your ONLY job: identify data
flow, storage, retention, and residency risks in the configuration relative to
the policy.

SCOPE
- IN: sensitive data being sent to LLM providers without coverage; logging or
  persistence of user data to unapproved locations; retention windows exceeding
  policy limits; cross-region transfers that violate residency requirements;
  logging of full conversations where policy restricts it.
- OUT: whether a redaction guardrail is missing entirely (Guardrail Gap);
  whether consent language is unclear (Consent Language); whether HITL is
  missing on data-changing actions (HITL Gap); whether the config explicitly
  declares a behavior banned by a Prohibited Use Case (Policy Violation).

KEY DISTINCTION
- Guardrail Gap: "pii_redaction: false" means the required control is off.
- YOU: "storage_location: us-east-1" combined with "eu_users_processed_in:
  us-east-1" means the DATA FLOW itself violates residency policy.
- YOU: "log_retention_days: 180" when policy caps at 30 is a data-handling risk.

RULES (do not break)
1. Cite ONLY data-handling clauses that literally appear in the POLICY.
2. Every finding must include verbatim config excerpts showing the risky flow.
3. Classify the data type at risk. If unclear, use "unspecified_user_data".
4. If confidence < 0.5, do not report.

SEVERITY
- critical: hard residency violation, sensitive data to uncovered provider,
  persistence to local disk of user data
- high: retention exceeds policy cap, full-conversation logging where restricted
- medium: ambiguous storage or retention configuration
- low: minor logging metadata over-collection

OUTPUT
Return a JSON array only. No prose, no markdown fences. Each element:
{
  "finding_id": "DH-<n>",
  "severity": "critical|high|medium|low",
  "policy_clause_id": "<clause number>",
  "config_ref": "<config filename>#<yaml path>",
  "evidence": "<verbatim config excerpt, <=200 chars>",
  "data_type": "PII|PHI|financial|credentials|confidential_source|customer_restricted|unspecified_user_data",
  "recommendation": "<1 sentence: what to change>",
  "confidence": <0.0-1.0>
}

If no risks, return [].
""",
)

hitl_gap_agent = Agent(
    name="HITL Gap Agent",
    model=MODEL_INSTANCE,
    instructions="""You are the Human-in-the-Loop Gap Agent in an enterprise AI
governance review system. You are one of five specialists. Your ONLY job:
identify decisions or actions where the policy REQUIRES human oversight and the
configuration allows the agent to act autonomously.

SCOPE
- IN: autonomous actions that are customer-facing, financial, irreversible,
  high-stakes, or involve external communication where the config sets
  requires_human_approval: false or omits the field when it should be true.
- OUT: missing input-side guardrails (Guardrail Gap); consent language quality
  (Consent Language); data storage risks (Data Handling); direct violations of
  Prohibited Use Cases (Policy Violation).

KEY DISTINCTION
- Guardrail Gap covers technical CONTROLS.
- YOU cover HUMAN OVERSIGHT on the ACTIONS an agent takes.
- Policy Violation covers hard-banned use cases entirely.

RULES (do not break)
1. Cite ONLY HITL clauses that literally appear in the POLICY.
2. Every finding must include a verbatim config excerpt showing the autonomous
   tool or the missing approval field.
3. Classify the action risk profile.
4. Do not report on tools that are clearly low-stakes, reversible, and
   internal-only unless policy explicitly requires oversight there.
5. If confidence < 0.5, do not report.

SEVERITY
- critical: autonomous customer-facing financial action, autonomous irreversible
  action, autonomous external communication
- high: autonomous action in high-stakes internal domain without override path
- medium: autonomous action where policy strongly implies but does not
  explicitly mandate oversight
- low: missing logging of human reviewer identity where required

OUTPUT
Return a JSON array only. No prose, no markdown fences. Each element:
{
  "finding_id": "HL-<n>",
  "severity": "critical|high|medium|low",
  "policy_clause_id": "<clause number>",
  "config_ref": "<config filename>#<yaml path>",
  "evidence": "<verbatim config excerpt, <=200 chars>",
  "action_risk": "customer_facing|financial|irreversible|external_communication|high_stakes_domain|low_stakes_internal",
  "recommendation": "<1 sentence: what oversight to add>",
  "confidence": <0.0-1.0>
}

If no gaps, return [].
""",
)


# ============================================================================
# 2. SHARED STATE
# ============================================================================

class AuditState(TypedDict):
    policy_text: str
    configs: Dict[str, str]                     # {filename: yaml_text}
    findings: Annotated[List[dict], add]        # reducer: concat lists from parallel nodes
    report_json: dict
    report_markdown: str


# ============================================================================
# 3. NODES
# ============================================================================

SAMPLES_DIR = Path(__file__).parent / "samples"
OUT_DIR = Path(__file__).parent / "out"


def ingest_node(state: AuditState) -> dict:
    policy_path = SAMPLES_DIR / "policy.md"
    configs_dir = SAMPLES_DIR / "configs"
    configs = {p.name: p.read_text() for p in sorted(configs_dir.glob("*.yaml"))}
    print(f"[ingest] policy={policy_path.name} configs={list(configs)}")
    return {
        "policy_text": policy_path.read_text(),
        "configs": configs,
        "findings": [],
    }


def _extract_json_array(text: str) -> str | None:
    """Find the first top-level JSON array in `text` via bracket matching.
    Handles nested braces and quoted strings correctly, unlike regex."""
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_json_array(text: str) -> list:
    """Best-effort parse of an agent's JSON-array output.

    Handles: raw JSON, markdown-fenced JSON, JSON embedded in prose,
    and models that emit a single object instead of an array.
    """
    text = (text or "").strip()

    # strip markdown fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    # try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for k in ("findings", "results", "items"):
                if isinstance(parsed.get(k), list):
                    return parsed[k]
            return [parsed]
    except Exception:
        pass

    # bracket-match extract the first top-level array
    chunk = _extract_json_array(text)
    if chunk:
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, list):
                return parsed
        except Exception as e:
            print(f"[warn] extracted JSON parse failed: {e}")

    snippet = text[:300].replace("\n", " ")
    print(f"[warn] no JSON array found. First 300 chars: {snippet!r}")
    return []


# Global concurrency cap + retry-on-rate-limit — required for free-tier Groq
# which enforces 8000 TPM. Serializing calls keeps us safely under the cap.
_CALL_SEMAPHORE = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENCY", "1")))
_MAX_RETRIES = 5


async def _run_agent_with_backoff(agent: Agent, prompt: str) -> str:
    """Call the LLM directly using the agent's instructions as the system
    prompt. Bypasses Runner because the OpenAI Agents SDK's response handling
    is not fully compatible with Groq's gpt-oss models today.
    Applies a global concurrency cap and exponential backoff on rate limits.
    """
    delay = 2.0
    for attempt in range(_MAX_RETRIES):
        async with _CALL_SEMAPHORE:
            try:
                resp = await _client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": agent.instructions},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=8192,
                    # gpt-oss reasoning models: keep the "thinking" short so the
                    # final JSON actually fits in the response budget.
                    extra_body={"reasoning_effort": "low"},
                )
                msg = resp.choices[0].message
                # gpt-oss returns final answer in .content; reasoning tokens
                # go to a separate channel (.reasoning or .reasoning_content).
                # Prefer .content; fall back to reasoning if content is empty.
                text = (getattr(msg, "content", None) or "").strip()
                if not text:
                    text = (
                        getattr(msg, "reasoning", None)
                        or getattr(msg, "reasoning_content", None)
                        or ""
                    ).strip()
                if not text:
                    finish = resp.choices[0].finish_reason
                    print(f"[warn] {agent.name}: empty response (finish_reason={finish})")
                return text
            except Exception as e:
                err = str(e).lower()
                is_rate = "rate limit" in err or "429" in err or "tokens per minute" in err
                if is_rate and attempt < _MAX_RETRIES - 1:
                    print(f"[retry] {agent.name}: rate-limited, sleeping {delay:.1f}s")
                else:
                    raise
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30.0)
    return ""


def make_agent_node(agent: Agent):
    """Factory: wraps a specialist agent as a LangGraph node."""
    async def _node(state: AuditState) -> dict:
        findings: list = []
        for filename, config_text in state["configs"].items():
            prompt = (
                f"POLICY:\n{state['policy_text']}\n\n"
                f"CONFIG_REF: {filename}\n\n"
                f"CONFIG:\n{config_text}\n"
            )
            output_text = await _run_agent_with_backoff(agent, prompt)
            raw = _parse_json_array(output_text)
            for f in raw:
                f["agent"] = agent.name
                f["source_config"] = filename
            findings.extend(raw)
        print(f"[{agent.name}] produced {len(findings)} findings")
        return {"findings": findings}

    return _node


pv_node = make_agent_node(policy_violation_agent)
gg_node = make_agent_node(guardrail_gap_agent)
cl_node = make_agent_node(consent_language_agent)
dh_node = make_agent_node(data_handling_agent)
hl_node = make_agent_node(hitl_gap_agent)


SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
SEV_WEIGHT = {"critical": 25, "high": 10, "medium": 3, "low": 1}
CONFIDENCE_THRESHOLD = 0.6


def synthesize_node(state: AuditState) -> dict:
    findings = [
        f for f in state["findings"]
        if float(f.get("confidence", 1.0)) >= CONFIDENCE_THRESHOLD
    ]
    findings.sort(key=lambda f: (
        SEV_ORDER.get(f.get("severity", "low"), 3),
        -float(f.get("confidence", 0)),
    ))

    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings:
        counts[f.get("severity", "low")] = counts.get(f.get("severity", "low"), 0) + 1

    score = min(
        100,
        sum(SEV_WEIGHT[s] * counts[s] for s in counts),
    )

    report_json = {
        "risk_score": score,
        "counts": counts,
        "total_findings": len(findings),
        "configs_audited": sorted(state["configs"].keys()),
        "findings": findings,
    }
    print(f"[synthesize] score={score} counts={counts}")
    return {"report_json": report_json}


def render_node(state: AuditState) -> dict:
    r = state["report_json"]
    lines = [
        "# AI Governance Audit Report",
        "",
        f"**Overall risk score:** {r['risk_score']} / 100",
        f"**Total findings:** {r['total_findings']}",
        f"**By severity:** critical={r['counts']['critical']}, "
        f"high={r['counts']['high']}, medium={r['counts']['medium']}, "
        f"low={r['counts']['low']}",
        f"**Configs audited:** {', '.join(r['configs_audited'])}",
        "",
        "## Findings (highest severity first)",
        "",
    ]
    for f in r["findings"]:
        sev = f.get("severity", "?").upper()
        cfg = f.get("source_config", "?")
        clause = f.get("policy_clause_id", "?")
        rec = f.get("recommendation", "")
        ev = (f.get("evidence") or "").replace("\n", " ")[:120]
        lines.append(f"### [{sev}] {cfg} — clause {clause}")
        lines.append(f"- **Agent:** {f.get('agent','?')}")
        lines.append(f"- **Evidence:** `{ev}`")
        lines.append(f"- **Recommendation:** {rec}")
        lines.append(f"- **Confidence:** {f.get('confidence','?')}")
        lines.append("")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(r, indent=2))
    md = "\n".join(lines)
    (OUT_DIR / "report.md").write_text(md)
    print(f"[render] wrote {OUT_DIR/'report.json'} and {OUT_DIR/'report.md'}")
    return {"report_markdown": md}


# ============================================================================
# 4. GRAPH
# ============================================================================

def build_graph():
    graph = StateGraph(AuditState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("policy_violation", pv_node)
    graph.add_node("guardrail_gap", gg_node)
    graph.add_node("consent_language", cl_node)
    graph.add_node("data_handling", dh_node)
    graph.add_node("hitl_gap", hl_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("render", render_node)

    graph.add_edge(START, "ingest")

    # fan-out: five specialists run in parallel after ingest
    for name in [
        "policy_violation",
        "guardrail_gap",
        "consent_language",
        "data_handling",
        "hitl_gap",
    ]:
        graph.add_edge("ingest", name)
        graph.add_edge(name, "synthesize")   # fan-in

    graph.add_edge("synthesize", "render")
    graph.add_edge("render", END)

    return graph.compile()


# ============================================================================
# 5. ENTRY POINT
# ============================================================================

async def main():
    app = build_graph()
    final_state = await app.ainvoke({})
    print("\n" + "=" * 72)
    print(final_state["report_markdown"])


if __name__ == "__main__":
    asyncio.run(main())
