# AI Policy & Guardrails Auditor

An agentic system that automates design-time governance review of LLM-based agents.

It ingests an organization's AI usage policy and a set of agent/prompt configurations, runs **five specialist agents in parallel** using the **OpenAI Agents SDK**, orchestrated with **LangGraph**, and produces a prioritized governance report (JSON + Markdown) identifying:

1. **Policy Violations** — configs that directly conflict with a stated policy clause
2. **Guardrail Gaps** — required controls absent from implementations
3. **Consent Language Issues** — unclear, missing, or misplaced disclosures
4. **Data Handling Risks** — sensitive data flowing where policy forbids
5. **Human-in-the-Loop Gaps** — autonomy granted where oversight is required

## Why this exists

Organizations ship LLM features faster than their AI policies can keep up. Policies live in Confluence. Prompts and agent configurations live in code. Nobody systematically checks whether one reflects the other — until an incident forces the question.

This tool makes AI governance **operational** — the same way linters and security scanners work in modern engineering pipelines.

## Architecture

```
        START
          │
      ingest_node
          │
    ┌─────┼─────┬──────┬──────┐
    ▼     ▼     ▼      ▼      ▼
   PV    GG    CL     DH     HL      ← 5 parallel specialists
    └─────┴─────┴──────┴──────┘
          │
     synthesize_node    ← merge, filter, sort, score
          │
      render_node       ← JSON + Markdown output
          │
         END
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit .env and add your OPENAI_API_KEY
python auditor.py
```

Outputs land in `out/report.json` and `out/report.md`.

## What ships

- `auditor.py` — the full runnable orchestrator (agents + graph + synthesizer + renderer)
- `samples/policy.md` — a realistic fictional AI usage policy
- `samples/configs/*.yaml` — four agent configs (three dirty, one clean baseline) with planted violations for demo

## Planted violations (for smoke-testing the audit)

The sample configs contain approximately 15 deliberate policy violations across
the four sample agents, spanning all five governance categories. The clean
baseline (`code-review-agent.yaml`) should produce ~0 findings.

## Non-goals

- Not a substitute for legal review or formal AI risk assessment
- Not a runtime guardrail enforcer — this is **design-time** auditing
- Not tied to NIST AI RMF, EU AI Act, or ISO 42001 in v1 (natural v2 extension)

## License

MIT
