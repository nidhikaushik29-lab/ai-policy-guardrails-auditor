# Acme Corp — AI Usage Policy
**Version:** 1.4
**Effective:** 2026-01-15
**Owner:** AI Governance Council

## 1. Purpose and Scope

This policy governs the design, deployment, and operation of all AI/LLM-based agents and features developed at Acme Corp. It applies to all internal and customer-facing AI systems.

## 2. Definitions

- **Agent:** Any system that uses an LLM to make decisions, take actions, or generate content on behalf of Acme or its users.
- **Sensitive Data:** PII, PHI, financial records, credentials, source code marked confidential, and any customer data classified Restricted or higher.
- **Autonomous Action:** Any action taken by an agent that changes external state without a human confirming the specific action.

## 3. Prohibited Use Cases

3.1 Agents shall not be used to make final decisions on hiring, firing, promotion, compensation, or performance management of Acme employees.

3.2 Agents shall not be used to generate legal advice, medical advice, or financial advice presented as authoritative to end users.

3.3 Agents shall not be deployed in workflows that impersonate a human without disclosure to the end user.

## 4. Required Guardrails

4.1 All agents processing user-submitted text shall apply a PII redaction step before the text is sent to an LLM provider.

4.2 All agents with tool-calling capability shall enforce an allowlist of permitted tools. Wildcard or unrestricted tool access is prohibited.

4.3 All agents shall implement prompt-injection defenses on any input originating from an untrusted source.

4.4 All agents shall enforce a maximum output token limit appropriate to their use case, not exceeding provider defaults.

## 5. Consent and Disclosure

5.1 Any user-facing surface where an AI agent generates content shall display a clear disclosure that the content is AI-generated. The disclosure shall appear before or alongside the content, not buried in footers or help pages.

5.2 Where an agent collects, stores, or processes user input for model training or evaluation, explicit opt-in consent shall be obtained. Opt-out defaults are not permitted for training data collection.

5.3 Consent language shall be written at or below a US 8th-grade reading level and shall state (a) what is collected, (b) how it is used, (c) how long it is retained, and (d) how the user can withdraw consent.

## 6. Data Handling

6.1 Sensitive Data as defined in Section 2 shall not be sent to third-party LLM providers unless a signed Data Processing Agreement is in place and the specific data category is covered by that agreement.

6.2 Prompt and response logs containing user data shall be retained no longer than 30 days unless a specific legal hold applies.

6.3 Agents shall not persist user data outside approved storage systems. Ephemeral in-memory processing is permitted; writing to local disk or unapproved cloud storage is prohibited.

6.4 Cross-region data transfer for AI processing shall follow Acme's data residency policy. EU user data shall be processed in EU regions only.

## 7. Human-in-the-Loop Requirements

7.1 Agents shall require human approval before taking any Autonomous Action that is customer-facing, financial, or irreversible.

7.2 Agents that draft communications to external parties shall require human review before send.

7.3 Agents operating in high-stakes domains (billing, account changes, security responses) shall log every decision and provide a human override path within one click.

7.4 Fully autonomous operation is permitted only for low-stakes, reversible, internal-only tasks explicitly approved by the AI Governance Council.

## 8. Logging and Auditability

8.1 All agent invocations shall log: timestamp, agent ID, input hash, output hash, tools invoked, and the human reviewer (if applicable).

8.2 Logs shall be tamper-evident and retained for 12 months minimum.

## 9. Review and Enforcement

9.1 Every new agent or material change to an existing agent requires review by the AI Governance Council prior to production deployment.

9.2 Violations of this policy may result in the agent being taken offline pending remediation.
