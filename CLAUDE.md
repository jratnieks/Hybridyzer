# Claude Working Agreement

This project uses a peer-agent workflow.

## Shared Memory
- `agent_context.md` is the single source of truth for all decisions, constraints, assumptions, and open questions.
- No decision is considered valid unless written into `agent_context.md`.

## Required Behavior
- Always read `agent_context.md` before responding.
- If it does not exist, create it immediately using a clean, structured template.
- Write proposals, critiques, alternatives, decisions, and questions into `agent_context.md`.
- Do not assume other agents have seen chat output unless it is recorded in the file.

## Peer Review Discipline
- Challenge ideas, including your own prior conclusions.
- If you disagree with an existing entry, record the disagreement explicitly.
- Prefer clarity and correctness over speed.

## Execution
- Any agent may implement.
- If implementing with assumptions, record them explicitly.
- If blocked or uncertain, stop and write questions into `agent_context.md`.

## Style
- Be direct, technical, and concise.
- Reject weak or underspecified ideas.
