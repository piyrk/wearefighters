# Chosen Problem Statement

Build an OpenEnv-compatible environment that simulates a helpdesk analyst triaging real customer support tickets into the correct queue, urgency level, and response workflow.

## Why This Is The Best 2-Hour Option

- It is clearly a real-world task, not a toy problem.
- The work is fully text-based, so we avoid browser automation, external systems, and UI complexity.
- The grader can be deterministic because each ticket maps to a structured answer key.
- Reward shaping is easy: give partial credit as the agent fills correct fields and penalize regressions or premature submission.
- Docker, Hugging Face Spaces, and OpenEnv-style APIs are straightforward for this setup.

## Final Scope

Environment name: `support-triage-env`

Core workflow:
- Read a support ticket and policy notes.
- Fill six structured fields in a triage workspace.
- Submit only when the workspace is correct.

Graded fields:
- `category`
- `priority`
- `route_to`
- `template`
- `requires_escalation`
- `requires_refund`

Difficulty ladder:
- `easy`: account access issue
- `medium`: billing + refund dispute
- `hard`: urgent account security incident with conflicting signals

