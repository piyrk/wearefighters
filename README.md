---
title: Support Triage OpenEnv
sdk: docker
app_port: 8000
tags:
  - openenv
  - customer-support
  - reinforcement-learning
  - fastapi
pinned: false
---

# Support Triage OpenEnv

`support-triage-env` is a compact OpenEnv-style environment for customer support ticket triage. The agent receives a realistic inbound support ticket plus internal policy notes, fills a structured workspace, and submits a decision only when all required routing fields are correct.

This problem statement is the best fit for a fast hackathon submission because it is clearly real-world, deterministic to grade, and small enough to package with Docker, docs, and a baseline script in a short time window.

## Environment Overview

### Motivation

Support teams handle large volumes of repetitive routing work: deciding whether a ticket belongs to billing, shipping, account support, or trust and safety; how urgent it is; whether escalation is needed; and which response macro should be used. This environment turns that workflow into a reproducible agent benchmark.

### Objective

For each task, the agent must classify the ticket into the correct:
- category
- priority
- destination queue
- response template
- escalation flag
- refund flag

The episode ends successfully only after the agent submits a fully correct workspace.

## Action Space

The action model is `SupportTriageAction`.

Fields the agent may set:
- `category`
- `priority`
- `route_to`
- `template`
- `requires_escalation`
- `requires_refund`
- `submit`

The agent can update some or all fields in one step. `submit=true` attempts to finalize the workspace.

## Observation Space

The observation model is `SupportTriageObservation`.

Each observation includes:
- the ticket payload
- internal policy notes
- the current workspace
- a `progress_score` between `0.0` and `1.0`
- remaining fields still needing correction
- shaped reward details
- the latest feedback message

## Reward Function

The environment rewards progress throughout the trajectory:
- positive reward for each field moved into a correct state
- negative reward for regressing a correct field
- small penalty for repeated/no-op updates
- penalty for premature submission
- bonus for a perfect final submission
- penalty when the agent runs out of steps

This makes the environment useful for RL or stepwise evaluation instead of only terminal scoring.

## Tasks

The repo ships with three tasks of increasing difficulty:

1. `easy_account_access`
Account access issue after a device change. Straightforward routing to account support with no escalation.

2. `medium_billing_refund`
Double charge plus delayed shipment. The policy makes billing the primary owner because payment/refund issues take precedence.

3. `hard_account_security`
Unauthorized purchase plus account takeover signals and public pressure. The correct action is urgent trust-and-safety handling, not immediate refund routing.

## Project Layout

```text
.
+-- Dockerfile
+-- __init__.py
+-- client.py
+-- models.py
+-- openenv.yaml
+-- pyproject.toml
+-- scripts/
|   +-- run_baseline.py
+-- server/
|   +-- __init__.py
|   +-- app.py
|   +-- environment.py
|   +-- Dockerfile
|   +-- requirements.txt
+-- support_triage_env/
    +-- __init__.py
    +-- client.py
    +-- compat.py
    +-- graders.py
    +-- models.py
    +-- tasks.py
    +-- data/
    |   +-- tasks.json
    +-- server/
        +-- __init__.py
        +-- app.py
        +-- environment.py
        +-- Dockerfile
        +-- requirements.txt
```

## Setup

### Local install

```bash
pip install -e .
```

### Run the environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Build with Docker

```bash
docker build -t support-triage-env .
docker run -p 8000:8000 support-triage-env
```

### Validate with OpenEnv

If you have the OpenEnv CLI available:

```bash
openenv validate --verbose
```

## Baseline Inference

The required submission script is [inference.py](C:/Users/Piyush%20Kumar/Desktop/the%20gamechanger/inference.py). It uses the OpenAI client and emits the mandatory `[START]`, `[STEP]`, and `[END]` stdout lines.

Required environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional environment variables:
- `LOCAL_IMAGE_NAME`

Run:

```bash
python inference.py
```

The legacy helper script at `scripts/run_baseline.py` is kept for local experimentation, but the official submission entrypoint is `inference.py`.

Defaults are set only for `API_BASE_URL` and `MODEL_NAME`. `HF_TOKEN` must be explicitly provided.

## Baseline Performance Scores

The current sandbox did not expose a runnable Python interpreter, so I could not execute `inference.py` here and record final numeric baseline scores. The script is configured for deterministic runs with `temperature=0`, and the final scores will be emitted in `[END]` lines during your local run.

## Hugging Face Spaces Notes

This project is container-first and suitable for a Hugging Face Space:
- includes a `Dockerfile`
- includes `openenv.yaml`
- tags include `openenv`
- server entrypoint is `server.app:app`

# wearefighters
