from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from support_triage_env.models import SupportTriageAction
from support_triage_env.server.environment import SupportTriageEnvironment
from support_triage_env.tasks import load_tasks


def build_prompt(observation: Any) -> str:
    return (
        "You are a customer support triage analyst.\n"
        "Read the ticket and policy notes, then return JSON only.\n"
        "Choose exactly one value for each structured field.\n\n"
        "Allowed values:\n"
        "- category: account_access, billing_refund, account_security\n"
        "- priority: medium, high, urgent\n"
        "- route_to: account_support, billing_ops, trust_safety\n"
        "- template: password_reset, refund_review, security_hold\n"
        "- requires_escalation: true or false\n"
        "- requires_refund: true or false\n"
        "- submit: false\n\n"
        f"Task ID: {observation.task_id}\n"
        f"Difficulty: {observation.difficulty}\n"
        f"Goal: {observation.goal}\n"
        f"Policy Notes: {json.dumps(observation.policy_notes, indent=2)}\n"
        f"Ticket: {observation.ticket.model_dump_json(indent=2)}\n"
        f"Workspace: {observation.workspace.model_dump_json(indent=2)}\n"
    )


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(match.group(0))


def score_task(client: OpenAI, model: str, task_id: str) -> dict[str, Any]:
    env = SupportTriageEnvironment()
    observation = env.reset(task_id=task_id)

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Return strict JSON only. Do not include markdown fences.",
            },
            {"role": "user", "content": build_prompt(observation)},
        ],
    )

    raw_text = completion.choices[0].message.content or "{}"
    payload = extract_json(raw_text)
    payload["submit"] = False

    action = SupportTriageAction.model_validate(payload)
    env.step_result(action)
    final_result = env.step_result(SupportTriageAction(submit=True))

    return {
        "task_id": task_id,
        "difficulty": observation.difficulty,
        "score": final_result.info["grader_score"],
        "done": final_result.done,
        "raw_model_output": raw_text,
        "final_feedback": final_result.observation.last_feedback,
        "steps": env.state.step_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an OpenAI-client baseline on the support triage environment.")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME"), help="Model name exposed by the API endpoint.")
    parser.add_argument(
        "--output",
        default="outputs/evals/baseline_scores.generated.json",
        help="Where to write the generated scores.",
    )
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("Missing model name. Pass --model or set MODEL_NAME.")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Missing HF_TOKEN environment variable.")

    base_url = os.getenv("API_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://router.huggingface.co/v1"))
    client = OpenAI(api_key=token, base_url=base_url)

    tasks = load_tasks()
    results = [score_task(client, args.model, task_id) for task_id in tasks]
    average_score = sum(item["score"] for item in results) / len(results)

    output = {
        "model": args.model,
        "base_url": base_url,
        "task_count": len(results),
        "average_score": round(average_score, 4),
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
