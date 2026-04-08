from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from openai import OpenAI

from support_triage_env.models import SupportTriageAction
from support_triage_env.server.environment import SupportTriageEnvironment
from support_triage_env.tasks import load_tasks

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_NAME = "support_triage_env"


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def to_bool_str(value: bool) -> str:
    return "true" if value else "false"


def fmt_reward(value: float | int | bool | None) -> str:
    return f"{float(value or 0.0):.2f}"


def bounded_fallback_score() -> float:
    return 0.01


def build_prompt(observation: Any) -> str:
    return (
        "You are a customer support triage analyst.\n"
        "Return strict JSON only. Do not add markdown.\n"
        "Choose exactly one value for each field and set submit=false.\n\n"
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


def action_to_str(action: SupportTriageAction) -> str:
    return json.dumps(
        action.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
        separators=(",", ":"),
    )


def run_task(client: OpenAI, model_name: str, task_id: str) -> None:
    env = SupportTriageEnvironment()
    step_count = 0
    rewards: list[str] = []
    final_score = bounded_fallback_score()
    success = False

    emit(f"[START] task={task_id} env={ENV_NAME} model={model_name}")

    try:
        observation = env.reset(task_id=task_id)
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": build_prompt(observation)},
            ],
        )
        raw_text = completion.choices[0].message.content or "{}"
        payload = extract_json(raw_text)
        payload["submit"] = False

        first_action = SupportTriageAction.model_validate(payload)
        first_result = env.step(first_action)
        step_count += 1
        rewards.append(fmt_reward(first_result.reward))
        emit(
            f"[STEP] step={step_count} action={action_to_str(first_action)} "
            f"reward={fmt_reward(first_result.reward)} done={to_bool_str(first_result.done)} error=null"
        )

        submit_action = SupportTriageAction(submit=True)
        final_result = env.step(submit_action)
        step_count += 1
        rewards.append(fmt_reward(final_result.reward))
        emit(
            f"[STEP] step={step_count} action={action_to_str(submit_action)} "
            f"reward={fmt_reward(final_result.reward)} done={to_bool_str(final_result.done)} error=null"
        )

        final_score = float(final_result.metadata.get("grader_score", bounded_fallback_score()))
        success = bool(final_result.metadata.get("task_completed", False))
    except Exception:
        pass
    finally:
        env.close()
        emit(
            f"[END] success={to_bool_str(success)} steps={step_count} "
            f"score={final_score:.2f} rewards={','.join(rewards) if rewards else '0.00'}"
        )


def main() -> None:
    if not HF_TOKEN:
        raise SystemExit("Missing HF_TOKEN environment variable.")

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    for task_id in load_tasks():
        run_task(client, MODEL_NAME, task_id)


if __name__ == "__main__":
    main()
