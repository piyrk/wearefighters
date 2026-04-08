from __future__ import annotations

from typing import Any

from .models import RewardBreakdown, SupportTaskSpec, TicketWorkspace

GRADABLE_FIELDS = (
    "category",
    "priority",
    "route_to",
    "template",
    "requires_escalation",
    "requires_refund",
)


def _to_jsonable(value: Any) -> Any:
    return getattr(value, "value", value)


def _bounded_score(correct: int, total: int) -> float:
    # Validator requires task scores to remain strictly inside (0, 1).
    return round((correct + 0.5) / (total + 1), 4)


def _raw_score(correct: int, total: int) -> float:
    return round(correct / total, 4)


def evaluate_workspace(
    task: SupportTaskSpec, workspace: TicketWorkspace
) -> tuple[int, int, dict[str, dict[str, Any]]]:
    details: dict[str, dict[str, Any]] = {}
    correct = 0

    for field_name in GRADABLE_FIELDS:
        expected = getattr(task.answer, field_name)
        actual = getattr(workspace, field_name)
        is_correct = actual == expected
        details[field_name] = {
            "expected": _to_jsonable(expected),
            "actual": _to_jsonable(actual),
            "correct": is_correct,
        }
        if is_correct:
            correct += 1

    return correct, len(GRADABLE_FIELDS), details


def grade_workspace(task: SupportTaskSpec, workspace: TicketWorkspace) -> tuple[float, dict[str, dict[str, Any]]]:
    correct, total, details = evaluate_workspace(task, workspace)
    return _bounded_score(correct, total), details


def raw_workspace_score(task: SupportTaskSpec, workspace: TicketWorkspace) -> float:
    correct, total, _ = evaluate_workspace(task, workspace)
    return _raw_score(correct, total)


def workspace_complete(task: SupportTaskSpec, workspace: TicketWorkspace) -> bool:
    correct, total, _ = evaluate_workspace(task, workspace)
    return correct == total


def remaining_fields(task: SupportTaskSpec, workspace: TicketWorkspace) -> list[str]:
    _, details = grade_workspace(task, workspace)
    return [field_name for field_name, payload in details.items() if not payload["correct"]]


def build_reward(
    *,
    improved_fields: list[str],
    regressed_fields: list[str],
    repeated_fields: list[str],
    submitted: bool,
    solved: bool,
    exhausted_steps: bool,
) -> RewardBreakdown:
    reward = RewardBreakdown()
    reward.step_penalty = -0.02
    reward.progress_reward = 0.12 * len(improved_fields)
    reward.regression_penalty = -0.12 * len(regressed_fields)
    reward.invalid_action_penalty = -0.01 * len(repeated_fields)

    if submitted and solved:
        reward.submit_bonus = 0.4
    elif submitted and not solved:
        reward.invalid_action_penalty += -0.15

    if exhausted_steps and not solved:
        reward.invalid_action_penalty += -0.2

    reward.total = round(
        reward.step_penalty
        + reward.progress_reward
        + reward.regression_penalty
        + reward.invalid_action_penalty
        + reward.submit_bonus,
        4,
    )
    return reward
