from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    Environment = object  # type: ignore[assignment]

from support_triage_env.graders import GRADABLE_FIELDS, build_reward, grade_workspace, remaining_fields

from support_triage_env.compat import StepResult
from support_triage_env.models import (
    RewardBreakdown,
    SupportTaskSpec,
    SupportTriageAction,
    SupportTriageObservation,
    SupportTriageState,
    TicketWorkspace,
)
from support_triage_env.tasks import load_tasks


class SupportTriageEnvironment(Environment):
    def __init__(self, max_steps: int = 6):
        self._tasks = load_tasks()
        self._task_ids = list(self._tasks.keys())
        self._max_steps = max_steps
        self._task: SupportTaskSpec | None = None
        self._state = SupportTriageState(max_steps=max_steps)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str | None = None,
        **_: Any,
    ) -> SupportTriageObservation:
        selected_id = task_id or self._task_ids[0]
        if selected_id not in self._tasks:
            raise ValueError(f"Unknown task_id: {selected_id}")

        self._task = self._tasks[selected_id]
        self._state = SupportTriageState(
            episode_id=episode_id or str(uuid4()),
            current_task_id=selected_id,
            task_cursor=self._task_ids.index(selected_id),
            max_steps=self._max_steps,
            workspace=TicketWorkspace(),
        )
        return self._build_observation(
            feedback="Workspace reset. Read the ticket, apply policy notes, and submit only when all fields are correct.",
            reward_breakdown=RewardBreakdown(),
            done=False,
        )

    @property
    def state(self) -> SupportTriageState:
        return self._state

    def get_state(self) -> SupportTriageState:
        return self._state

    def step_result(self, action: SupportTriageAction) -> StepResult[SupportTriageObservation]:
        if self._task is None:
            self.reset()

        assert self._task is not None

        before_score, before_details = grade_workspace(self._task, self._state.workspace)
        workspace = self._state.workspace
        repeated_fields: list[str] = []

        for field_name in GRADABLE_FIELDS:
            new_value = getattr(action, field_name)
            if new_value is None:
                continue

            current_value = getattr(workspace, field_name)
            if current_value == new_value:
                repeated_fields.append(field_name)
                continue

            workspace = workspace.model_copy(update={field_name: new_value})

        self._state.workspace = workspace
        self._state.step_count += 1

        after_score, after_details = grade_workspace(self._task, self._state.workspace)
        improved_fields = [
            field_name
            for field_name in GRADABLE_FIELDS
            if after_details[field_name]["correct"] and not before_details[field_name]["correct"]
        ]
        regressed_fields = [
            field_name
            for field_name in GRADABLE_FIELDS
            if before_details[field_name]["correct"] and not after_details[field_name]["correct"]
        ]

        solved = after_score == 1.0 and action.submit
        exhausted_steps = self._state.step_count >= self._state.max_steps and not solved
        reward_breakdown = build_reward(
            improved_fields=improved_fields,
            regressed_fields=regressed_fields,
            repeated_fields=repeated_fields,
            submitted=action.submit,
            solved=solved,
            exhausted_steps=exhausted_steps,
        )

        self._state.cum_reward = round(self._state.cum_reward + reward_breakdown.total, 4)
        self._state.best_score = max(self._state.best_score, after_score)
        self._state.submitted = solved

        remaining = remaining_fields(self._task, self._state.workspace)
        if solved:
            feedback = "Submission accepted. All triage fields are correct."
        elif action.submit:
            feedback = f"Submission rejected. Review these fields: {', '.join(remaining)}."
        elif improved_fields:
            feedback = f"Progress made. Newly correct fields: {', '.join(improved_fields)}."
        elif regressed_fields:
            feedback = f"Regression detected. Re-check: {', '.join(regressed_fields)}."
        elif repeated_fields:
            feedback = "No-op update detected. Change the workspace before submitting."
        else:
            feedback = "Workspace updated, but more corrections are still needed."

        if exhausted_steps and not solved:
            if remaining:
                feedback = f"Step limit reached. Episode ended with remaining fields: {', '.join(remaining)}."
            else:
                feedback = "Step limit reached before final submission."

        observation = self._build_observation(
            feedback=feedback,
            reward_breakdown=reward_breakdown,
            done=solved or exhausted_steps,
        )
        info: dict[str, Any] = {
            "grader_score": round(after_score, 4),
            "improved_fields": improved_fields,
            "regressed_fields": regressed_fields,
            "repeated_fields": repeated_fields,
            "remaining_fields": remaining,
        }
        return StepResult(
            observation=observation,
            reward=reward_breakdown.total,
            done=observation.done,
            info=info,
        )

    def step(self, action: SupportTriageAction) -> SupportTriageObservation:
        result = self.step_result(action)
        observation = result.observation.model_copy(
            update={
                "metadata": {
                    **result.observation.metadata,
                    **(result.info or {}),
                }
            }
        )
        return observation

    def close(self) -> None:
        return None

    def _build_observation(
        self,
        *,
        feedback: str,
        reward_breakdown: RewardBreakdown,
        done: bool,
    ) -> SupportTriageObservation:
        assert self._task is not None

        score, _ = grade_workspace(self._task, self._state.workspace)
        return SupportTriageObservation(
            task_id=self._task.id,
            title=self._task.title,
            difficulty=self._task.difficulty,
            goal=self._task.goal,
            ticket=self._task.ticket,
            policy_notes=self._task.policy_notes,
            workspace=self._state.workspace,
            remaining_fields=remaining_fields(self._task, self._state.workspace),
            progress_score=round(score, 4),
            last_feedback=feedback,
            reward_breakdown=reward_breakdown,
            reward=reward_breakdown.total,
            done=done,
            metadata={
                "title": self._task.title,
                "max_steps": self._state.max_steps,
                "step_count": self._state.step_count,
                "best_score": round(self._state.best_score, 4),
            },
        )
