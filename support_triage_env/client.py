from __future__ import annotations

from typing import Any

import requests

from .compat import StepResult
from .models import SupportTriageAction, SupportTriageObservation, SupportTriageState


class SupportTriageEnvClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def __enter__(self) -> "SupportTriageEnvClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    def reset(self, task_id: str | None = None) -> StepResult[SupportTriageObservation]:
        payload: dict[str, Any] = {}
        if task_id is not None:
            payload["task_id"] = task_id

        response = self._session.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        reset_payload = response.json()
        observation = SupportTriageObservation.model_validate(reset_payload.get("observation", reset_payload))
        return StepResult(
            observation=observation,
            reward=reset_payload.get("reward", observation.reward),
            done=reset_payload.get("done", observation.done),
            info=observation.metadata,
        )

    def step(self, action: SupportTriageAction) -> StepResult[SupportTriageObservation]:
        response = self._session.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump(mode="json", exclude_none=True)},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        observation = SupportTriageObservation.model_validate(payload["observation"])
        return StepResult(
            observation=observation,
            reward=float(payload["reward"]),
            done=bool(payload["done"]),
            info=payload.get("info", {}),
        )

    def state(self) -> SupportTriageState:
        response = self._session.get(f"{self.base_url}/state", timeout=self.timeout_s)
        response.raise_for_status()
        return SupportTriageState.model_validate(response.json())
