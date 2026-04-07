from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar
from uuid import uuid4

from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict, Field

ObsT = TypeVar("ObsT")


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    done: bool = False
    reward: float | int | bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    step_count: int = 0
    cum_reward: float = 0.0


@dataclass
class StepResult(Generic[ObsT]):
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
    info: Optional[dict[str, Any]] = None


def _model_dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def create_app(
    env_factory: Callable[[], Any] | Any,
    action_model: type[BaseModel],
    observation_model: type[BaseModel],
    env_name: str,
) -> FastAPI:
    app = FastAPI(title=env_name)
    env = env_factory() if callable(env_factory) else env_factory

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"env_name": env_name, "status": "ready"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"env_name": env_name, "status": "ok"}

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        current_state = env.state() if callable(getattr(env, "state", None)) else env.state
        state_schema = (
            current_state.__class__.model_json_schema()
            if isinstance(current_state, BaseModel)
            else State.model_json_schema()
        )
        return {
            "action": action_model.model_json_schema(),
            "observation": observation_model.model_json_schema(),
            "state": state_schema,
        }

    @app.post("/reset")
    def reset(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        observation = env.reset(**(payload or {}))
        return {
            "observation": _model_dump(observation),
            "reward": observation.reward,
            "done": observation.done,
        }

    @app.post("/step")
    def step(step_payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        action_payload = step_payload.get("action", step_payload)
        action = action_model.model_validate(action_payload)
        result = env.step_result(action) if hasattr(env, "step_result") else env.step(action)

        if isinstance(result, StepResult):
            final = result
        else:
            observation = result
            final = StepResult(
                observation=observation,
                reward=observation.reward,
                done=observation.done,
                info=observation.metadata,
            )

        return {
            "observation": _model_dump(final.observation),
            "reward": final.reward or 0.0,
            "done": final.done,
            "info": final.info or {},
        }

    @app.get("/state")
    def state() -> dict[str, Any]:
        current_state = env.state() if callable(getattr(env, "state", None)) else env.state
        return _model_dump(current_state)

    return app
