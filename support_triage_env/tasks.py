from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from .models import SupportTaskSpec


@lru_cache(maxsize=1)
def load_tasks() -> dict[str, SupportTaskSpec]:
    raw = resources.files("support_triage_env.data").joinpath("tasks.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    tasks = [SupportTaskSpec.model_validate(item) for item in payload]
    return {task.id: task for task in tasks}

