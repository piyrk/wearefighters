from __future__ import annotations

try:
    from openenv.core.env_server import create_app
except ImportError:
    from support_triage_env.compat import create_app

from support_triage_env.models import SupportTriageAction, SupportTriageObservation
from support_triage_env.server.environment import SupportTriageEnvironment

app = create_app(
    SupportTriageEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support-triage-env",
)
