try:
    from openenv.core.env_server import create_app
except ImportError:
    from support_triage_env.compat import create_app

import uvicorn

from support_triage_env.models import SupportTriageAction, SupportTriageObservation
from server.environment import SupportTriageEnvironment

app = create_app(
    SupportTriageEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support-triage-env",
)


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
