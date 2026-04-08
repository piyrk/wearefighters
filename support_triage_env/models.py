from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.interfaces import Action, Observation, State
except ImportError:
    from .compat import Action, Observation, State

Difficulty = Literal["easy", "medium", "hard"]


class TypedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TicketCategory(str, Enum):
    ACCOUNT_ACCESS = "account_access"
    BILLING_REFUND = "billing_refund"
    ACCOUNT_SECURITY = "account_security"


class TicketPriority(str, Enum):
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class QueueName(str, Enum):
    ACCOUNT_SUPPORT = "account_support"
    BILLING_OPS = "billing_ops"
    TRUST_SAFETY = "trust_safety"


class ResponseTemplate(str, Enum):
    PASSWORD_RESET = "password_reset"
    REFUND_REVIEW = "refund_review"
    SECURITY_HOLD = "security_hold"


class RewardBreakdown(TypedModel):
    step_penalty: float = 0.0
    progress_reward: float = 0.0
    regression_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    submit_bonus: float = 0.0
    total: float = 0.0


class SupportTicket(TypedModel):
    channel: str
    customer_tier: str
    subject: str
    body: str
    account_age_days: int
    prior_contacts: int
    order_value_usd: float | None = None


class TicketWorkspace(TypedModel):
    category: TicketCategory | None = None
    priority: TicketPriority | None = None
    route_to: QueueName | None = None
    template: ResponseTemplate | None = None
    requires_escalation: bool | None = None
    requires_refund: bool | None = None


class SupportTaskAnswer(TicketWorkspace):
    pass


class SupportTaskSpec(TypedModel):
    id: str
    title: str
    difficulty: Difficulty
    goal: str
    policy_notes: list[str]
    ticket: SupportTicket
    answer: SupportTaskAnswer


class SupportTriageAction(Action):
    category: TicketCategory | None = None
    priority: TicketPriority | None = None
    route_to: QueueName | None = None
    template: ResponseTemplate | None = None
    requires_escalation: bool | None = None
    requires_refund: bool | None = None
    submit: bool = False


class SupportTriageObservation(Observation):
    task_id: str
    title: str
    difficulty: Difficulty
    goal: str
    ticket: SupportTicket
    policy_notes: list[str]
    workspace: TicketWorkspace
    remaining_fields: list[str] = Field(default_factory=list)
    progress_score: float = 0.0
    last_feedback: str = ""
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)

    def __iter__(self):
        yield self
        yield self.reward
        yield self.done
        yield self.metadata


class SupportTriageState(State):
    cum_reward: float = 0.0
    current_task_id: str | None = None
    task_cursor: int = 0
    max_steps: int = 6
    best_score: float = 0.0
    submitted: bool = False
    workspace: TicketWorkspace = Field(default_factory=TicketWorkspace)
