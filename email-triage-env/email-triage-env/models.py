"""
Typed Pydantic models for the Email Triage OpenEnv environment.
Observation, Action, Reward — full OpenEnv spec compliance.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    has_attachment: bool = False
    thread_length: int = 1


class Observation(BaseModel):
    """Returned by step() and reset(). Contains current inbox state."""
    task_id: str
    step_number: int
    emails: List[Email]
    processed_count: int
    total_emails: int
    current_email_index: int
    inbox_labels: Dict[str, str] = Field(default_factory=dict)  # email_id -> label assigned
    rules_context: Optional[str] = None  # For hard task: active routing rules
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """Action the agent takes. Triage a single email."""
    email_id: str
    label: str  # "urgent", "normal", "low", "spam", "archive"
    priority: int = Field(ge=1, le=5, description="1=highest priority, 5=lowest")
    assign_to: Optional[str] = None  # For hard task: route to team
    reply_draft: Optional[str] = None  # Optional: draft a reply snippet
    reasoning: Optional[str] = None  # Agent's reasoning (logged, not graded)


class Reward(BaseModel):
    """Reward signal returned by step()."""
    value: float = Field(ge=0.0, le=1.0)
    label_correct: bool
    priority_correct: bool
    partial_credit: float  # 0.0–1.0 partial progress component
    penalties: float  # Deductions for bad behavior
    breakdown: Dict[str, float] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    """Metadata for a single task."""
    task_id: str
    name: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    max_steps: int
    num_emails: int
    success_threshold: float = 0.7


class EpisodeState(BaseModel):
    """Full episode state returned by state()."""
    task_id: str
    task_spec: TaskSpec
    step_number: int
    emails: List[Email]
    actions_taken: List[Dict[str, Any]]
    rewards_earned: List[float]
    cumulative_reward: float
    done: bool
    episode_score: float
    ground_truth: Dict[str, Any] = Field(default_factory=dict)
