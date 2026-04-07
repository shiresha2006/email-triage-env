"""
EmailTriageEnv — Core OpenEnv environment implementation.
Implements step(), reset(), state() with typed Pydantic models.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action, Email, EpisodeState, Observation, Reward, TaskSpec
)
from data.email_data import TASK_SPECS


VALID_LABELS = {"urgent", "normal", "low", "spam", "archive"}
VALID_TEAMS = {"sales", "security", "engineering", "hr", "legal", "general"}


class EmailTriageEnv:
    """
    Real-world email triage environment for RL/agent evaluation.

    The agent receives a professional inbox and must classify each email
    with a label (urgent/normal/low/spam/archive), a priority (1-5),
    and optionally route it to a team (hard task).

    Reward is shaped across the trajectory — not sparse end-of-episode.
    """

    LABEL_WEIGHTS = {
        "urgent":  1.0,
        "normal":  1.0,
        "low":     1.0,
        "spam":    1.0,
        "archive": 1.0,
    }

    # Priority tolerance: off by 1 gets partial credit
    PRIORITY_EXACT_REWARD = 0.3
    PRIORITY_CLOSE_REWARD = 0.15  # within ±1
    LABEL_REWARD = 0.5
    ROUTING_REWARD = 0.2  # Only for task3

    def __init__(self, task_id: str = "task1"):
        assert task_id in TASK_SPECS, f"Unknown task_id '{task_id}'. Valid: {list(TASK_SPECS)}"
        self.task_id = task_id
        self._task_data = TASK_SPECS[task_id]
        self._task_spec = TaskSpec(**{
            k: v for k, v in self._task_data.items()
            if k in TaskSpec.model_fields
        })

        # Episode state
        self._step_number: int = 0
        self._emails: List[Email] = []
        self._email_index: int = 0
        self._actions_taken: List[Dict[str, Any]] = []
        self._rewards_earned: List[float] = []
        self._cumulative_reward: float = 0.0
        self._inbox_labels: Dict[str, str] = {}
        self._done: bool = False

    # ─────────────────────────────────────────
    # OpenEnv Interface
    # ─────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment to initial state. Returns first observation."""
        self._step_number = 0
        self._email_index = 0
        self._actions_taken = []
        self._rewards_earned = []
        self._cumulative_reward = 0.0
        self._inbox_labels = {}
        self._done = False

        # Build Email objects from task data
        self._emails = [
            Email(**e) for e in self._task_data["emails"]
        ]

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one agent action (triaging one email).

        Returns:
            observation: Updated inbox state
            reward: Shaped reward signal
            done: Whether episode is complete
            info: Diagnostic info
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        self._step_number += 1

        # Validate action
        info: Dict[str, Any] = {"step": self._step_number}
        invalid_reason = self._validate_action(action)
        if invalid_reason:
            reward = Reward(
                value=0.0,
                label_correct=False,
                priority_correct=False,
                partial_credit=0.0,
                penalties=0.1,
                breakdown={"invalid_action": -0.1, "reason": invalid_reason},
            )
            self._rewards_earned.append(0.0)
            self._cumulative_reward += 0.0
            info["invalid_action"] = invalid_reason
            obs = self._make_observation()
            return obs, reward, self._done, info

        # Compute reward for this action
        reward = self._compute_reward(action)

        # Record action
        self._actions_taken.append(action.model_dump())
        self._inbox_labels[action.email_id] = action.label
        self._rewards_earned.append(reward.value)
        self._cumulative_reward += reward.value

        # Advance email pointer
        self._email_index += 1

        # Check termination
        max_reached = self._step_number >= self._task_spec.max_steps
        all_processed = self._email_index >= len(self._emails)
        self._done = max_reached or all_processed

        obs = self._make_observation()
        info.update({
            "email_id": action.email_id,
            "label_correct": reward.label_correct,
            "priority_correct": reward.priority_correct,
            "cumulative_reward": self._cumulative_reward,
            "episode_score": self._episode_score(),
        })

        return obs, reward, self._done, info

    def state(self) -> EpisodeState:
        """Return the full current episode state."""
        return EpisodeState(
            task_id=self.task_id,
            task_spec=self._task_spec,
            step_number=self._step_number,
            emails=self._emails,
            actions_taken=self._actions_taken,
            rewards_earned=self._rewards_earned,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            episode_score=self._episode_score(),
            ground_truth=self._task_data["ground_truth"],
        )

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _make_observation(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            step_number=self._step_number,
            emails=self._emails,
            processed_count=self._email_index,
            total_emails=len(self._emails),
            current_email_index=self._email_index,
            inbox_labels=dict(self._inbox_labels),
            rules_context=self._task_data.get("rules_context"),
            done=self._done,
            info={
                "cumulative_reward": self._cumulative_reward,
                "steps_remaining": self._task_spec.max_steps - self._step_number,
                "task_description": self._task_spec.description,
            },
        )

    def _validate_action(self, action: Action) -> Optional[str]:
        """Return an error string if action is invalid, else None."""
        if action.label not in VALID_LABELS:
            return f"Invalid label '{action.label}'. Must be one of {VALID_LABELS}"
        gt = self._task_data["ground_truth"]
        if action.email_id not in gt:
            return f"Unknown email_id '{action.email_id}'"
        if self.task_id == "task3" and action.assign_to and action.assign_to not in VALID_TEAMS:
            return f"Invalid team '{action.assign_to}'. Must be one of {VALID_TEAMS}"
        return None

    def _compute_reward(self, action: Action) -> Reward:
        """
        Compute shaped reward for this action.

        Reward breakdown:
        - Label correctness:   0.50 (binary)
        - Priority exact:      0.30 (exact match)
        - Priority close:      0.15 (off by 1)
        - Team routing:        0.20 (task3 only, else folded into label)
        - Penalty: spam as urgent or urgent as spam: -0.20
        - Penalty: per step beyond halfway mark: -0.01
        """
        gt = self._task_data["ground_truth"][action.email_id]
        breakdown: Dict[str, float] = {}

        # ── Label component ──
        label_correct = (action.label == gt["label"])
        label_reward = self.LABEL_REWARD if label_correct else 0.0
        breakdown["label"] = label_reward

        # ── Priority component ──
        priority_diff = abs(action.priority - gt["priority"])
        if priority_diff == 0:
            priority_reward = self.PRIORITY_EXACT_REWARD
            priority_correct = True
        elif priority_diff == 1:
            priority_reward = self.PRIORITY_CLOSE_REWARD
            priority_correct = False
        else:
            priority_reward = 0.0
            priority_correct = False
        breakdown["priority"] = priority_reward

        # ── Routing component (task3 only) ──
        routing_reward = 0.0
        if self.task_id == "task3" and "assign_to" in gt:
            if action.assign_to == gt["assign_to"]:
                routing_reward = self.ROUTING_REWARD
        breakdown["routing"] = routing_reward

        # ── Penalties ──
        penalties = 0.0
        # Critical misclassification: calling spam "urgent" or urgent "spam"
        if gt["label"] == "urgent" and action.label == "spam":
            penalties += 0.3
            breakdown["critical_miss_penalty"] = -0.3
        elif gt["label"] == "spam" and action.label == "urgent":
            penalties += 0.2
            breakdown["false_alarm_penalty"] = -0.2

        # Step efficiency penalty: small penalty after halfway
        halfway = self._task_spec.max_steps // 2
        if self._step_number > halfway:
            step_penalty = 0.01 * (self._step_number - halfway)
            penalties += step_penalty
            breakdown["efficiency_penalty"] = -step_penalty

        # ── Final reward ──
        raw = label_reward + priority_reward + routing_reward - penalties

        # Normalize to [0, 1]
        max_possible = self.LABEL_REWARD + self.PRIORITY_EXACT_REWARD
        if self.task_id == "task3":
            max_possible += self.ROUTING_REWARD
        value = max(0.0, min(1.0, raw / max_possible))

        breakdown["raw"] = raw
        breakdown["max_possible"] = max_possible

        return Reward(
            value=round(value, 4),
            label_correct=label_correct,
            priority_correct=priority_correct,
            partial_credit=round((label_reward + priority_reward) / max_possible, 4),
            penalties=round(penalties, 4),
            breakdown=breakdown,
        )

    def _episode_score(self) -> float:
        """Compute normalized episode score as fraction of max possible reward."""
        if not self._rewards_earned:
            return 0.0
        total_possible = len(self._emails)  # 1.0 per email at best
        score = self._cumulative_reward / total_possible
        return round(min(1.0, score), 4)


def run_task_grader(task_id: str, actions: List[Dict[str, Any]]) -> float:
    """
    Standalone grader: given a task_id and list of action dicts, return final score 0.0–1.0.
    Used for evaluation and validation.
    """
    env = EmailTriageEnv(task_id=task_id)
    env.reset()
    for action_dict in actions:
        if env._done:
            break
        action = Action(**action_dict)
        env.step(action)
    return env._episode_score()
