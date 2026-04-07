"""
inference.py — Baseline inference script for Email Triage OpenEnv.

Uses the OpenAI API client to run a model against all 3 tasks.
Reads credentials from environment variables.
Emits structured [START], [STEP], [END] logs to stdout.

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="your_hf_token"
    python inference.py
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN)

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASK_IDS = ["task1", "task2", "task3"]

# ─────────────────────────────────────────────
# OpenAI client
# ─────────────────────────────────────────────
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL,
)

# ─────────────────────────────────────────────
# Environment API helpers
# ─────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def env_step(task_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_BASE_URL}/step", json={"task_id": task_id, "action": action})
    resp.raise_for_status()
    return resp.json()


def env_state(task_id: str) -> Dict[str, Any]:
    resp = requests.get(f"{ENV_BASE_URL}/state", params={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def env_task_spec(task_id: str) -> Dict[str, Any]:
    resp = requests.get(f"{ENV_BASE_URL}/tasks/{task_id}/spec")
    resp.raise_for_status()
    return resp.json()

# ─────────────────────────────────────────────
# LLM agent logic
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage assistant. Your job is to classify professional emails.

For each email, you must respond with a JSON object containing:
{
  "email_id": "<the email ID>",
  "label": "<one of: urgent, normal, low, spam, archive>",
  "priority": <integer 1-5 where 1=highest, 5=lowest>,
  "assign_to": "<one of: sales, security, engineering, hr, legal, general> (only if routing rules are provided)",
  "reasoning": "<brief explanation>"
}

Label guidelines:
- urgent: immediate action required, production issues, security breaches, C-level requests, legal deadlines
- normal: standard business tasks, invoices, project updates, meetings
- low: social, non-critical, can wait days
- spam: unsolicited commercial email, phishing, scams
- archive: newsletters, shipping confirmations, FYIs that need no action

Priority guidelines:
- 1: Must handle within the hour (outages, security, board/CEO, legal deadlines)
- 2: Handle today (important business matters, pending decisions)
- 3: Handle this week (routine tasks, non-urgent requests)
- 4: Handle when convenient (low-stakes, social)
- 5: No action needed (spam, newsletters, auto-notifications)

Respond ONLY with the JSON object. No preamble, no explanation outside the JSON."""


def build_user_prompt(obs: Dict[str, Any], email: Dict[str, Any], task_spec: Dict[str, Any]) -> str:
    rules = obs.get("rules_context") or ""
    rules_section = f"\n\nROUTING RULES:\n{rules}" if rules else ""

    prompt = f"""Task: {task_spec['name']} ({task_spec['difficulty']} difficulty)
{task_spec['description']}{rules_section}

EMAIL TO TRIAGE:
ID: {email['id']}
From: {email['sender']}
Subject: {email['subject']}
Timestamp: {email['timestamp']}
Has Attachment: {email['has_attachment']}
Thread Length: {email['thread_length']}

Body:
{email['body']}

Processed so far: {obs['processed_count']}/{obs['total_emails']}
Labels assigned so far: {json.dumps(obs.get('inbox_labels', {}), indent=2)}

Respond with JSON only."""
    return prompt


def call_llm(system: str, user: str) -> Optional[str]:
    """Call the LLM and return response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
        return None


def parse_action(response_text: str, email_id: str, has_routing: bool) -> Dict[str, Any]:
    """Parse LLM response into an action dict. Fall back to defaults on error."""
    try:
        # Strip markdown fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        action = json.loads(text.strip())
        # Ensure required fields
        action.setdefault("email_id", email_id)
        action.setdefault("label", "normal")
        action.setdefault("priority", 3)
        if not has_routing:
            action.pop("assign_to", None)
        # Clamp priority
        action["priority"] = max(1, min(5, int(action["priority"])))
        return action
    except Exception as e:
        print(f"[WARN] Failed to parse LLM response: {e}. Using fallback.", file=sys.stderr)
        action = {
            "email_id": email_id,
            "label": "normal",
            "priority": 3,
            "reasoning": "parse_error_fallback",
        }
        return action


# ─────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────

def run_task(task_id: str) -> float:
    """Run the agent on one task. Return the final episode score."""
    task_spec = env_task_spec(task_id)
    obs = env_reset(task_id)
    has_routing = task_spec.get("has_routing_rules", False)

    step_num = 0
    total_reward = 0.0
    done = obs.get("done", False)

    emails = obs.get("emails", [])

    while not done and step_num < task_spec["max_steps"]:
        current_idx = obs.get("current_email_index", step_num)
        if current_idx >= len(emails):
            break

        email = emails[current_idx]
        user_prompt = build_user_prompt(obs, email, task_spec)

        # LLM call
        llm_response = call_llm(SYSTEM_PROMPT, user_prompt)
        if llm_response is None:
            # Default safe action
            action = {
                "email_id": email["id"],
                "label": "normal",
                "priority": 3,
                "reasoning": "llm_error_fallback",
            }
        else:
            action = parse_action(llm_response, email["id"], has_routing)

        # Step the environment
        result = env_step(task_id, action)
        step_reward = result["reward"]["value"]
        done = result["done"]
        obs = result["observation"]
        info = result.get("info", {})

        step_num += 1
        total_reward += step_reward

        # ── [STEP] log ──
        print(json.dumps({
            "type": "STEP",
            "task_id": task_id,
            "step": step_num,
            "email_id": action["email_id"],
            "label": action["label"],
            "priority": action["priority"],
            "assign_to": action.get("assign_to"),
            "reward": step_reward,
            "label_correct": result["reward"]["label_correct"],
            "priority_correct": result["reward"]["priority_correct"],
            "cumulative_reward": info.get("cumulative_reward", total_reward),
            "done": done,
        }))
        sys.stdout.flush()

    # Get final state for episode score
    final_state = env_state(task_id)
    episode_score = final_state.get("episode_score", 0.0)
    return episode_score


def main():
    print("=" * 60, file=sys.stderr)
    print(f"Email Triage OpenEnv — Baseline Inference", file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    print(f"API Base: {API_BASE_URL}", file=sys.stderr)
    print(f"Environment: {ENV_BASE_URL}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # ── [START] ──
    print(json.dumps({
        "type": "START",
        "model": MODEL_NAME,
        "tasks": TASK_IDS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    sys.stdout.flush()

    task_scores: Dict[str, float] = {}

    for task_id in TASK_IDS:
        print(f"\n[INFO] Running task: {task_id}", file=sys.stderr)
        try:
            score = run_task(task_id)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
            score = 0.0
        task_scores[task_id] = score
        print(f"[INFO] {task_id} score: {score:.4f}", file=sys.stderr)

    overall_score = sum(task_scores.values()) / len(task_scores)

    # ── [END] ──
    print(json.dumps({
        "type": "END",
        "model": MODEL_NAME,
        "task_scores": task_scores,
        "overall_score": round(overall_score, 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    sys.stdout.flush()

    print("\n" + "=" * 60, file=sys.stderr)
    print("RESULTS:", file=sys.stderr)
    for tid, score in task_scores.items():
        print(f"  {tid}: {score:.4f}", file=sys.stderr)
    print(f"  OVERALL: {overall_score:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
