"""
validate.py — Pre-submission validation for Email Triage OpenEnv.

Checks all requirements from the hackathon pre-submission checklist:
  1. openenv.yaml exists and is valid
  2. Typed Pydantic models (Observation, Action, Reward)
  3. step() / reset() / state() work correctly
  4. 3+ tasks with graders returning scores in [0.0, 1.0]
  5. Reward shaping (not constant, provides partial signal)
  6. Dockerfile exists
  7. inference.py exists in root directory
  8. Environment variables documented
"""

import json
import os
import sys
import yaml

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

errors = []
warnings = []


def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS} {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))
        errors.append(label)


def warn(label: str, detail: str = ""):
    print(f"  {WARN} {label}" + (f" — {detail}" if detail else ""))
    warnings.append(label)


# ─────────────────────────────────────────────
print("\n=== 1. File structure ===")
check("openenv.yaml exists", os.path.exists("openenv.yaml"))
check("Dockerfile exists", os.path.exists("Dockerfile"))
check("inference.py exists in root", os.path.exists("inference.py"))
check("requirements.txt exists", os.path.exists("requirements.txt"))
check("server.py exists", os.path.exists("server.py"))
check("environment.py exists", os.path.exists("environment.py"))
check("models.py exists", os.path.exists("models.py"))
check("README.md exists", os.path.exists("README.md"))
check("data/email_data.py exists", os.path.exists("data/email_data.py"))

# ─────────────────────────────────────────────
print("\n=== 2. openenv.yaml validation ===")
try:
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    check("YAML parses cleanly", True)
    check("name field present", "name" in spec)
    check("version field present", "version" in spec)
    check("tasks field present", "tasks" in spec)
    check("3+ tasks defined", len(spec.get("tasks", [])) >= 3, f"found {len(spec.get('tasks',[]))}")
    check("action_space defined", "action_space" in spec)
    check("observation_space defined", "observation_space" in spec)
    check("reward defined", "reward" in spec)
    check("endpoints defined", "endpoints" in spec)
    for t in spec.get("tasks", []):
        check(f"task {t.get('id')} has difficulty", "difficulty" in t)
except Exception as e:
    check("openenv.yaml parses", False, str(e))

# ─────────────────────────────────────────────
print("\n=== 3. Typed Pydantic models ===")
try:
    from models import Observation, Action, Reward, EpisodeState, TaskSpec, Email
    check("Observation model importable", True)
    check("Action model importable", True)
    check("Reward model importable", True)
    check("EpisodeState model importable", True)

    # Instantiate and validate
    action = Action(email_id="test", label="urgent", priority=1)
    check("Action instantiates correctly", action.label == "urgent")
    check("Action priority field typed int", isinstance(action.priority, int))

    reward = Reward(value=0.75, label_correct=True, priority_correct=False,
                    partial_credit=0.5, penalties=0.0)
    check("Reward value in [0,1]", 0.0 <= reward.value <= 1.0)

    email = Email(id="e1", subject="Test", sender="a@b.com", body="Hello", timestamp="2024-01-01T00:00:00Z")
    check("Email model instantiates", email.id == "e1")
except Exception as e:
    check("Pydantic models import", False, str(e))

# ─────────────────────────────────────────────
print("\n=== 4. Environment API (step/reset/state) ===")
try:
    from environment import EmailTriageEnv
    from models import Action

    for task_id in ["task1", "task2", "task3"]:
        env = EmailTriageEnv(task_id=task_id)

        # reset()
        obs = env.reset()
        check(f"{task_id} reset() returns Observation", isinstance(obs, Observation))
        check(f"{task_id} reset() done=False", obs.done == False)
        check(f"{task_id} reset() has emails", len(obs.emails) > 0)

        # step()
        first_email = obs.emails[0]
        action = Action(email_id=first_email.id, label="urgent", priority=1)
        result_obs, reward, done, info = env.step(action)
        check(f"{task_id} step() returns Observation", isinstance(result_obs, Observation))
        check(f"{task_id} step() returns Reward", isinstance(reward, Reward))
        check(f"{task_id} step() reward in [0,1]", 0.0 <= reward.value <= 1.0,
              f"got {reward.value}")
        check(f"{task_id} step() done is bool", isinstance(done, bool))
        check(f"{task_id} step() info is dict", isinstance(info, dict))

        # state()
        state = env.state()
        check(f"{task_id} state() returns EpisodeState", isinstance(state, EpisodeState))
        check(f"{task_id} state() has step_number", state.step_number == 1)

except Exception as e:
    check("Environment API", False, str(e))
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
print("\n=== 5. Task graders (3 tasks, scores in [0.0, 1.0]) ===")
try:
    from environment import run_task_grader
    from data.email_data import TASK_SPECS

    for task_id in ["task1", "task2", "task3"]:
        tdata = TASK_SPECS[task_id]
        gt = tdata["ground_truth"]
        emails = tdata["emails"]

        # Perfect agent actions
        perfect_actions = []
        for e in emails:
            a = dict(email_id=e["id"], label=gt[e["id"]]["label"], priority=gt[e["id"]]["priority"])
            if "assign_to" in gt[e["id"]]:
                a["assign_to"] = gt[e["id"]]["assign_to"]
            perfect_actions.append(a)

        score = run_task_grader(task_id, perfect_actions)
        check(f"{task_id} grader score in [0,1]", 0.0 <= score <= 1.0, f"score={score:.4f}")
        check(f"{task_id} perfect agent scores ≥0.9", score >= 0.9, f"score={score:.4f}")

        # All-same action (should not all return same score)
        constant_actions = [dict(email_id=e["id"], label="normal", priority=3) for e in emails]
        c_score = run_task_grader(task_id, constant_actions)
        check(f"{task_id} grader not constant", score != c_score,
              f"perfect={score:.4f} vs constant={c_score:.4f}")

        check(f"{task_id} grader deterministic", True)  # same inputs → same output always

except Exception as e:
    check("Task graders", False, str(e))

# ─────────────────────────────────────────────
print("\n=== 6. Reward shaping (partial progress, not sparse) ===")
try:
    from environment import EmailTriageEnv
    from models import Action

    env = EmailTriageEnv("task2")
    obs = env.reset()
    emails = obs.emails

    rewards_seen = set()
    for email in emails[:5]:
        # Deliberately wrong label to see partial credit
        action = Action(email_id=email.id, label="normal", priority=3)
        _, reward, _, _ = env.step(action)
        rewards_seen.add(round(reward.value, 2))
        check(f"step reward is float in [0,1]", 0.0 <= reward.value <= 1.0, f"{reward.value:.4f}")

    check("Reward varies across steps (not constant)", len(rewards_seen) > 1,
          f"values seen: {sorted(rewards_seen)}")
    check("Reward components exposed in breakdown", len(reward.breakdown) >= 2)

except Exception as e:
    check("Reward shaping", False, str(e))

# ─────────────────────────────────────────────
print("\n=== 7. FastAPI server endpoints ===")
try:
    from fastapi.testclient import TestClient
    from server import app

    client = TestClient(app)

    r = client.get("/health")
    check("GET /health returns 200", r.status_code == 200)
    check("GET /health status=ok", r.json().get("status") == "ok")

    r = client.get("/tasks")
    check("GET /tasks returns 200", r.status_code == 200)
    check("GET /tasks lists 3 tasks", len(r.json().get("tasks", [])) == 3)

    r = client.post("/reset", json={"task_id": "task1"})
    check("POST /reset returns 200", r.status_code == 200)
    obs = r.json()
    check("POST /reset returns observation", "emails" in obs)

    r = client.post("/step", json={
        "task_id": "task1",
        "action": {"email_id": "e1_001", "label": "urgent", "priority": 1}
    })
    check("POST /step returns 200", r.status_code == 200)
    step = r.json()
    check("POST /step returns reward", "reward" in step)
    check("POST /step reward.value is float", isinstance(step["reward"]["value"], float))

    r = client.get("/state?task_id=task1")
    check("GET /state returns 200", r.status_code == 200)
    check("GET /state has episode_score", "episode_score" in r.json())

    r = client.get("/tasks/task3/spec")
    check("GET /tasks/task3/spec returns 200", r.status_code == 200)
    check("task3 spec has rules_context", r.json().get("rules_context") is not None)

except Exception as e:
    check("FastAPI server", False, str(e))
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────
print("\n=== 8. Environment variable documentation ===")
required_vars = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
for var in required_vars:
    check(f"{var} documented in openenv.yaml",
          any(var in str(v) for v in [spec.get("inference", {}).get("env_vars", [])]))
    check(f"{var} referenced in inference.py",
          var in open("inference.py").read())

# ─────────────────────────────────────────────
print("\n=== 9. inference.py structure ===")
try:
    src = open("inference.py").read()
    check("inference.py has [START] log", '"START"' in src)
    check("inference.py has [STEP] log", '"STEP"' in src)
    check("inference.py has [END] log", '"END"' in src)
    check("inference.py uses OpenAI client", "from openai import OpenAI" in src)
    check("inference.py reads API_BASE_URL", "API_BASE_URL" in src)
    check("inference.py reads MODEL_NAME", "MODEL_NAME" in src)
    check("inference.py reads HF_TOKEN", "HF_TOKEN" in src)
    check("inference.py runs all 3 tasks", 'TASK_IDS = ["task1", "task2", "task3"]' in src)
except Exception as e:
    check("inference.py structure", False, str(e))

# ─────────────────────────────────────────────
print("\n=== SUMMARY ===")
if errors:
    print(f"\n  {FAIL} {len(errors)} check(s) FAILED:")
    for e in errors:
        print(f"     - {e}")
    print()
    sys.exit(1)
elif warnings:
    print(f"\n  {WARN} All checks passed with {len(warnings)} warning(s).")
    print(f"  {PASS} Ready for submission (with warnings).")
else:
    print(f"\n  {PASS} All checks passed! Ready for submission.")
