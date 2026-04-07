# email-triage-env
# Email Triage OpenEnv

> A real-world **email triage environment** for AI agent training and evaluation, fully compliant with the [OpenEnv](https://huggingface.co/openenv) specification.

---

## Overview

**Email Triage OpenEnv** simulates the professional task of managing a busy inbox. Agents must classify incoming emails with the correct label, assign a priority score, and (in the hard task) route each email to the correct team — all while maximizing reward across the full episode.

This is a task humans perform every day. It requires:
- Reading comprehension and context understanding
- Distinguishing urgency from noise
- Applying business logic and routing rules
- Making consistent, accurate decisions at scale

Unlike toy environments, email triage has real-world consequences: misclassifying a production outage as "low" or routing a $2.4M contract to the wrong team has measurable business impact. This makes it an excellent benchmark for evaluating agent reliability, rule-following, and nuanced language understanding.

---

## Tasks

| Task | Difficulty | Emails | Max Steps | Description |
|------|-----------|--------|-----------|-------------|
| `task1` | Easy | 5 | 10 | Single professional inbox — clear signals for urgent, normal, low, spam |
| `task2` | Medium | 10 | 15 | Busy executive inbox — separate critical items from noise |
| `task3` | Hard | 12 | 20 | Multi-account + routing rules — follow team assignment rules under ambiguity |

### Task 1 — Single Inbox Triage (Easy)
Five emails with unambiguous signals: a production alert, a social invite, an invoice, a spam prize email, and a manager's draft review. Tests whether the agent can distinguish urgency levels from clear textual cues.

### Task 2 — Batch Inbox Under Time Pressure (Medium)
Ten emails from a busy executive inbox. The agent must identify three genuinely urgent items (security breach, CEO presentation in 2 hours, production bug affecting 15% of mobile orders) among newsletters, shipping notifications, survey requests, and phishing attempts. Tests prioritization under noise.

### Task 3 — Multi-Account Triage with Routing Rules (Hard)
Twelve enterprise-level emails plus explicit routing rules. The agent must assign each email to the correct team (`sales`, `security`, `engineering`, `hr`, `legal`, `general`) in addition to label and priority. Emails include a $2.4M enterprise deal, ransomware on engineering workstations, an NDA expiring in 7 days, a payment system outage generating $18k/minute in losses, and two phishing attempts. Tests rule-following under high stakes and ambiguity.

---

## Action Space

Each action triages exactly one email:

```json
{
  "email_id": "e1_001",
  "label": "urgent",
  "priority": 1,
  "assign_to": "security",
  "reply_draft": "We are investigating immediately...",
  "reasoning": "Production system down with customer impact"
}
```

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `email_id` | string | Yes | ID from the current observation |
| `label` | string | Yes | `urgent`, `normal`, `low`, `spam`, `archive` |
| `priority` | int | Yes | 1 (highest) → 5 (lowest) |
| `assign_to` | string | task3 only | `sales`, `security`, `engineering`, `hr`, `legal`, `general` |
| `reply_draft` | string | No | Optional draft reply (not graded) |
| `reasoning` | string | No | Agent reasoning (logged, not graded) |

**Label semantics:**
- `urgent` — needs action within the hour (outages, security, C-level, legal deadlines)
- `normal` — standard business task, handle today or this week
- `low` — social, low-stakes, can wait days
- `spam` — unsolicited commercial email, phishing, scams
- `archive` — newsletters, shipping notifications, FYIs requiring no action

---

## Observation Space

Returned by `reset()` and `step()`:

```json
{
  "task_id": "task1",
  "step_number": 2,
  "emails": [...],
  "processed_count": 1,
  "total_emails": 5,
  "current_email_index": 1,
  "inbox_labels": {"e1_001": "urgent"},
  "rules_context": null,
  "done": false,
  "info": {
    "cumulative_reward": 1.0,
    "steps_remaining": 8,
    "task_description": "..."
  }
}
```

Each `Email` object contains: `id`, `subject`, `sender`, `body`, `timestamp`, `has_attachment`, `thread_length`.

For `task3`, `rules_context` contains the routing rule set as a plain-text string.

---

## Reward Function

Reward is computed **at every step** (dense shaping — not sparse end-of-episode):

| Component | Weight | Condition |
|-----------|--------|-----------|
| Label correct | +0.50 | Exact match |
| Priority exact | +0.30 | Exact match |
| Priority close | +0.15 | Within ±1 |
| Team routing | +0.20 | Exact match (task3 only) |
| Critical miss penalty | −0.30 | Marking `urgent` as `spam` |
| False alarm penalty | −0.20 | Marking `spam` as `urgent` |
| Efficiency penalty | −0.01/step | Steps after the halfway mark |

**Max reward per step:** 0.80 (task1/2), 1.00 (task3 with routing)

**Episode score:** `cumulative_reward / total_emails`, normalized to [0, 1].

The reward function penalizes the two most consequential mistakes (missing a real emergency, or wasting time on fake ones) while giving partial credit for getting the label right but the priority slightly off.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/tasks/{task_id}/spec` | Task spec with rules |
| `POST` | `/reset` | Reset episode, returns initial observation |
| `POST` | `/step` | Take one action, returns obs/reward/done/info |
| `GET` | `/state?task_id=task1` | Full episode state |
| `GET` | `/` | Environment info and API summary |

### Example: Full Episode

```python
import requests

BASE = "http://localhost:7860"

# Reset
obs = requests.post(f"{BASE}/reset", json={"task_id": "task1"}).json()

# Step through all emails
while not obs["done"]:
    email = obs["emails"][obs["current_email_index"]]
    action = {
        "task_id": "task1",
        "action": {
            "email_id": email["id"],
            "label": "urgent",   # agent decides
            "priority": 1,
        }
    }
    result = requests.post(f"{BASE}/step", json=action).json()
    obs = result["observation"]
    print(f"Reward: {result['reward']['value']}")
```

---

## Setup & Usage

### Local (Python)

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
cd email-triage-env

pip install -r requirements.txt

# Start the server
uvicorn server:app --host 0.0.0.0 --port 7860

# In another terminal, run the inference baseline
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

### Docker

```bash
docker build -t email-triage-env .

docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e OPENAI_API_KEY="sk-..." \
  email-triage-env

# Run inference against the container
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

### Validation

```bash
python validate.py
```

---

## Baseline Scores

Baseline scores using `gpt-4o-mini` (temperature=0.1):

| Task | Difficulty | Baseline Score | Random Baseline |
|------|-----------|----------------|-----------------|
| task1 | Easy | ~0.87 | ~0.04 |
| task2 | Medium | ~0.79 | ~0.12 |
| task3 | Hard | ~0.68 | ~0.16 |
| **Overall** | | **~0.78** | **~0.11** |

*Perfect agent achieves: task1=1.00, task2=0.99, task3=1.00*

---

## Log Format

The inference script emits structured JSON logs to stdout:

```json
{"type": "START", "model": "gpt-4o-mini", "tasks": ["task1","task2","task3"], "timestamp": "..."}
{"type": "STEP", "task_id": "task1", "step": 1, "email_id": "e1_001", "label": "urgent", "priority": 1, "assign_to": null, "reward": 1.0, "label_correct": true, "priority_correct": true, "cumulative_reward": 1.0, "done": false}
{"type": "END", "model": "gpt-4o-mini", "task_scores": {"task1": 0.87, "task2": 0.79, "task3": 0.68}, "overall_score": 0.78, "timestamp": "..."}
```

---

## Project Structure

```
email-triage-env/
├── server.py          # FastAPI server (OpenEnv HTTP interface)
├── environment.py     # EmailTriageEnv core logic
├── models.py          # Pydantic typed models (Observation, Action, Reward)
├── inference.py       # Baseline inference script (OpenAI client)
├── validate.py        # Pre-submission validation script
├── openenv.yaml       # OpenEnv metadata
├── requirements.txt
├── Dockerfile
├── README.md
└── data/
    └── email_data.py  # Email datasets + ground truth for all 3 tasks
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | LLM API base URL | Yes (inference) |
| `MODEL_NAME` | Model identifier | Yes (inference) |
| `HF_TOKEN` | Hugging Face / API key | Yes (inference) |
| `OPENAI_API_KEY` | OpenAI-compatible key (falls back to HF_TOKEN) | Yes (inference) |
| `ENV_BASE_URL` | Environment server URL | inference only |

---

## Design Notes

**Why email triage?** Email triage is a universal professional task with billions of daily instances. It requires nuanced language understanding, contextual reasoning, and rule-following — all key agent capabilities. Unlike games, performance on this benchmark directly predicts real-world utility.

**Why dense rewards?** Sparse rewards (score only at episode end) make learning slow and unstable. Every step in this environment returns a shaped reward that distinguishes label accuracy, priority accuracy, and routing accuracy independently — giving agents rich learning signal.

**Why 3 tasks at different difficulties?** A single task can't distinguish agents that got lucky from agents that generalize. Easy→Medium→Hard progression tests whether agents scale from basic classification to complex rule-following under adversarial conditions.
