"""
FastAPI server for the Email Triage OpenEnv environment.
Implements the full OpenEnv HTTP interface:
  POST /reset
  POST /step
  GET  /state
  GET  /tasks
  GET  /health
"""
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import Action, Observation, Reward, EpisodeState, TaskSpec
from environment import EmailTriageEnv
from data.email_data import TASK_SPECS

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "An OpenEnv-compliant reinforcement learning environment simulating "
        "real-world professional email triage. Agents must classify emails "
        "by label, priority, and routing with partial reward shaping."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instances (one per task_id, lazily initialized)
_envs: Dict[str, EmailTriageEnv] = {}


def get_env(task_id: str) -> EmailTriageEnv:
    if task_id not in TASK_SPECS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid options: {list(TASK_SPECS)}",
        )
    if task_id not in _envs:
        _envs[task_id] = EmailTriageEnv(task_id=task_id)
    return _envs[task_id]


# ─────────────────────────────────────────────
# Request/Response schemas
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1"


class StepRequest(BaseModel):
    task_id: str = "task1"
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "environment": "email-triage", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata."""
    tasks = []
    for tid, tdata in TASK_SPECS.items():
        tasks.append({
            "task_id": tid,
            "name": tdata["name"],
            "description": tdata["description"],
            "difficulty": tdata["difficulty"],
            "num_emails": tdata["num_emails"],
            "max_steps": tdata["max_steps"],
            "success_threshold": tdata["success_threshold"],
            "has_routing_rules": tdata["rules_context"] is not None,
        })
    return {"tasks": tasks}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest):
    """
    Reset the environment for a given task.
    Returns the initial observation.
    """
    env = get_env(request.task_id)
    obs = env.reset()
    return obs


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Take one action in the environment.
    Returns (observation, reward, done, info).
    """
    env = get_env(request.task_id)
    if env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset first.",
        )
    try:
        obs, reward, done, info = env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=EpisodeState)
def get_state(task_id: str = Query(default="task1")):
    """
    Return the full current episode state (for debugging/inspection).
    """
    env = get_env(task_id)
    return env.state()


@app.get("/tasks/{task_id}/spec")
def get_task_spec(task_id: str):
    """Get the spec for a specific task."""
    if task_id not in TASK_SPECS:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    tdata = TASK_SPECS[task_id]
    return {
        "task_id": task_id,
        "name": tdata["name"],
        "description": tdata["description"],
        "difficulty": tdata["difficulty"],
        "num_emails": tdata["num_emails"],
        "max_steps": tdata["max_steps"],
        "success_threshold": tdata["success_threshold"],
        "valid_labels": ["urgent", "normal", "low", "spam", "archive"],
        "valid_teams": ["sales", "security", "engineering", "hr", "legal", "general"],
        "rules_context": tdata.get("rules_context"),
    }


@app.get("/")
def root():
    """Root endpoint with environment info."""
    return {
        "name": "Email Triage OpenEnv",
        "description": "Real-world email triage environment for AI agent evaluation",
        "tasks": list(TASK_SPECS.keys()),
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state?task_id=task1",
            "tasks": "GET /tasks",
            "health": "GET /health",
        },
        "action_space": {
            "email_id": "string — ID of the email being triaged",
            "label": "string — one of: urgent, normal, low, spam, archive",
            "priority": "int — 1 (highest) to 5 (lowest)",
            "assign_to": "string (task3 only) — one of: sales, security, engineering, hr, legal, general",
            "reply_draft": "string (optional) — draft reply text",
            "reasoning": "string (optional) — agent reasoning (not graded)",
        },
        "observation_space": {
            "task_id": "string",
            "step_number": "int",
            "emails": "list of Email objects",
            "processed_count": "int",
            "total_emails": "int",
            "current_email_index": "int",
            "inbox_labels": "dict — email_id to label assigned so far",
            "rules_context": "string or null — routing rules for task3",
            "done": "bool",
        },
    }
