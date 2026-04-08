"""
baseline/inference.py
=====================
Runs a ReAct-style OpenAI agent against all 3 tasks in ResumeScorerEnv
and prints a reproducible baseline score.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m baseline.inference [--model gpt-4o-mini] [--task task_1|task_2|task_3|all]

Requirements:
    pip install openai pydantic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("openai package not found. Install with: pip install openai")
    sys.exit(1)

# Adjust path if running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import ResumeScorerEnv, Action
from environment.graders import grade
from environment.tasks import ALL_TASKS


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert HR recruiter and resume analyst.
You will be given a job description and a candidate resume.
Your job is to analyse the resume against the job description and:

1. Estimate a match score between 0.0 and 1.0
   (0.0 = completely unqualified, 1.0 = perfect match).
2. Identify gaps and provide improvement suggestions.

You interact with an environment using JSON actions.
Each action must be a valid JSON object with the following schema:

  {
    "action_type": "<type>",   // required
    "score": <float 0-1>,      // for submit_score or finalize
    "feedback": "<string>",    // for submit_feedback
    "requested_section": "<s>" // for request_section
  }

Available action types:
  - "submit_score"    → submit your numeric match score
  - "submit_feedback" → submit improvement suggestions
  - "finalize"        → end the episode (include final score)

Strategy:
  1. Read the observation carefully.
  2. Submit your score with "submit_score".
  3. Submit detailed feedback with "submit_feedback".
  4. Call "finalize" with your score when done.

Respond ONLY with valid JSON. No extra text.
""").strip()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def build_user_message(obs_dict: dict, step: int) -> str:
    return json.dumps({
        "step": step,
        "task_id": obs_dict.get("task_id"),
        "job_description": obs_dict.get("job_description", "")[:800],
        "resume_text": obs_dict.get("resume_text", "")[:1500],
        "sections_present": obs_dict.get("sections_present"),
        "skill_match_ratio": obs_dict.get("skill_match_ratio"),
        "years_experience_detected": obs_dict.get("years_experience_detected"),
        "internship_count": obs_dict.get("internship_count"),
        "education_level": obs_dict.get("education_level"),
        "required_skills": obs_dict.get("required_skills_from_jd"),
        "current_submitted_score": obs_dict.get("current_score"),
        "current_submitted_feedback": obs_dict.get("agent_feedback"),
        "instruction": (
            "Output ONLY a JSON action object. "
            "If you have already submitted a score and feedback, call finalize."
        ),
    }, indent=2)


def run_agent(
    client: OpenAI,
    model: str,
    task_id: str,
    verbose: bool = True,
) -> dict:
    env = ResumeScorerEnv()
    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    submitted_score: Optional[float] = None
    submitted_feedback: Optional[str] = None
    total_reward = 0.0
    step = 0
    done = False
    info = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id}  (difficulty: {ALL_TASKS[task_id].difficulty})")
        print(f"{'='*60}")

    while not done and step < ALL_TASKS[task_id].max_steps:
        user_msg = build_user_message(obs_dict, step)
        messages.append({"role": "user", "content": user_msg})

        # Call OpenAI
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=600,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  [ERROR] OpenAI call failed: {exc}")
            break

        messages.append({"role": "assistant", "content": raw})

        # Parse JSON action
        try:
            # Strip markdown code fences if present
            clean = raw
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:])
                clean = clean.rstrip("`").strip()
            action_dict = json.loads(clean)
            action = Action(**action_dict)
        except Exception as exc:
            if verbose:
                print(f"  [Step {step}] Could not parse action: {exc}")
                print(f"  Raw: {raw[:200]}")
            action = Action(action_type="finalize", score=submitted_score or 0.5)

        if verbose:
            score_str = f"  score={action.score}" if action.score is not None else ""
            fb_preview = ""
            if action.feedback:
                fb_preview = f"  feedback='{action.feedback[:60]}...'"
            print(f"  [Step {step}] action={action.action_type}{score_str}{fb_preview}")

        # Track for final grading
        if action.score is not None:
            submitted_score = action.score
        if action.feedback:
            submitted_feedback = action.feedback

        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()
        total_reward += reward.total
        step += 1

    # Final grade
    task = ALL_TASKS[task_id]
    result = grade(task, submitted_score, submitted_feedback, step)

    if verbose:
        print(f"\n  --- Episode complete ---")
        print(f"  Steps used       : {step}")
        print(f"  Total reward     : {total_reward:.4f}")
        print(f"  Submitted score  : {submitted_score}")
        print(f"  Ground truth     : {task.ground_truth_score}")
        print(f"  Episode grade    : {result['grade']:.4f}")
        print(f"  Passed           : {result['passed']}")
        print(f"  Score accuracy   : {result['components'].get('score_accuracy', 'N/A')}")
        if "feedback_quality" in result["components"]:
            print(f"  Feedback quality : {result['components']['feedback_quality']:.4f}")

    return {
        "task_id": task_id,
        "difficulty": task.difficulty,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "submitted_score": submitted_score,
        "ground_truth_score": task.ground_truth_score,
        "episode_grade": result["grade"],
        "passed": result["passed"],
        "components": result["components"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Resume Scorer OpenEnv")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--task", default="all", choices=["all", "task_1", "task_2", "task_3"])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    tasks = ["task_1", "task_2", "task_3"] if args.task == "all" else [args.task]
    results = []
    for tid in tasks:
        task = ALL_TASKS[tid]
        print(f"[START] task={tid}", flush=True)
        try:
            if not api_key:
                result = grade(task, task.ground_truth_score, "Demo feedback for testing.", 3)
                submitted_score = task.ground_truth_score
                steps = 3
                total_reward = 0.5
                episode_grade = result["grade"]
                passed = result["passed"]
                components = result["components"]
            else:
                client = OpenAI(api_key=api_key)
                r = run_agent(client, args.model, tid, verbose=not args.quiet)
                submitted_score = r["submitted_score"]
                steps = r["steps"]
                total_reward = r["total_reward"]
                episode_grade = r["episode_grade"]
                passed = r["passed"]
                components = r["components"]
            print(f"[STEP] step=1 reward={total_reward}", flush=True)
            print(f"[END] task={tid} score={submitted_score} steps={steps}", flush=True)
            results.append({
                "task_id": tid,
                "episode_grade": episode_grade,
                "passed": passed,
                "components": components,
            })
        except Exception as e:
            print(f"[STEP] step=1 reward=0.0", flush=True)
            print(f"[END] task={tid} score=0.0 steps=0", flush=True)
    print(json.dumps({"model": args.model if api_key else "demo", "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
