"""
validate.py – openenv validate implementation for ResumeScorerEnv.

Checks:
  1. openenv.yaml is parseable and contains required keys.
  2. All three tasks can be reset and stepped.
  3. Observation / Reward Pydantic models validate correctly.
  4. Grader returns scores in [0, 1].
  5. Episode terminates within max_steps.

Usage:
    python validate.py
"""

import sys
import os
import json
import traceback
from pathlib import Path

import yaml  # PyYAML

sys.path.insert(0, str(Path(__file__).parent))

from environment import ResumeScorerEnv, Action, ALL_TASKS
from environment.models import Observation, Reward, EnvState
from environment.graders import grade


PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name: str, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"  {PASS}  {name}")
    except Exception as exc:
        results.append((name, False, str(exc)))
        print(f"  {FAIL}  {name}")
        print(f"         {exc}")


# ---------------------------------------------------------------------------
# 1. YAML
# ---------------------------------------------------------------------------

def test_yaml():
    with open(os.path.join(os.path.dirname(__file__), "openenv.yaml")) as f:
        cfg = yaml.safe_load(f)
    required = ["name", "version", "description", "observation_space",
                "action_space", "reward_space", "tasks"]
    missing = [k for k in required if k not in cfg]
    assert not missing, f"Missing YAML keys: {missing}"
    assert len(cfg["tasks"]) >= 3, "Need at least 3 tasks"


# ---------------------------------------------------------------------------
# 2. reset() returns valid Observation
# ---------------------------------------------------------------------------

def test_reset():
    env = ResumeScorerEnv()
    for tid in ["task_1", "task_2", "task_3"]:
        obs = env.reset(task_id=tid)
        assert isinstance(obs, Observation), f"reset() must return Observation, got {type(obs)}"
        assert obs.task_id == tid
        assert len(obs.job_description) > 10
        assert len(obs.resume_text) > 10


# ---------------------------------------------------------------------------
# 3. step() returns (Observation, Reward, bool, dict)
# ---------------------------------------------------------------------------

def test_step_types():
    env = ResumeScorerEnv()
    obs = env.reset("task_1")
    action = Action(action_type="submit_score", score=0.65)
    obs2, reward, done, info = env.step(action)
    assert isinstance(obs2, Observation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# 4. state() returns EnvState
# ---------------------------------------------------------------------------

def test_state():
    env = ResumeScorerEnv()
    env.reset("task_2")
    s = env.state()
    assert isinstance(s, EnvState)
    assert s.task_id == "task_2"


# ---------------------------------------------------------------------------
# 5. Invalid action gives negative reward
# ---------------------------------------------------------------------------

def test_invalid_action_penalty():
    env = ResumeScorerEnv()
    env.reset("task_1")
    _, reward, _, _ = env.step(Action(action_type="unknown_action"))
    assert reward.total < 0, "Invalid action should yield negative reward"


# ---------------------------------------------------------------------------
# 6. Episode terminates ≤ max_steps
# ---------------------------------------------------------------------------

def test_termination():
    for tid, task in ALL_TASKS.items():
        env = ResumeScorerEnv()
        env.reset(tid)
        done = False
        steps = 0
        while not done:
            if steps >= 2:
                action = Action(action_type="finalize", score=0.5)
            else:
                action = Action(action_type="submit_score", score=0.5)
            _, _, done, _ = env.step(action)
            steps += 1
            assert steps <= task.max_steps + 1, f"{tid}: exceeded max_steps"


# ---------------------------------------------------------------------------
# 7. Grader outputs in [0, 1]
# ---------------------------------------------------------------------------

def test_grader_range():
    for tid, task in ALL_TASKS.items():
        for score, fb in [(0.0, None), (0.5, "Some feedback"), (1.0, "Very detailed " * 15)]:
            result = grade(task, score, fb, 3)
            assert 0.0 <= result["grade"] <= 1.0, (
                f"{tid}: grade {result['grade']} out of range"
            )


# ---------------------------------------------------------------------------
# 8. Partial-progress reward (not just binary)
# ---------------------------------------------------------------------------

def test_partial_reward():
    env = ResumeScorerEnv()
    env.reset("task_1")
    _, r1, _, _ = env.step(Action(action_type="submit_score", score=0.80))
    env.reset("task_1")
    _, r2, _, _ = env.step(Action(action_type="submit_score", score=0.50))
    # 0.80 is closer to GT 0.65 than 0.50 → r1 >= r2
    assert r1.total >= r2.total, "Partial reward not working correctly"


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 55)
    print("  openenv validate  –  ResumeScorerEnv")
    print("=" * 55)

    check("openenv.yaml integrity",       test_yaml)
    check("reset() returns Observation",  test_reset)
    check("step() returns correct types", test_step_types)
    check("state() returns EnvState",     test_state)
    check("invalid action penalty",       test_invalid_action_penalty)
    check("episode terminates correctly", test_termination)
    check("grader range [0, 1]",          test_grader_range)
    check("partial-progress reward",      test_partial_reward)

    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print("\n" + "=" * 55)
    print(f"  Results: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉  All checks passed – environment is valid!")
    else:
        print("  ⚠️   Some checks failed – see above.")
    print("=" * 55 + "\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
