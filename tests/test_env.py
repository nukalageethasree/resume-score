"""
tests/test_env.py
pytest test suite for ResumeScorerEnv.
Run: pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ResumeScorerEnv, Action, ALL_TASKS
from environment.models import Observation, Reward, EnvState
from environment.graders import grade


class TestReset:
    def test_reset_returns_observation(self):
        env = ResumeScorerEnv()
        obs = env.reset("task_1")
        assert isinstance(obs, Observation)

    def test_reset_all_tasks(self):
        env = ResumeScorerEnv()
        for tid in ALL_TASKS:
            obs = env.reset(tid)
            assert obs.task_id == tid

    def test_reset_custom_jd_and_resume(self):
        env = ResumeScorerEnv()
        obs = env.reset("task_1", job_description="Need Python dev", resume_text="Python expert")
        assert "python" in obs.job_description.lower()
        assert "python" in obs.resume_text.lower()

    def test_reset_invalid_task_raises(self):
        env = ResumeScorerEnv()
        with pytest.raises(ValueError):
            env.reset("task_99")

    def test_step_before_reset_raises(self):
        env = ResumeScorerEnv()
        with pytest.raises(RuntimeError):
            env.step(Action(action_type="submit_score", score=0.5))


class TestStep:
    def test_step_returns_tuple(self):
        env = ResumeScorerEnv()
        env.reset("task_1")
        obs, reward, done, info = env.step(Action(action_type="submit_score", score=0.65))
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_increments_step_number(self):
        env = ResumeScorerEnv()
        env.reset("task_1")
        env.step(Action(action_type="submit_score", score=0.5))
        assert env.state().step_number == 1

    def test_finalize_ends_episode(self):
        env = ResumeScorerEnv()
        env.reset("task_1")
        env.step(Action(action_type="submit_score", score=0.65))
        _, _, done, _ = env.step(Action(action_type="finalize", score=0.65))
        assert done is True

    def test_step_after_done_raises(self):
        env = ResumeScorerEnv()
        env.reset("task_1")
        env.step(Action(action_type="finalize", score=0.65))
        with pytest.raises(RuntimeError):
            env.step(Action(action_type="submit_score", score=0.5))

    def test_submit_score_records_score(self):
        env = ResumeScorerEnv()
        env.reset("task_1")
        env.step(Action(action_type="submit_score", score=0.77))
        assert env.state().agent_submitted_score == 0.77

    def test_submit_feedback_records_feedback(self):
        env = ResumeScorerEnv()
        env.reset("task_1")
        env.step(Action(action_type="submit_feedback", feedback="Add quantified results."))
        assert env.state().agent_submitted_feedback is not None


class TestRewards:
    def test_accurate_score_gives_positive_reward(self):
        env = ResumeScorerEnv()
        env.reset("task_1")  # GT = 0.65
        _, reward, _, _ = env.step(Action(action_type="submit_score", score=0.65))
        assert reward.total > 0

    def test_wildly_wrong_score_gives_less_reward(self):
        env1, env2 = ResumeScorerEnv(), ResumeScorerEnv()
        env1.reset("task_1"); env2.reset("task_1")
        _, r_good, _, _ = env1.step(Action(action_type="submit_score", score=0.65))
        _, r_bad,  _, _ = env2.step(Action(action_type="submit_score", score=0.10))
        assert r_good.total > r_bad.total

    def test_invalid_action_penalises(self):
        env = ResumeScorerEnv()
        env.reset("task_1")
        _, reward, _, _ = env.step(Action(action_type="not_a_real_action"))
        assert reward.total < 0

    def test_short_feedback_penalises(self):
        env = ResumeScorerEnv()
        env.reset("task_2")
        _, reward, _, _ = env.step(Action(action_type="submit_feedback", feedback="ok"))
        assert reward.total <= 0

    def test_no_score_before_finalize_penalises(self):
        env = ResumeScorerEnv()
        env.reset("task_1")
        _, reward, _, _ = env.step(Action(action_type="finalize"))
        assert reward.total < 0


class TestGraders:
    @pytest.mark.parametrize("tid", ["task_1", "task_2", "task_3"])
    def test_grade_range(self, tid):
        task = ALL_TASKS[tid]
        result = grade(task, task.ground_truth_score, "detailed improvement list " * 5, 3)
        assert 0.0 <= result["grade"] <= 1.0

    def test_task1_perfect_score_passes(self):
        task = ALL_TASKS["task_1"]
        result = grade(task, task.ground_truth_score, None, 2)
        assert result["grade"] >= 0.85

    def test_task2_needs_feedback(self):
        task = ALL_TASKS["task_2"]
        r_with    = grade(task, 0.72, "Mention A/B testing and PyTorch." * 4, 4)
        r_without = grade(task, 0.72, None, 4)
        assert r_with["grade"] > r_without["grade"]

    def test_task3_needs_4_items(self):
        task = ALL_TASKS["task_3"]
        short_fb  = "Missing P&L experience."
        detail_fb = (
            "1. Only 3 years management experience.\n"
            "2. Missing P&L ownership.\n"
            "3. No SOC2 experience.\n"
            "4. No board presentations.\n"
            "5. MBA preferred."
        )
        r_short  = grade(task, 0.55, short_fb, 5)
        r_detail = grade(task, 0.55, detail_fb, 5)
        assert r_detail["grade"] >= r_short["grade"]


class TestObservationFields:
    def test_skill_match_ratio_in_range(self):
        env = ResumeScorerEnv()
        obs = env.reset("task_1")
        assert 0.0 <= obs.skill_match_ratio <= 1.0

    def test_word_count_positive(self):
        env = ResumeScorerEnv()
        obs = env.reset("task_1")
        assert obs.word_count > 0

    def test_education_detected(self):
        env = ResumeScorerEnv()
        obs = env.reset("task_1")
        assert obs.education_level in ("bachelor", "master", "phd", "high_school", None)

    def test_internship_count(self):
        env = ResumeScorerEnv()
        obs = env.reset("task_1")
        assert obs.internship_count >= 1  # Task 1 resume has an intern

    def test_state_returns_env_state(self):
        env = ResumeScorerEnv()
        env.reset("task_3")
        s = env.state()
        assert isinstance(s, EnvState)
        assert s.task_id == "task_3"
