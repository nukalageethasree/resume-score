"""
ResumeScorerEnv – OpenEnv-compliant environment for resume scoring.

Interface:
    reset(task_id, job_description=None, resume_text=None)
        → Observation

    step(action: Action)
        → (Observation, Reward, done: bool, info: dict)

    state()
        → EnvState
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, Optional, Tuple

from .models import (
    Action, EnvState, Observation, Reward, SectionPresence, SkillMatch,
)
from .tasks import ALL_TASKS, TaskSpec
from .graders import grade


# ---------------------------------------------------------------------------
# NLP helpers (pure Python, no heavy deps)
# ---------------------------------------------------------------------------

_SECTION_KEYWORDS = {
    "contact_info":    ["email", "phone", "linkedin", "github", "@"],
    "summary":         ["summary", "objective", "profile", "about"],
    "education":       ["education", "degree", "university", "college", "b.sc", "m.sc", "phd", "bachelor", "master"],
    "experience":      ["experience", "employment", "work history", "intern"],
    "skills":          ["skills", "technologies", "tools", "languages"],
    "projects":        ["project", "portfolio", "open-source", "github.com"],
    "certifications":  ["certification", "certificate", "certified", "licence"],
    "achievements":    ["achievement", "award", "honor", "recognition", "ranking"],
}

_EDU_LEVELS = {
    "phd":       ["phd", "ph.d", "doctorate", "doctor of"],
    "master":    ["m.sc", "msc", "m.s.", "master", "mba", "m.eng"],
    "bachelor":  ["b.sc", "bsc", "b.s.", "bachelor", "b.eng", "b.tech"],
    "high_school": ["high school", "secondary"],
}

_YEAR_PATTERN = re.compile(
    r"\b(20\d{2})\b.*?(?:–|-|to)\s*(?:(20\d{2})|present|current)",
    re.IGNORECASE,
)

_INTERNSHIP_PATTERN = re.compile(r"\bintern\b", re.IGNORECASE)


def _detect_sections(text: str) -> SectionPresence:
    lower = text.lower()
    sp = SectionPresence()
    for field, kws in _SECTION_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            setattr(sp, field, True)
    return sp


def _detect_education_level(text: str) -> Optional[str]:
    lower = text.lower()
    for level, kws in _EDU_LEVELS.items():
        if any(kw in lower for kw in kws):
            return level
    return None


def _detect_years_experience(text: str) -> float:
    total = 0.0
    for m in _YEAR_PATTERN.finditer(text):
        start = int(m.group(1))
        end_raw = m.group(2)
        end = int(end_raw) if end_raw else 2024
        total += max(0, end - start)
    return min(total, 30.0)


def _count_internships(text: str) -> int:
    return len(_INTERNSHIP_PATTERN.findall(text))


def _extract_skills_from_jd(jd: str) -> list[str]:
    """Naive keyword extraction from the job description."""
    tech_pattern = re.compile(
        r"\b(python|java(?:script)?|typescript|react|vue|angular|node\.?js|"
        r"django|flask|fastapi|spring|ruby|rails|php|go|rust|scala|kotlin|swift|"
        r"c\+\+|c#|\.net|r\b|matlab|tensorflow|pytorch|keras|scikit[\-_]learn|"
        r"pandas|numpy|spark|hadoop|kafka|airflow|mlflow|"
        r"sql|postgresql|mysql|mongodb|redis|elasticsearch|"
        r"docker|kubernetes|aws|gcp|azure|terraform|ansible|"
        r"git|ci/?cd|rest\s?api|graphql|"
        r"a/b\s?test|machine\s?learning|deep\s?learning|nlp|"
        r"soc\s?2|agile|scrum)\b",
        re.IGNORECASE,
    )
    return list({m.lower() for m in tech_pattern.findall(jd)})


def _match_skills(resume: str, skills: list[str]) -> list[SkillMatch]:
    lower = resume.lower()
    matches = []
    for skill in skills:
        found = skill.lower() in lower
        conf = 1.0 if found else 0.0
        matches.append(SkillMatch(skill=skill, found_in_resume=found, confidence=conf))
    return matches


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ResumeScorerEnv:
    """
    OpenEnv-compliant Resume Scorer environment.

    Supports three tasks:
        task_1  – Easy    (score only)
        task_2  – Medium  (score + 2 feedback points)
        task_3  – Hard    (accurate score + 4+ feedback points)
    """

    VERSION = "1.0.0"

    def __init__(self) -> None:
        self._task: Optional[TaskSpec] = None
        self._state: Optional[EnvState] = None

    # ------------------------------------------------------------------
    # Public OpenEnv Interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = "task_1",
        job_description: Optional[str] = None,
        resume_text: Optional[str] = None,
    ) -> Observation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of 'task_1', 'task_2', 'task_3'.
            job_description: Override the task's default JD (optional).
            resume_text: Override the task's default resume text (optional).

        Returns:
            Initial Observation.
        """
        if task_id not in ALL_TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(ALL_TASKS)}")

        self._task = ALL_TASKS[task_id]
        jd = job_description or self._task.job_description
        resume = resume_text or self._task.resume_text

        self._state = EnvState(
            task_id=task_id,
            step_number=0,
            max_steps=self._task.max_steps,
            job_description=jd,
            resume_text=resume,
            ground_truth_score=self._task.ground_truth_score,
            ground_truth_feedback=self._task.ground_truth_feedback,
            agent_submitted_score=None,
            agent_submitted_feedback=None,
            done=False,
            cumulative_reward=0.0,
        )

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply an action and advance the environment.

        Returns:
            (observation, reward, done, info)
        """
        if self._state is None or self._task is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        reward = self._process_action(action)
        self._state.step_number += 1
        self._state.cumulative_reward += reward.total

        # Episode ends on 'finalize', reaching max_steps, or already scored+done
        done = (
            action.action_type == "finalize"
            or self._state.step_number >= self._state.max_steps
        )
        self._state.done = done

        obs = self._build_observation()
        info = {
            "step": self._state.step_number,
            "cumulative_reward": self._state.cumulative_reward,
            "task_id": self._state.task_id,
        }
        if done:
            result = grade(
                self._task,
                self._state.agent_submitted_score,
                self._state.agent_submitted_feedback,
                self._state.step_number,
            )
            info["episode_grade"] = result
            self._state.episode_info = result

        return obs, reward, done, info

    def state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_action(self, action: Action) -> Reward:
        s = self._state
        t = self._task

        if action.action_type == "submit_score":
            if action.score is None:
                return Reward(
                    total=-0.05,
                    invalid_action_penalty=-0.05,
                    explanation="submit_score requires a numeric score.",
                )
            s.agent_submitted_score = action.score
            delta = abs(action.score - t.ground_truth_score)
            acc_reward = max(0.0, 1.0 - delta / 0.40) * 0.5
            return Reward(
                total=round(acc_reward, 4),
                score_accuracy_reward=round(acc_reward, 4),
                explanation=f"Score submitted: {action.score:.2f} (Δ={delta:.2f} from GT).",
            )

        elif action.action_type == "submit_feedback":
            if not action.feedback or len(action.feedback.strip()) < 20:
                return Reward(
                    total=-0.05,
                    invalid_action_penalty=-0.05,
                    explanation="Feedback too short or empty.",
                )
            s.agent_submitted_feedback = action.feedback
            words = len(action.feedback.split())
            fq_reward = min(words / 100.0, 1.0) * 0.4
            return Reward(
                total=round(fq_reward, 4),
                feedback_quality_reward=round(fq_reward, 4),
                explanation=f"Feedback received ({words} words).",
            )

        elif action.action_type == "request_section":
            # Mild step cost; environment will reveal section info in next obs
            return Reward(
                total=-0.01,
                efficiency_penalty=-0.01,
                explanation=f"Requested section: {action.requested_section}.",
            )

        elif action.action_type == "finalize":
            # Final reconciliation reward
            if s.agent_submitted_score is None:
                return Reward(
                    total=-0.10,
                    invalid_action_penalty=-0.10,
                    explanation="Finalized without submitting a score.",
                )
            delta = abs(s.agent_submitted_score - t.ground_truth_score)
            final_bonus = max(0.0, 0.50 - delta * 2)
            return Reward(
                total=round(final_bonus, 4),
                score_accuracy_reward=round(final_bonus, 4),
                explanation=f"Episode finalized. Final accuracy bonus={final_bonus:.3f}.",
            )

        else:
            return Reward(
                total=-0.10,
                invalid_action_penalty=-0.10,
                explanation=f"Unknown action_type: '{action.action_type}'.",
            )

    def _build_observation(self) -> Observation:
        s = self._state
        resume = s.resume_text
        jd = s.job_description

        sections = _detect_sections(resume)
        edu = _detect_education_level(resume)
        yrs = _detect_years_experience(resume)
        interns = _count_internships(resume)
        req_skills = _extract_skills_from_jd(jd)
        skill_matches = _match_skills(resume, req_skills)
        matched = sum(1 for sm in skill_matches if sm.found_in_resume)
        ratio = matched / max(len(req_skills), 1)

        return Observation(
            job_description=jd,
            resume_text=resume,
            sections_present=sections,
            word_count=len(resume.split()),
            years_experience_detected=yrs,
            internship_count=interns,
            education_level=edu,
            required_skills_from_jd=req_skills,
            skill_matches=skill_matches,
            skill_match_ratio=round(ratio, 4),
            agent_feedback=s.agent_submitted_feedback,
            current_score=s.agent_submitted_score or 0.0,
            task_id=s.task_id,
            step_number=s.step_number,
            done=s.done,
        )
