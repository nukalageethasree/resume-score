"""
OpenEnv typed Pydantic models for Resume Scorer environment.
Observation, Action, Reward, and State models.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SkillMatch(BaseModel):
    skill: str
    found_in_resume: bool
    confidence: float = Field(ge=0.0, le=1.0)


class SectionPresence(BaseModel):
    contact_info: bool = False
    summary: bool = False
    education: bool = False
    experience: bool = False
    skills: bool = False
    projects: bool = False
    certifications: bool = False
    achievements: bool = False


class Observation(BaseModel):
    """Typed observation returned by the environment after each step."""

    # Core content
    job_description: str = Field(description="The job description text provided at reset")
    resume_text: str = Field(description="Extracted plain-text from the uploaded resume")

    # Structural analysis
    sections_present: SectionPresence = Field(default_factory=SectionPresence)
    word_count: int = Field(ge=0, default=0)
    years_experience_detected: float = Field(ge=0.0, default=0.0)
    internship_count: int = Field(ge=0, default=0)
    education_level: Optional[str] = Field(
        default=None,
        description="Highest detected: 'high_school' | 'bachelor' | 'master' | 'phd'"
    )

    # Skill matching
    required_skills_from_jd: List[str] = Field(default_factory=list)
    skill_matches: List[SkillMatch] = Field(default_factory=list)
    skill_match_ratio: float = Field(ge=0.0, le=1.0, default=0.0)

    # Previous agent actions / feedback
    agent_feedback: Optional[str] = Field(
        default=None,
        description="Feedback text submitted by the agent in the previous step"
    )
    current_score: float = Field(ge=0.0, le=1.0, default=0.0)

    # Task metadata
    task_id: str = Field(default="task_1")
    step_number: int = Field(ge=0, default=0)
    done: bool = False


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Typed action the agent sends to the environment."""

    action_type: str = Field(
        description=(
            "One of: 'submit_score' | 'submit_feedback' | 'request_section' | 'finalize'"
        )
    )
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Numeric score in [0,1] (used with 'submit_score' or 'finalize')"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Textual feedback or improvement suggestions (used with 'submit_feedback')"
    )
    requested_section: Optional[str] = Field(
        default=None,
        description="Section name to inspect more closely (used with 'request_section')"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Detailed reward breakdown for one environment step."""

    total: float = Field(description="Scalar reward for this step")

    # Components (sum to total)
    score_accuracy_reward: float = Field(default=0.0)
    feedback_quality_reward: float = Field(default=0.0)
    efficiency_penalty: float = Field(default=0.0)   # negative
    invalid_action_penalty: float = Field(default=0.0)  # negative

    explanation: str = Field(default="")


# ---------------------------------------------------------------------------
# Environment State
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """Full internal state snapshot (returned by state())."""

    task_id: str
    step_number: int
    max_steps: int
    job_description: str
    resume_text: str
    ground_truth_score: float
    ground_truth_feedback: List[str]
    agent_submitted_score: Optional[float]
    agent_submitted_feedback: Optional[str]
    done: bool
    cumulative_reward: float
    episode_info: Dict[str, Any] = Field(default_factory=dict)
