"""
Programmatic graders for the Resume Scorer environment.

Each grader receives the task spec plus whatever the agent submitted
and returns a scalar grade in [0.0, 1.0] plus an explanation dict.
"""

from __future__ import annotations
import re
from typing import Dict, Optional, Tuple

from .tasks import TaskSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_accuracy(
    submitted: Optional[float],
    ground_truth: float,
    tolerance: float,
) -> Tuple[float, str]:
    """
    Returns (score_component 0-1, explanation).
    Full credit within tolerance; linearly decays to 0 at ±0.40.
    """
    if submitted is None:
        return 0.0, "No score submitted."
    delta = abs(submitted - ground_truth)
    if delta <= tolerance:
        return 1.0, f"Score {submitted:.2f} is within ±{tolerance} of ground truth {ground_truth:.2f}."
    elif delta >= 0.40:
        return 0.0, f"Score {submitted:.2f} is far from ground truth {ground_truth:.2f} (Δ={delta:.2f})."
    else:
        grade = 1.0 - (delta - tolerance) / (0.40 - tolerance)
        return round(grade, 4), (
            f"Score {submitted:.2f} partially matches ground truth {ground_truth:.2f} "
            f"(Δ={delta:.2f}, partial credit={grade:.2f})."
        )


def _feedback_quality(
    submitted_feedback: Optional[str],
    expected_topics: list[str],
    min_items: int = 1,
) -> Tuple[float, str]:
    """
    Score how well the feedback covers expected topics.
    Returns (score 0-1, explanation).
    """
    if not submitted_feedback or len(submitted_feedback.strip()) < 20:
        return 0.0, "No meaningful feedback provided."

    fb_lower = submitted_feedback.lower()
    covered = [t for t in expected_topics if t.lower() in fb_lower]
    coverage_ratio = len(covered) / max(len(expected_topics), 1)

    # Reward for sufficient depth (word count)
    words = len(submitted_feedback.split())
    depth_bonus = min(words / 80.0, 1.0) * 0.2   # up to 0.2 bonus

    raw = coverage_ratio * 0.8 + depth_bonus
    grade = min(round(raw, 4), 1.0)

    explanation = (
        f"Feedback covered {len(covered)}/{len(expected_topics)} expected topics "
        f"({', '.join(covered) if covered else 'none'}). "
        f"Word count: {words}. Quality grade: {grade:.2f}."
    )
    return grade, explanation


# ---------------------------------------------------------------------------
# Per-Task Graders
# ---------------------------------------------------------------------------

def grade_task_1(
    task: TaskSpec,
    submitted_score: Optional[float],
    submitted_feedback: Optional[str],
    steps_used: int,
) -> Dict:
    """
    Task 1 (Easy) grader.
    Weights: 90 % score accuracy + 10 % efficiency bonus.
    """
    accuracy, acc_exp = _score_accuracy(
        submitted_score, task.ground_truth_score, task.score_tolerance
    )

    # Efficiency bonus: full bonus if completed in ≤3 steps, decay after that
    efficiency = max(0.0, 1.0 - max(0, steps_used - 3) * 0.2)

    total = round(accuracy * 0.90 + efficiency * 0.10, 4)

    return {
        "grade": total,
        "passed": total >= 0.70,
        "components": {
            "score_accuracy": accuracy,
            "efficiency": efficiency,
        },
        "explanation": {
            "score_accuracy": acc_exp,
            "efficiency": f"Used {steps_used} steps (efficiency={efficiency:.2f}).",
        },
    }


def grade_task_2(
    task: TaskSpec,
    submitted_score: Optional[float],
    submitted_feedback: Optional[str],
    steps_used: int,
) -> Dict:
    """
    Task 2 (Medium) grader.
    Weights: 60 % score accuracy + 40 % feedback quality.
    Requires ≥2 feedback points to pass.
    """
    accuracy, acc_exp = _score_accuracy(
        submitted_score, task.ground_truth_score, task.score_tolerance
    )

    # Extract keyword topics from ground-truth feedback
    topics = ["a/b test", "pytorch", "sagemaker", "cloud", "latency", "quantif"]
    fq, fq_exp = _feedback_quality(submitted_feedback, topics, min_items=2)

    total = round(accuracy * 0.60 + fq * 0.40, 4)

    return {
        "grade": total,
        "passed": total >= 0.60 and (submitted_feedback is not None and len(submitted_feedback) > 30),
        "components": {
            "score_accuracy": accuracy,
            "feedback_quality": fq,
        },
        "explanation": {
            "score_accuracy": acc_exp,
            "feedback_quality": fq_exp,
        },
    }


def grade_task_3(
    task: TaskSpec,
    submitted_score: Optional[float],
    submitted_feedback: Optional[str],
    steps_used: int,
) -> Dict:
    """
    Task 3 (Hard) grader.
    Weights: 50 % score accuracy + 50 % feedback quality.
    Requires ≥4 distinct improvement areas in feedback.
    """
    accuracy, acc_exp = _score_accuracy(
        submitted_score, task.ground_truth_score, task.score_tolerance
    )

    topics = [
        "manag", "budget", "p&l", "soc2", "security",
        "board", "executive", "mba", "msc", "saas",
        "team size", "roadmap",
    ]
    fq, fq_exp = _feedback_quality(submitted_feedback, topics, min_items=4)

    # Count bullet-like items in feedback as a depth signal
    if submitted_feedback:
        items = re.split(r"[\n\-\•\d\.]", submitted_feedback)
        item_count = sum(1 for i in items if len(i.strip()) > 15)
    else:
        item_count = 0

    depth_penalty = 0.0 if item_count >= 4 else (4 - item_count) * 0.05

    total = round(max(0.0, accuracy * 0.50 + fq * 0.50 - depth_penalty), 4)

    return {
        "grade": total,
        "passed": total >= 0.55 and item_count >= 4,
        "components": {
            "score_accuracy": accuracy,
            "feedback_quality": fq,
            "depth_penalty": -depth_penalty,
        },
        "explanation": {
            "score_accuracy": acc_exp,
            "feedback_quality": fq_exp,
            "depth": f"Detected {item_count} feedback items (need ≥4).",
        },
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "task_1": grade_task_1,
    "task_2": grade_task_2,
    "task_3": grade_task_3,
}


def grade(
    task: TaskSpec,
    submitted_score: Optional[float],
    submitted_feedback: Optional[str],
    steps_used: int,
) -> Dict:
    grader = GRADERS.get(task.task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task.task_id}")
    return grader(task, submitted_score, submitted_feedback, steps_used)
