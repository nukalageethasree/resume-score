"""
app.py – FastAPI backend for Resume Scorer OpenEnv.
Serves the HTML frontend and exposes a /score API endpoint.

Run:
    python app.py
Then open: http://localhost:7860
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

from environment import ResumeScorerEnv, Action
from environment.tasks import ALL_TASKS

# PDF extraction (optional)
try:
    import fitz
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

app = FastAPI(title="Resume Scorer OpenEnv")


# ── Serve the single-page frontend ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "frontend" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Score endpoint ───────────────────────────────────────────────────────────
@app.post("/score")
async def score_resume(
    job_description: str = Form(...),
    resume_text: str = Form(""),
    task_id: str = Form("task_1"),
    pdf_file: Optional[UploadFile] = File(None),
):
    try:
        # Resolve resume text
        resume = resume_text.strip()
        if pdf_file and pdf_file.filename:
            raw = await pdf_file.read()
            if HAS_PDF:
                doc = fitz.open(stream=raw, filetype="pdf")
                extracted = "\n".join(p.get_text() for p in doc)
                if extracted.strip():
                    resume = extracted
            else:
                try:
                    resume = raw.decode("utf-8", errors="ignore")
                except Exception:
                    pass

        if not resume:
            return JSONResponse({"error": "No resume text provided."}, status_code=400)
        if not job_description.strip():
            return JSONResponse({"error": "No job description provided."}, status_code=400)
        if task_id not in ALL_TASKS:
            task_id = "task_1"

        env = ResumeScorerEnv()
        obs = env.reset(
            task_id=task_id,
            job_description=job_description.strip(),
            resume_text=resume,
        )

        # ── Heuristic scoring ────────────────────────────────────────────
        skill_ratio = obs.skill_match_ratio
        section_score = sum([
            obs.sections_present.contact_info,
            obs.sections_present.summary,
            obs.sections_present.education,
            obs.sections_present.experience,
            obs.sections_present.skills,
            obs.sections_present.projects,
        ]) / 6.0
        edu_bonus = {"phd": 0.10, "master": 0.07, "bachelor": 0.04}.get(
            obs.education_level or "", 0.0
        )
        exp_bonus = min(obs.years_experience_detected / 10.0, 0.10)
        raw_score = skill_ratio * 0.55 + section_score * 0.25 + edu_bonus + exp_bonus
        final_score = round(min(max(raw_score, 0.0), 1.0), 4)

        env.step(Action(action_type="submit_score", score=final_score))

        # ── Feedback generation ──────────────────────────────────────────
        gaps = [sm.skill for sm in obs.skill_matches if not sm.found_in_resume]
        matched = [sm.skill for sm in obs.skill_matches if sm.found_in_resume]
        missing_sections = [
            k for k, v in obs.sections_present.model_dump().items() if not v
        ]

        feedback = []
        if gaps:
            feedback.append({
                "type": "gap", "icon": "⚠️",
                "title": "Missing Skills",
                "body": f"Add these skills if applicable: {', '.join(gaps[:8])}."
            })
        if missing_sections:
            feedback.append({
                "type": "section", "icon": "📋",
                "title": "Missing Resume Sections",
                "body": f"Add: {', '.join(missing_sections)} to improve ATS scoring."
            })
        if obs.word_count < 250:
            feedback.append({
                "type": "length", "icon": "📝",
                "title": "Resume Too Brief",
                "body": "Expand experience with quantified achievements (numbers, %, $)."
            })
        if obs.years_experience_detected == 0:
            feedback.append({
                "type": "experience", "icon": "🗓️",
                "title": "Experience Dates Not Detected",
                "body": "Use clear date ranges like 'Jan 2022 – Dec 2023' per role."
            })
        if not feedback:
            feedback.append({
                "type": "success", "icon": "✅",
                "title": "Strong Match",
                "body": "Solid resume for this role. Quantify all achievements for max impact."
            })

        fb_text = " | ".join(f["body"] for f in feedback)
        env.step(Action(action_type="submit_feedback", feedback=fb_text))
        _, _, _, info = env.step(Action(action_type="finalize", score=final_score))

        return JSONResponse({
            "score": final_score,
            "percent": int(final_score * 100),
            "skill_match_ratio": round(obs.skill_match_ratio, 4),
            "section_score": round(section_score, 4),
            "years_experience": obs.years_experience_detected,
            "internship_count": obs.internship_count,
            "education_level": obs.education_level or "Not detected",
            "word_count": obs.word_count,
            "skills_matched": matched,
            "skills_missing": gaps,
            "sections_present": [k for k, v in obs.sections_present.model_dump().items() if v],
            "sections_missing": missing_sections,
            "feedback": feedback,
            "task_id": task_id,
            "difficulty": ALL_TASKS[task_id].difficulty,
            "episode_grade": info.get("episode_grade", {}).get("grade", 0),
        })

    except Exception:
        return JSONResponse({"error": traceback.format_exc()}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "tasks": list(ALL_TASKS.keys())}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
