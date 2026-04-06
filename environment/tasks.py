"""
Task definitions for the Resume Scorer OpenEnv environment.

Task 1 (Easy)   – Generic software-engineer resume vs. broad JD.
Task 2 (Medium) – Specialised ML-engineer JD with explicit skill requirements.
Task 3 (Hard)   – Senior leadership role; agent must produce high-quality
                   actionable feedback AND an accurate score.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class TaskSpec:
    task_id: str
    difficulty: str            # easy | medium | hard
    description: str
    job_description: str
    resume_text: str
    ground_truth_score: float  # 0.0 – 1.0
    required_skills: List[str]
    ground_truth_feedback: List[str]
    max_steps: int = 10
    score_tolerance: float = 0.15   # acceptable delta from ground truth


# ============================================================
# TASK 1 – Easy
# ============================================================
TASK_1 = TaskSpec(
    task_id="task_1",
    difficulty="easy",
    description=(
        "Score a junior software-engineer resume against a general "
        "full-stack web-developer job description. "
        "Expected score ≈ 0.65.  Submit the score within 3 steps."
    ),
    job_description="""\
Job Title: Junior Full-Stack Web Developer
Company: TechStartup Inc.

We are looking for a motivated Junior Full-Stack Web Developer to join our growing team.

Requirements:
- 0–2 years of experience in web development
- Proficiency in HTML, CSS, and JavaScript
- Familiarity with React or another modern front-end framework
- Basic knowledge of Python or Node.js for back-end development
- Experience with REST APIs
- Understanding of version control (Git)
- Good communication skills
- Bachelor's degree in Computer Science or related field (preferred)

Nice to have:
- Exposure to cloud services (AWS, GCP, or Azure)
- Experience with SQL or NoSQL databases
""",
    resume_text="""\
ALEX JOHNSON
alex.johnson@email.com | github.com/alexj | linkedin.com/in/alexj

SUMMARY
Recent Computer Science graduate with hands-on experience in React and Python.
Passionate about building clean, user-friendly web applications.

EDUCATION
B.Sc. Computer Science – State University (2023)  GPA 3.5/4.0

EXPERIENCE
Software Engineering Intern – WebAgency Co.  (Jun 2022 – Aug 2022)
  - Built 3 responsive landing pages using React and CSS
  - Integrated REST APIs from third-party services
  - Participated in daily stand-ups and sprint planning

SKILLS
Languages : JavaScript, Python, HTML, CSS
Frameworks: React, Flask
Tools     : Git, GitHub, VS Code, Postman

PROJECTS
Personal Portfolio (React, deployed on Netlify)
Todo App with Flask REST backend and React frontend

CERTIFICATIONS
freeCodeCamp – Responsive Web Design (2022)
""",
    ground_truth_score=0.65,
    required_skills=["javascript", "react", "python", "html", "css", "git", "rest api"],
    ground_truth_feedback=[
        "Add quantified achievements to the internship (e.g. page load improvements).",
        "Include database experience (SQL/NoSQL) to satisfy the nice-to-have.",
        "Expand projects section with links and tech-stack details.",
    ],
)


# ============================================================
# TASK 2 – Medium
# ============================================================
TASK_2 = TaskSpec(
    task_id="task_2",
    difficulty="medium",
    description=(
        "Score a mid-level ML-engineer resume against a specialised "
        "Machine Learning Engineer job description. "
        "Expected score ≈ 0.72.  Also provide at least 2 improvement suggestions."
    ),
    job_description="""\
Job Title: Machine Learning Engineer
Company: DataDriven Analytics

We seek an experienced Machine Learning Engineer to productionise ML models.

Requirements:
- 2–5 years of industry ML experience
- Proficiency in Python and ML libraries (TensorFlow or PyTorch)
- Experience deploying models via REST APIs (FastAPI / Flask)
- Knowledge of feature engineering, model evaluation, A/B testing
- Familiarity with MLflow or similar experiment tracking
- Experience with cloud ML platforms (AWS SageMaker, GCP Vertex AI)
- Strong understanding of SQL and data pipelines
- Version control (Git), CI/CD understanding

Nice to have:
- Kubernetes / Docker for model serving
- NLP or Computer Vision specialisation
- Publications or Kaggle rankings
""",
    resume_text="""\
PRIYA SHARMA
priya@email.com | github.com/priyaml

SUMMARY
ML Engineer with 3 years of experience building and deploying predictive models
in fintech and e-commerce domains. Strong Python background.

EDUCATION
M.Sc. Data Science – Tech University (2021)

EXPERIENCE
Machine Learning Engineer – FinTech Corp  (Jul 2021 – Present)
  - Trained and deployed credit-risk models achieving 12 % lift over baseline
  - Built FastAPI microservices to serve models in production (99.5 % uptime)
  - Used MLflow for experiment tracking across 40+ model iterations
  - Wrote complex SQL queries for feature engineering pipelines

Data Science Intern – E-Commerce Ltd  (Jan 2021 – Jun 2021)
  - Exploratory data analysis on 5 M-row purchase datasets
  - Built recommendation prototype using collaborative filtering (Python, Pandas)

SKILLS
Python, TensorFlow, scikit-learn, FastAPI, MLflow, SQL, Git, Docker
AWS (S3, EC2 – basic), Pandas, NumPy, Matplotlib

PROJECTS
Kaggle: Ranked top 15 % in two tabular-data competitions
Open-source contribution: mlflow-utils (GitHub, 180 stars)
""",
    ground_truth_score=0.72,
    required_skills=[
        "python", "tensorflow", "pytorch", "fastapi", "mlflow",
        "sql", "git", "aws", "docker", "model deployment",
    ],
    ground_truth_feedback=[
        "Add explicit mention of A/B testing experience.",
        "PyTorch is not listed — add if applicable to strengthen framework coverage.",
        "Expand cloud section beyond S3/EC2 (e.g. SageMaker pipeline experience).",
        "Quantify SLA/latency metrics on the FastAPI service.",
    ],
)


# ============================================================
# TASK 3 – Hard
# ============================================================
TASK_3 = TaskSpec(
    task_id="task_3",
    difficulty="hard",
    description=(
        "Score a senior engineering-manager resume against a VP Engineering JD. "
        "Expected score ≈ 0.55 (notable gaps). "
        "Agent must produce an accurate score AND detailed, actionable feedback "
        "covering at least 4 improvement areas."
    ),
    job_description="""\
Job Title: VP of Engineering
Company: ScaleUp SaaS

We are seeking a VP of Engineering to lead our 60-person engineering organisation.

Requirements:
- 10+ years in software engineering; 5+ years in engineering leadership
- Proven track record managing managers (org size 30+)
- Experience with P&L ownership and engineering budget planning
- Defined and executed multi-year technical roadmaps
- Deep expertise in distributed systems and cloud-native architecture
- Experience with SOC 2 compliance and security best practices
- Strong hiring and talent-development track record
- Excellent executive-level communication and board presentation experience
- M.Sc. or MBA preferred

Nice to have:
- Experience with SaaS metrics (ARR, churn, NPS)
- Open-source leadership or conference speaking
- International team management
""",
    resume_text="""\
MARCUS LEE
marcus.lee@email.com

SUMMARY
Engineering Manager with 8 years in software engineering and 3 years managing
a single team of 8 engineers. Passionate about agile processes and clean code.

EDUCATION
B.Sc. Electrical Engineering – City College (2015)

EXPERIENCE
Engineering Manager – MidCorp SaaS  (2020 – Present)
  - Manages a team of 8 backend engineers
  - Conducted quarterly performance reviews
  - Drove adoption of CI/CD pipelines; reduced deploy time by 40 %
  - Contributed to hiring 4 engineers over 2 years

Senior Software Engineer – MidCorp SaaS  (2017 – 2020)
  - Designed microservices architecture handling 10 k RPS
  - Led migration from monolith to Kubernetes-based deployment
  - Mentored 2 junior engineers

Software Engineer – StartupX  (2015 – 2017)
  - Full-stack development in Node.js and React

SKILLS
Python, Node.js, Kubernetes, AWS, React, PostgreSQL, CI/CD, Agile

ACHIEVEMENTS
Reduced cloud spend by $120 k/year through resource optimisation
""",
    ground_truth_score=0.55,
    required_skills=[
        "engineering leadership", "distributed systems", "cloud-native",
        "p&l", "technical roadmap", "soc2", "talent development",
        "board communication", "managing managers",
    ],
    ground_truth_feedback=[
        "Only 3 years of management experience vs. 5+ years required; clarify scope.",
        "No evidence of managing managers — team size of 8 is too small for VP role.",
        "Missing P&L / budget ownership experience; add if applicable.",
        "No mention of SOC 2 compliance or security best practices.",
        "Executive communication / board-level presentation experience absent.",
        "Upgrade education section — M.Sc. or MBA preferred; note any exec courses.",
        "Add SaaS business metrics (ARR, churn) if familiar with them.",
    ],
    max_steps=15,
    score_tolerance=0.12,
)


ALL_TASKS = {
    "task_1": TASK_1,
    "task_2": TASK_2,
    "task_3": TASK_3,
}
