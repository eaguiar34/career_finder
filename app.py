"""
Career Fair Companion — Streamlit App (Multi‑Major + Apply + Resume + Themes + Assistant)
Author: You 😊
License: MIT

New in this version:
- Multi‑major support: presets for many majors (Engineering, CS, Data, Business/Finance, Marketing/Comms, Design/UX, Life Sciences, Social Sciences, Accounting, **Nursing, Education, Architecture, Environmental Science**).
- One‑click "Apply Preset" fills role, interests, and assistant defaults; tools/examples in recruiter questions adapt to the chosen major.
- AI Assistant + preference‑driven search + opt‑in resume tailoring remain.

Run locally:
  pip install -r requirements.txt
  streamlit run app.py

Deploy free on Streamlit Community Cloud:
  Push to GitHub and deploy. Secrets supported: openai_api_key, openai_model, openai_base_url.
"""

from __future__ import annotations
import re
import os
import io
import json
import difflib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st

# --- Optional libs ---
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None
try:
    import wikipedia
except Exception:
    wikipedia = None
try:
    import yake
except Exception:
    yake = None
try:
    import segno
except Exception:
    segno = None
try:
    from openai import OpenAI  # official or compatible via base_url
except Exception:
    OpenAI = None
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None
try:
    from docx import Document
except Exception:
    Document = None

from bs4 import BeautifulSoup  # type: ignore
import requests

# ---------------------------
# Data Paths
# ---------------------------
DATA_DIR = os.path.join("data")
RESUME_DIR = os.path.join(DATA_DIR, "resumes")
os.makedirs(RESUME_DIR, exist_ok=True)
INDEX_CSV = os.path.join(RESUME_DIR, "index.csv")

# ---------------------------
# Themes
# ---------------------------
DEFAULT_PRIMARY = "#6C63FF"   # pop purple
DEFAULT_ACCENT  = "#00D4FF"   # neon cyan

SCHOOL_THEMES = {
    "Cal State LA": {"primary": "#000000", "accent": "#FFC72C"},
    "UCLA":         {"primary": "#2774AE", "accent": "#FFD100"},
    "USC":          {"primary": "#990000", "accent": "#FFCC00"},
    "UC Berkeley":  {"primary": "#003262", "accent": "#FDB515"},
    "MIT":          {"primary": "#A31F34", "accent": "#8A8B8C"},
}

# ---------------------------
# Multi‑Major Presets
# ---------------------------
MAJOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "(Custom)": {
        "role": "Intern",
        "interests": ["projects", "research", "leadership"],
        "tasks": "",
        "skills": "",
        "tools": ["Excel", "Python"],
    },
    "Civil Engineering": {
        "role": "Civil Engineering Intern",
        "interests": ["transportation", "water", "sustainability", "Python"],
        "tasks": "transportation design, hydrology/hydraulics, site visits, CAD/BIM",
        "skills": "Civil 3D, AutoCAD, HEC-RAS, BIM, GIS",
        "tools": ["Civil 3D", "AutoCAD", "HEC-RAS", "MicroStation", "BIM", "GIS"],
    },
    "Mechanical Engineering": {
        "role": "Mechanical Engineering Intern",
        "interests": ["product design", "FEA", "manufacturing", "MATLAB"],
        "tasks": "CAD modeling, prototyping, testing, DFM/DFA",
        "skills": "SolidWorks, CATIA, ANSYS, MATLAB",
        "tools": ["SolidWorks", "CATIA", "ANSYS", "MATLAB", "Onshape"],
    },
    "Electrical/Computer Engineering": {
        "role": "Electrical Engineering Intern",
        "interests": ["embedded", "pcb", "firmware", "power"],
        "tasks": "PCB design, firmware, lab testing",
        "skills": "Altium, SPICE, Verilog, Python",
        "tools": ["Altium", "KiCad", "SPICE", "Verilog", "MATLAB"],
    },
    "Computer Science / Software": {
        "role": "Software Engineer Intern",
        "interests": ["backend", "web", "AI", "data"],
        "tasks": "APIs, web apps, data pipelines",
        "skills": "Python, JavaScript, React, Node, SQL",
        "tools": ["Python", "JS/TS", "React", "Node", "SQL"],
    },
    "Data / Analytics": {
        "role": "Data Analyst Intern",
        "interests": ["analytics", "dashboards", "A/B testing", "SQL"],
        "tasks": "dashboards, ETL, experimentation",
        "skills": "SQL, Python, Tableau, Power BI",
        "tools": ["SQL", "Python", "Tableau", "Power BI", "R"],
    },
    "Business / Finance": {
        "role": "Finance Intern",
        "interests": ["FP&A", "valuation", "modeling", "PowerPoint"],
        "tasks": "financial modeling, reporting, variance analysis",
        "skills": "Excel, PowerPoint, Bloomberg",
        "tools": ["Excel", "PowerPoint", "Bloomberg", "SAP"],
    },
    "Marketing / Communications": {
        "role": "Marketing Intern",
        "interests": ["content", "SEO", "social", "analytics"],
        "tasks": "campaigns, content, SEO/SEM, analytics",
        "skills": "GA, SEO tools, CMS",
        "tools": ["Google Analytics", "Search Console", "Hootsuite", "HubSpot", "Canva"],
    },
    "Design / UX": {
        "role": "UX Designer Intern",
        "interests": ["user research", "prototyping", "accessibility"],
        "tasks": "wireframes, prototypes, usability studies",
        "skills": "Figma, Adobe XD, Sketch",
        "tools": ["Figma", "Sketch", "Adobe XD", "Miro"],
    },
    "Biology / Life Sciences": {
        "role": "Research Assistant",
        "interests": ["wet lab", "PCR", "cell culture", "data analysis"],
        "tasks": "experiments, protocols, data",
        "skills": "R, Python, Prism",
        "tools": ["Pipettes", "PCR", "Flow cytometry", "R", "Python"],
    },
    "Psychology / Social Science": {
        "role": "Research Assistant",
        "interests": ["experiment design", "stats", "survey"],
        "tasks": "study design, data collection, analysis",
        "skills": "R, SPSS, Qualtrics",
        "tools": ["Qualtrics", "SPSS", "R", "Excel"],
    },
    "Accounting": {
        "role": "Accounting Intern",
        "interests": ["audit", "tax", "GAAP"],
        "tasks": "bookkeeping, audit/tax support",
        "skills": "Excel, QuickBooks",
        "tools": ["Excel", "QuickBooks", "SAP"],
    },
    # === Newly added majors ===
    "Nursing": {
        "role": "Student Nurse / Nursing Intern",
        "interests": ["clinical", "patient care", "med-surg", "ICU", "EMR"],
        "tasks": "vitals, charting, patient education, care coordination",
        "skills": "BLS, EMR/EHR, phlebotomy, infection control",
        "tools": ["Epic", "Cerner", "MEDITECH", "Vitals monitors"],
    },
    "Education": {
        "role": "Teaching Assistant / Education Intern",
        "interests": ["curriculum", "lesson planning", "edtech", "assessment"],
        "tasks": "lesson prep, grading, classroom management, small‑group instruction",
        "skills": "Google Classroom, Canvas, formative assessment, IEP basics",
        "tools": ["Google Classroom", "Canvas", "Kahoot!", "Nearpod", "Zoom"],
    },
    "Architecture": {
        "role": "Architecture Intern",
        "interests": ["concept design", "BIM", "sustainable design", "3D modeling"],
        "tasks": "drafting, redlines, 3D modeling, rendering, documentation",
        "skills": "Revit, AutoCAD, Rhino, SketchUp, Adobe Suite",
        "tools": ["Revit", "AutoCAD", "Rhino", "SketchUp", "Enscape"],
    },
    "Environmental Science": {
        "role": "Environmental Science Intern",
        "interests": ["field sampling", "GIS", "EHS", "air/water", "sustainability"],
        "tasks": "sampling, data analysis, QA/QC, report writing",
        "skills": "ArcGIS/QGIS, R, Excel, HAZWOPER awareness",
        "tools": ["ArcGIS", "QGIS", "R", "Excel", "Handheld meters"],
    },
}

# ---------------------------
# Page & Base Styles
# ---------------------------
st.set_page_config(
    page_title="Career Fair Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .app-title {{
        background: linear-gradient(90deg, {DEFAULT_PRIMARY}, {DEFAULT_ACCENT});
        -webkit-background-clip: text; background-clip: text; color: transparent;
        font-weight: 800; letter-spacing: -0.5px;
      }}
      .card {{
        border-radius: 16px; padding: 1rem 1.25rem; border: 1px solid #e5e7eb; background: white; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
      }}
      .btn-primary button {{ background: {DEFAULT_PRIMARY} !important; border-color: {DEFAULT_PRIMARY} !important; }}
      .btn-accent button {{ background: {DEFAULT_ACCENT} !important; border-color: {DEFAULT_ACCENT} !important; color: black !important; }}
      .muted {{ color: #94a3b8; }}
      .small {{ font-size: 0.85rem; }}
      .match-pill {{ padding: 2px 8px; border-radius: 999px; background: #ecfeff; border:1px solid #a5f3fc; font-weight:600; }}
      .pref-pill {{ padding: 2px 8px; border-radius: 999px; background: #f0fdf4; border:1px solid #bbf7d0; font-weight:500; margin-right:6px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

def apply_theme(primary: str, accent: str):
    st.markdown(
        f"""
        <style>
          .app-title {{
            background: linear-gradient(90deg, {primary}, {accent});
            -webkit-background-clip: text; background-clip: text; color: transparent;
          }}
          .btn-primary button {{ background: {primary} !important; border-color: {primary} !important; }}
          .btn-accent button {{ background: {accent} !important; border-color: {accent} !important; color: black !important; }}
          .match-pill {{ background: {accent}20; border-color: {accent}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# AI Client
# ---------------------------

def get_ai_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None
    api_key = st.secrets.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = st.secrets.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        return None
    try:
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)
    except Exception:
        return None

# ---------------------------
# Search + Parsing Helpers
# ---------------------------

def ddg_search(query: str, max_results: int = 10, timelimit: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if DDGS is None:
        return rows
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, timelimit=timelimit, safesearch="moderate"):
                rows.append({"title": r.get("title"), "href": r.get("href"), "body": r.get("body")})
    except Exception:
        pass
    return rows


def clean_snippet(text: Optional[str], length: int = 220) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", str(text)).strip()
    return (t if len(t) <= length else t[: length - 1].rsplit(" ", 1)[0] + "…")


def wiki_summary(company: str) -> Optional[str]:
    if not wikipedia:
        return None
    try:
        wikipedia.set_lang("en")
        page_title = wikipedia.search(company, results=1)
        if not page_title:
            return None
        page = wikipedia.page(page_title[0])
        return clean_snippet(page.summary, 600)
    except Exception:
        return None


def extract_keywords(text: str, top_k: int = 12) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    if yake is not None:
        try:
            kw = yake.KeywordExtractor(lan="en", n=1, top=top_k)
            candidates = [k for k, _ in kw.extract_keywords(text)]
            seen = set(); uniq = []
            for c in candidates:
                c = c.lower()
                if c not in seen and c.isalpha():
                    seen.add(c); uniq.append(c)
            return uniq[:top_k]
        except Exception:
            pass
    words = re.findall(r"[A-Za-z][A-Za-z\-\+\.]{1,}", text.lower())
    stop = set("""
        a an and are as at be by for from has have in is it its of on or that the to with your you we they this
        our their will can us if not but than then over under more less most least etc job role position company
    """.split())
    freq: Dict[str, int] = {}
    for w in words:
        if len(w) <= 2 or w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]


def categorize_domain(url: str) -> str:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0]
    except Exception:
        return "Other"
    if any(k in host for k in ["lever.co", "greenhouse.io", "myworkdayjobs.com", "smartrecruiters.com", "icims.com"]):
        return "ATS"
    if any(k in host for k in ["indeed.com", "linkedin.com/jobs", "glassdoor.com/job"]):
        return "Job Board"
    if any(k in host for k in ["newsroom", "press", "media", "blog."]):
        return "Press/Blog"
    if "linkedin.com/in" in host:
        return "LinkedIn"
    return "Other"


def fetch_text(url: str, limit: int = 6000) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (CareerFairComp/1.0)"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        txt = soup.get_text(" ", strip=True)
        return clean_snippet(txt, length=limit)
    except Exception:
        return ""

# ---------------------------
# Projects / Jobs (company‑centric)
# ---------------------------

def recent_projects(company: str, max_results: int = 12) -> List[Dict[str, Any]]:
    queries = [
        f"{company} recent projects",
        f"{company} case study",
        f"{company} wins contract",
        f"{company} announces project",
        f"{company} press release project",
    ]
    rows: List[Dict[str, Any]] = []
    for q in queries:
        for r in ddg_search(q, max_results=5, timelimit="y"):
            rows.append({"title": r["title"], "url": r["href"], "snippet": clean_snippet(r["body"]), "source": categorize_domain(r["href"])})
    seen = set(); uniq = []
    for row in rows:
        u = row["url"]
        if u in seen: continue
        seen.add(u); uniq.append(row)
    return uniq[:max_results]


def job_search(company: str, role: str = "", location: str = "", max_results: int = 18) -> List[Dict[str, Any]]:
    base = [
        f"site:lever.co {company} {role}",
        f"site:greenhouse.io {company} {role}",
        f"site:myworkdayjobs.com {company} {role}",
        f"site:smartrecruiters.com {company} {role}",
        f"site:icims.com {company} {role}",
        f"{company} careers {role}",
    ]
    if location:
        base = [q + f" {location}" for q in base]
    rows: List[Dict[str, Any]] = []
    for q in base:
        for r in ddg_search(q, max_results=6, timelimit="m"):
            rows.append({"title": r["title"], "url": r["href"], "snippet": clean_snippet(r["body"]), "source": categorize_domain(r["href"])})
    seen = set(); uniq = []
    for row in rows:
        u = row["url"]
        if u in seen: continue
        seen.add(u); uniq.append(row)
    return uniq[:max_results]

# ---------------------------
# Preference‑driven search (cross‑company)
# ---------------------------

def prefs_to_queries(p: Dict[str, Any]) -> List[str]:
    role = p.get("role","")
    tasks = p.get("tasks","")
    skills = p.get("skills","")
    location = p.get("location","")
    companies = [c for c in (p.get("companies") or []) if c]
    base_terms = f"{role} {skills} {tasks}".strip()
    seeds = [
        "site:lever.co", "site:greenhouse.io", "site:myworkdayjobs.com", "site:smartrecruiters.com", "site:icims.com",
        "careers",
    ]
    queries = []
    if companies:
        for c in companies:
            queries += [f"{s} {c} {base_terms}" for s in seeds]
    else:
        queries += [f"{s} {base_terms}" for s in seeds]
    if location:
        queries = [q + f" {location}" for q in queries]
    return queries


def parse_pay(text: str) -> Optional[Tuple[float,float,str]]:
    t = text.lower().replace(",","")
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*(?:-|to)\s*\$\s*(\d+(?:\.\d+)?)(?:\s*/?\s*(hour|hr|year|yr|annum))?", t)
    if not m:
        m = re.search(r"\$\s*(\d{2,6})(?:\s*/?\s*(hour|hr|year|yr))", t)
        if m:
            v = float(m.group(1)); unit = (m.group(2) or "year").lower()
            return (v, v, unit)
        return None
    lo = float(m.group(1)); hi = float(m.group(2)); unit = (m.group(3) or "year").lower()
    return (min(lo,hi), max(lo,hi), unit)


def score_job_vs_prefs(job: Dict[str, Any], prefs: Dict[str, Any]) -> Tuple[int, List[str]]:
    title = job.get("title") or ""; body = job.get("snippet") or ""
    if not body:
        body = fetch_text(job.get("url",""))
    text = (title + "\n" + body).lower()

    task_kw = re.findall(r"[a-z0-9\+\-]+", (prefs.get("tasks","") or "").lower())
    skill_kw = re.findall(r"[a-z0-9\+\-]+", (prefs.get("skills","") or "").lower())
    loc = (prefs.get("location") or "").lower()
    remote = str(prefs.get("remote") or "").lower()
    pay_min = prefs.get("pay_min"); pay_max = prefs.get("pay_max")

    details = []
    role_ratio = difflib.SequenceMatcher(a=(prefs.get("role","" ) or "").lower(), b=text).ratio() if prefs.get("role") else 0.0
    def hit_count(kws):
        return sum(1 for k in kws if re.search(fr"\b{re.escape(k)}\b", text))
    task_hits = hit_count(task_kw)
    skill_hits = hit_count(skill_kw)

    if role_ratio > 0.4:
        details.append(f"role match ~{int(role_ratio*100)}%")
    if task_hits:
        details.append(f"{task_hits} task keyword(s)")
    if skill_hits:
        details.append(f"{skill_hits} skill keyword(s)")
    loc_hit = bool(loc and loc in text)
    if loc_hit:
        details.append("location match")
    if remote:
        r_hit = ("remote" in text) or ("hybrid" in text and remote == "hybrid")
        if r_hit:
            details.append("remote/hybrid match")
    pay_info = parse_pay(text)
    pay_ok = False
    if pay_info and (pay_min or pay_max):
        lo, hi, unit = pay_info
        if unit.startswith("hour"):
            lo, hi = lo*2080, hi*2080
        low_ok = (pay_min is None) or (hi >= pay_min)
        hi_ok = (pay_max is None) or (lo <= pay_max)
        pay_ok = low_ok and hi_ok
        if pay_ok:
            details.append("pay within range")

    score = (
        35 * role_ratio +
        20 * (task_hits > 0) +
        20 * (skill_hits > 0) +
        10 * (1 if loc_hit else 0) +
        5  * (1 if pay_ok else 0) +
        10 * (1 if job.get("source") == "ATS" else 0)
    )
    score = max(0, min(100, int(round(score))))
    if not details:
        details.append("basic text match")
    return score, details

# ---------------------------
# % Match scoring for jobs (company view)
# ---------------------------

def _norm_tokens(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\+\-\s]", " ", s)
    return [t for t in s.split() if len(t) > 2]


def score_job(job: Dict[str, Any], role: str, interests: List[str], location: str) -> Tuple[int, str]:
    title = job.get("title") or ""
    snip = job.get("snippet") or ""
    url = job.get("url") or ""
    body = snip or fetch_text(url)

    role_ratio = difflib.SequenceMatcher(a=role.lower(), b=(title + " " + body).lower()).ratio() if role else 0.0
    body_tokens = set(_norm_tokens(title + " " + body))
    interests_norm = [i.lower().strip() for i in interests if i.strip()]
    overlap = sum(1 for i in interests_norm if any(tok in body_tokens for tok in _norm_tokens(i)))
    interest_ratio = (overlap / max(1, len(interests_norm))) if interests_norm else 0.0
    loc_hit = 1 if (location and location.lower() in (title + " " + body + " " + url).lower()) else 0
    src = job.get("source", "Other")
    source_bonus = {"ATS": 1.0, "Job Board": 0.6, "Other": 0.4}.get(src, 0.4)

    score = 40 * role_ratio + 40 * interest_ratio + 10 * loc_hit + 10 * source_bonus
    score = max(0, min(100, int(round(score))))

    reason = f"role~{int(role_ratio*100)}%, interests {overlap}/{len(interests_norm) or 1}, location {'✓' if loc_hit else '—'}, source {src}"
    return score, reason

# ---------------------------
# Questions (LLM or heuristic)
# ---------------------------

def smart_questions(company: str, role: str, interests: List[str], job_desc: Optional[str] = None, tool_examples: Optional[List[str]] = None) -> Dict[str, List[str]]:
    topics = set(i.strip() for i in interests if i.strip())
    if job_desc:
        topics.update(extract_keywords(job_desc, top_k=8))
    topics = {t for t in topics if 2 < len(t) < 28 and re.match(r"^[A-Za-z][A-Za-z0-9\-\+\.\s]+$", t)}
    topic_sample = list(topics)[:6]
    tt = ", ".join(topic_sample) if topic_sample else "the core skills for this role"
    tool_text = ", ".join(tool_examples[:6]) if tool_examples else "industry‑standard tools"
    buckets = {
        "Role & Impact": [
            f"How does {role or 'this role'} define success in the first 90 days, and what metrics matter most?",
            f"What projects are top priority this quarter, and how would someone in {role or 'this role'} contribute?",
            f"Among {tt}, which skills would help a candidate stand out in interviews?",
        ],
        "Team & Mentorship": [
            "What does onboarding look like, and how soon do interns/new grads ship meaningful work?",
            "How is feedback delivered on the team — regular 1:1s, reviews, shadowing?",
            "Could you share an example of a recent cross‑team collaboration that went well, and why?",
        ],
        "Projects & Methods": [
            f"Which tools are standard here (e.g., {tool_text}) and how consistently are they used across projects?",
            f"Does the team have a preferred approach for scoping projects and managing risk on tight timelines?",
            f"For a student interested in {tt}, what projects or case studies would you suggest I study?",
        ],
        "Culture & Growth": [
            "How do interns/new grads find mentors and stretch projects? Any formal rotation or training programs?",
            "How does the company encourage learning (certifications, conference budgets, courses)?",
            "What traits do you see in people who thrive here in their first year?",
        ],
        "Recruiter‑Specific": [
            f"Based on my interest in {tt}, which teams should I keep an eye on?",
            f"What are common pitfalls in applications for {role or 'this role'}, and how can I avoid them?",
            f"If I follow up after the fair, what should I include to make it easiest for you to move me forward?",
        ],
    }
    if company:
        for k in buckets:
            buckets[k] = [q.replace("the company", company).replace("this company", company) for q in buckets[k]]
    return buckets

# ---------------------------
# AI wrappers
# ---------------------------

def summarize_ai(client: "OpenAI", text: str, model: Optional[str] = None) -> Optional[str]:
    try:
        mdl = st.secrets.get("openai_model") or os.environ.get("OPENAI_MODEL") or model or "gpt-4o-mini"
        out = client.chat.completions.create(model=mdl, messages=[
            {"role": "system", "content": "You write crisp, factual summaries for career prep."},
            {"role": "user", "content": f"Summarize this in 4-5 sentences for a student:\n\n{text}"},
        ], temperature=0.4)
        return out.choices[0].message.content.strip()
    except Exception:
        return None


def tailor_resume_ai(client: "OpenAI", resume_text: str, company: str, role: str, job_text: str) -> Optional[str]:
    try:
        mdl = st.secrets.get("openai_model") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        system = (
            "You are an expert resume editor for early‑career candidates. Rewrite the resume to target the company and role. "
            "Preserve truthful content, quantify impact where possible, add strong verbs, and align with the job description. "
            "Keep a clean sectioned structure (Summary, Education, Skills, Experience, Projects, Leadership). Output plain text."
        )
        user = (
            f"Company: {company}\nRole: {role}\nJob description/keywords:\n{job_text}\n\nOriginal resume:\n{resume_text}\n\nReturn ONLY the revised resume as plain text."
        )
        out = client.chat.completions.create(model=mdl, messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.4)
        return out.choices[0].message.content.strip()
    except Exception:
        return None

# ---------------------------
# Sidebar Inputs (now with Major Preset)
# ---------------------------

# init state holders for sticky inputs
if "role_input" not in st.session_state:
    st.session_state.role_input = MAJOR_PRESETS["Civil Engineering"]["role"]
if "interests_input" not in st.session_state:
    st.session_state.interests_input = ", ".join(MAJOR_PRESETS["Civil Engineering"]["interests"])
if "preset_tools" not in st.session_state:
    st.session_state.preset_tools = MAJOR_PRESETS["Civil Engineering"]["tools"]
if "pref_defaults" not in st.session_state:
    st.session_state.pref_defaults = {
        "role": MAJOR_PRESETS["Civil Engineering"]["role"],
        "tasks": MAJOR_PRESETS["Civil Engineering"]["tasks"],
        "skills": MAJOR_PRESETS["Civil Engineering"]["skills"],
    }

st.sidebar.header("🎓 Major Preset + Research Inputs")
major = st.sidebar.selectbox("Major (preset)", options=list(MAJOR_PRESETS.keys()), index=list(MAJOR_PRESETS.keys()).index("Civil Engineering"))

if st.sidebar.button("Apply Preset"):
    p = MAJOR_PRESETS.get(major, MAJOR_PRESETS["(Custom)"])
    st.session_state.role_input = p.get("role", "Intern")
    st.session_state.interests_input = ", ".join(p.get("interests", []))
    st.session_state.preset_tools = p.get("tools", [])
    st.session_state.pref_defaults = {
        "role": p.get("role", "Intern"),
        "tasks": p.get("tasks", ""),
        "skills": p.get("skills", ""),
    }

company = st.sidebar.text_input("Company name", placeholder="e.g., Jacobs, AECOM, Google")
role = st.sidebar.text_input("Target role/title", value=st.session_state.role_input, key="role_sidebar")
location = st.sidebar.text_input("Location filter (optional)", placeholder="e.g., Los Angeles, CA")
interests_str = st.sidebar.text_input("Interests / keywords (comma‑separated)", value=st.session_state.interests_input, key="interests_sidebar")
interests = [s.strip() for s in interests_str.split(",") if s.strip()]

st.sidebar.subheader("🎨 Theme")
school = st.sidebar.selectbox("School", options=["(Default)"] + list(SCHOOL_THEMES.keys()), index=0)
custom_primary = st.sidebar.color_picker("Primary", value=SCHOOL_THEMES.get("Cal State LA", {}).get("primary", DEFAULT_PRIMARY))
custom_accent  = st.sidebar.color_picker("Accent", value=SCHOOL_THEMES.get("Cal State LA", {}).get("accent", DEFAULT_ACCENT))

# Compute theme
p = DEFAULT_PRIMARY; a = DEFAULT_ACCENT
if school != "(Default)":
    t = SCHOOL_THEMES.get(school, {})
    p = t.get("primary", p); a = t.get("accent", a)
# custom overrides
p = custom_primary or p
a = custom_accent or a
apply_theme(p, a)

st.sidebar.subheader("🤖 AI Settings")
use_ai = st.sidebar.toggle("Use AI (OpenAI or compatible)", value=False)
model_hint = st.sidebar.text_input("Model (optional)", value=os.environ.get("OPENAI_MODEL", ""))
base_url_hint = st.sidebar.text_input("Base URL (optional)", value=os.environ.get("OPENAI_BASE_URL", ""))
if base_url_hint:
    st.caption(f"Using custom base_url: {base_url_hint}")

notes = st.sidebar.text_area("Personal notes (kept on device)", height=100)

# Actions
colA, colB = st.sidebar.columns(2)
with colA:
    go_research = st.button("Research", type="primary")
with colB:
    go_questions = st.button("Make Questions")
colC, colD = st.sidebar.columns(2)
with colC:
    go_jobs = st.button("Find Jobs")
with colD:
    go_export = st.button("Export Sheet")

# ---------------------------
# Header
# ---------------------------

st.markdown("""
<div class="card" style="margin-bottom: 0.75rem;">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="font-size:2.3rem;">🎓</div>
    <div>
      <div class="app-title" style="font-size:2rem;">Career Fair Companion</div>
      <div class="muted small">Research companies fast. Ask sharper questions. Find relevant roles (for any major). Tailor resumes. Swap QR cards.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Session State
# ---------------------------

for key, default in {
    "projects": [],
    "jobs": [],
    "pref_jobs": [],
    "questions": {},
    "wiki": None,
    "saved": [],
    "contacts": [],
    "resume_text": "",
    "resume_bytes": b"",
    "resume_ext": "",
    "assistant_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------
# Actions logic — company view
# ---------------------------

if go_research and company:
    with st.spinner("Researching…"):
        st.session_state.wiki = wiki_summary(company)
        if use_ai and (client := get_ai_client()) and st.session_state.wiki:
            s = summarize_ai(client, st.session_state.wiki, model_hint)
            if s:
                st.session_state.wiki = s
        st.session_state.projects = recent_projects(company)
    st.success("Company research updated.")

if go_jobs and company:
    with st.spinner("Searching for jobs…"):
        base_jobs = job_search(company, role=role, location=location)
        scored = []
        for j in base_jobs:
            pct, why = score_job(j, role, interests, location)
            j2 = {**j, "match": pct, "why": why}
            scored.append(j2)
        st.session_state.jobs = sorted(scored, key=lambda x: x.get("match", 0), reverse=True)
    st.success("Job results updated.")

if go_questions:
    with st.spinner("Generating questions…"):
        st.session_state.questions = smart_questions(company, role, interests, tool_examples=st.session_state.get("preset_tools", []))
    st.success("Questions ready.")

if go_export:
    def build_cheatsheet(company: str, role: str, wiki: Optional[str], projects: List[Dict[str, Any]], jobs: List[Dict[str, Any]], questions: Dict[str, List[str]], notes: str = "") -> str:
        now = datetime.now().strftime("%b %d, %Y")
        lines = [f"# {company} — {role} (Career Fair Companion)", f"_Generated {now}_\n"]
        if wiki:
            lines += ["## Snapshot", wiki, ""]
        if projects:
            lines.append("## Recent Projects / News")
            for p in projects[:8]:
                lines.append(f"- [{p['title']}]({p['url']}) — {p['snippet']}")
            lines.append("")
        if jobs:
            lines.append("## Job Listings (sorted by match)")
            for j in sorted(jobs, key=lambda x: x.get('match',0), reverse=True)[:10]:
                lines.append(f"- [{j['title']}]({j['url']}) — **{j.get('match',0)}% match** *({j['source']})* — {j['snippet']}")
            lines.append("")
        if questions:
            lines.append("## Questions for Recruiters")
            for bucket, qs in questions.items():
                lines.append(f"### {bucket}")
                for q in qs:
                    lines.append(f"- {q}")
            lines.append("")
        if notes:
            lines += ["## My Notes", notes, ""]
        return "\n".join(lines)
    md = build_cheatsheet(company or "(no company)", role or "(role)", st.session_state.wiki, st.session_state.projects, st.session_state.jobs, st.session_state.questions, notes)
    st.download_button("⬇️ Download Cheat Sheet (Markdown)", data=md.encode("utf-8"), file_name=f"{(company or 'company').lower().replace(' ','_')}_cheatsheet.md", mime="text/markdown")

# ---------------------------
# Tabs
# ---------------------------

overview_tab, projects_tab, jobs_tab, apply_tab, assistant_tab, q_tab, prep_tab, contacts_tab, qr_tab, saved_tab, help_tab = st.tabs([
    "Overview",
    "Recent Projects",
    "Jobs",
    "Apply",
    "AI Assistant",
    "Recruiter Questions",
    "Prep: Understand a Job Posting",
    "Contacts",
    "My Card (QR)",
    "Saved",
    "Help & Setup",
])

with overview_tab:
    if not company:
        st.info("Enter a company name in the sidebar and click **Research** to get started.")
    else:
        st.subheader(f"{company}")
        cols = st.columns([2, 1])
        with cols[0]:
            if st.session_state.wiki:
                st.markdown("**Company Snapshot**")
                st.write(st.session_state.wiki)
            else:
                st.caption("No snapshot yet.")
        with cols[1]:
            st.markdown("**Quick Actions**")
            st.markdown("- **Make Questions** for a tailored list.")
            st.markdown("- **Find Jobs** to surface ATS/board links and % match.")
            st.markdown("- **Apply** tab to tailor resume and open the application link.")

with projects_tab:
    st.subheader("Recent Projects / Press")
    if st.session_state.projects:
        df = pd.DataFrame(st.session_state.projects)
        st.dataframe(df[["title", "source", "url", "snippet"]], use_container_width=True, hide_index=True)
    else:
        st.caption("No projects yet — click Research in the sidebar.")

with jobs_tab:
    st.subheader("Job Listings (from ATS + boards)")
    if st.session_state.jobs:
        sort_match = st.checkbox("Sort by % match", value=True)
        jobs_view = sorted(st.session_state.jobs, key=lambda x: x.get("match", 0), reverse=True) if sort_match else st.session_state.jobs
        for j in jobs_view:
            with st.container(border=True):
                st.markdown(f"**[{j['title']}]({j['url']})**  ")
                left, right = st.columns([3,1])
                with left:
                    st.caption(j["source"])  
                    st.write(j["snippet"])  
                    st.caption(f"Why: {j.get('why','')}")
                with right:
                    st.markdown(f"<div class='match-pill'>{j.get('match',0)}% match</div>", unsafe_allow_html=True)
                st.button("Save Job", key=f"save_job_{j['url']}", on_click=lambda row=j: st.session_state.saved.append({"type":"job","data":row}))
    else:
        st.caption("No jobs yet — try Find Jobs in the sidebar.")

with apply_tab:
    st.subheader("Apply — Tailor Resume & Open Application")
    if not st.session_state.jobs:
        st.info("Find jobs first to select a posting.")
    else:
        options = [f"{j['title']} — {j.get('match',0)}%" for j in st.session_state.jobs]
        idx = st.selectbox("Choose a job", options=range(len(options)), format_func=lambda i: options[i])
        job = st.session_state.jobs[idx]
        st.link_button("Open job application (new tab)", url=job["url"], help="You will complete submission on the employer's site.")

        st.markdown("---")
        st.markdown("**1) Upload your resume (PDF/DOCX/TXT)**")
        up = st.file_uploader("Resume file", type=["pdf","docx","txt"])
        if up is not None:
            txt, raw, ext = parse_resume(up)
            st.session_state.resume_text = txt
            st.session_state.resume_bytes = raw
            st.session_state.resume_ext = ext
            with st.expander("Preview extracted text", expanded=True):
                st.text_area("Extracted", value=txt, height=240)
        else:
            st.caption("No resume uploaded yet.")

        st.markdown("**2) (Optional) Pull job text**")
        if st.button("Fetch job page text"):
            jt = fetch_text(job["url"]) or job.get("snippet","")
            st.session_state.job_text = jt
            st.success("Fetched page text.")
        job_text = st.session_state.get("job_text", job.get("snippet",""))
        st.text_area("Job description / context", value=job_text, height=160)

        st.markdown("**3) Tailor** (opt‑in)")
        use_ai_tailor = st.checkbox("Use AI for tailoring", value=False, help="If off, a simple keyword‑highlight tailoring is used.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Tailor now", type="primary"):
                if not st.session_state.resume_text.strip():
                    st.warning("Upload a resume first.")
                else:
                    if use_ai_tailor and use_ai and (client := get_ai_client()):
                        out = tailor_resume_ai(client, st.session_state.resume_text, company or "(Company)", role or "(Role)", job_text)
                        if out:
                            st.session_state.tailored = out
                            st.success("Tailored resume ready (AI).")
                        else:
                            st.error("AI tailoring failed. Falling back to keyword highlighting.")
                            kws = extract_keywords(job_text, top_k=12)
                            base = st.session_state.resume_text
                            for k in kws:
                                base = re.sub(fr"\b({re.escape(k)})\b", r"**\1**", base, flags=re.I)
                            st.session_state.tailored = base
                    else:
                        kws = extract_keywords(job_text, top_k=12)
                        base = st.session_state.resume_text
                        for k in kws:
                            base = re.sub(fr"\b({re.escape(k)})\b", r"**\1**", base, flags=re.I)
                        st.session_state.tailored = base
                        st.info("Tailored (heuristic) using job keywords.")
        with col2:
            if st.button("Reset tailored"):
                st.session_state.pop("tailored", None)

        if (tail := st.session_state.get("tailored")):
            st.markdown("---")
            st.markdown("**Tailored Resume (preview)**")
            st.text_area("Tailored", value=tail, height=320)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"resume_{(company or 'company').lower().replace(' ','_')}_{ts}.docx"
            full = os.path.join(RESUME_DIR, fname)
            if save_docx_from_text(tail, full):
                append_resume_index({
                    "timestamp": ts,
                    "company": company,
                    "role": role,
                    "job_url": job["url"],
                    "file": fname,
                })
                with open(full, "rb") as f:
                    st.download_button("⬇️ Download tailored DOCX", data=f.read(), file_name=fname, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                st.success(f"Saved to databank: data/resumes/{fname}")
            else:
                st.download_button("⬇️ Download tailored TXT", data=tail.encode("utf-8"), file_name=fname.replace(".docx",".txt"), mime="text/plain")

with assistant_tab:
    st.subheader("AI Assistant — Describe what you want in a job")
    st.caption("Chat uses OpenAI or any OpenAI‑compatible local server if configured. Otherwise responses will be heuristic.")

    left, right = st.columns([2,1])
    with right:
        st.markdown("**Your preferences**")
        pd = st.session_state.pref_defaults
        pref_role = st.text_input("Desired role/title", value=pd.get("role","Intern"))
        pref_tasks = st.text_area("Tasks / responsibilities", value=pd.get("tasks",""))
        pref_skills = st.text_input("Skills / keywords", value=pd.get("skills",""))
        pref_location = st.text_input("Preferred location", value="Los Angeles, CA")
        pref_remote = st.selectbox("Work mode", ["any","on-site","hybrid","remote"], index=1)
        colx, coly = st.columns(2)
        with colx:
            pref_pay_min = st.number_input("Min pay (annual $)", value=0, step=1000)
        with coly:
            pref_pay_max = st.number_input("Max pay (annual $)", value=0, step=1000)
        pref_companies = st.text_input("Target companies (comma‑separated)", placeholder="optional")
        pref_companies_list = [c.strip() for c in pref_companies.split(",") if c.strip()]
        prefs = {
            "role": pref_role,
            "tasks": pref_tasks,
            "skills": pref_skills,
            "location": pref_location,
            "remote": pref_remote if pref_remote != "any" else "",
            "pay_min": int(pref_pay_min) if pref_pay_min else None,
            "pay_max": int(pref_pay_max) if pref_pay_max else None,
            "companies": pref_companies_list,
        }
        if st.button("Find jobs by preferences", type="primary"):
            with st.spinner("Searching across ATS and boards…"):
                rows: List[Dict[str,Any]] = []
                for q in prefs_to_queries(prefs):
                    for r in ddg_search(q, max_results=6, timelimit="m"):
                        rows.append({"title": r["title"], "url": r["href"], "snippet": clean_snippet(r["body"]), "source": categorize_domain(r["href"])})
                uniq = []
                seen = set()
                for row in rows:
                    u = row["url"]
                    if u in seen: continue
                    seen.add(u)
                    sc, det = score_job_vs_prefs(row, prefs)
                    row.update({"match": sc, "details": det})
                    uniq.append(row)
                st.session_state.pref_jobs = sorted(uniq, key=lambda x: x.get("match",0), reverse=True)
            st.success("Preference‑based results updated.")
    with left:
        st.markdown("**Chat**")
        chat_container = st.container(height=380, border=True)
        for m in st.session_state.assistant_history:
            role_lbl = "🧑‍🎓 You" if m["role"] == "user" else "🤖 Assistant"
            chat_container.markdown(f"**{role_lbl}:** {m['content']}")
        user_msg = st.chat_input("Describe the kind of role you want, ask questions, etc.")
        if user_msg:
            st.session_state.assistant_history.append({"role":"user","content":user_msg})
            reply = None
            if use_ai and (client := get_ai_client()):
                try:
                    mdl = st.secrets.get("openai_model") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
                    sys = (
                        "You are a helpful career search copilot. Ask clarifying questions if the request is vague; "
                        "propose search strategies and concrete keywords for ATS sites; keep answers concise."
                    )
                    out = client.chat.completions.create(model=mdl, temperature=0.3, messages=[
                        {"role":"system","content":sys},
                        *st.session_state.assistant_history,
                    ])
                    reply = out.choices[0].message.content
                except Exception:
                    reply = "I couldn't reach the AI service. Try entering details in the preference form on the right, then click 'Find jobs by preferences'."
            else:
                reply = "Got it — fill out the preferences on the right and hit 'Find jobs by preferences'. You can also add keywords like 'internship', 'junior', or a certification to tighten results."
            st.session_state.assistant_history.append({"role":"assistant","content":reply})
            st.rerun()

    st.markdown("---")
    st.markdown("**Results matched to your preferences**")
    if st.session_state.pref_jobs:
        for j in st.session_state.pref_jobs[:20]:
            with st.container(border=True):
                st.markdown(f"**[{j['title']}]({j['url']})**")
                left, right = st.columns([3,1])
                with left:
                    st.caption(j["source"])  
                    st.write(j["snippet"])  
                    if j.get("details"):
                        st.markdown("Matched: " + ", ".join([f"`{d}`" for d in j["details"]]))
                with right:
                    st.markdown(f"<div class='match-pill'>{j.get('match',0)}% match</div>", unsafe_allow_html=True)

with q_tab:
    st.subheader("Questions to Ask Recruiters")
    if st.session_state.questions:
        for bucket, qs in st.session_state.questions.items():
            with st.expander(bucket, expanded=True):
                for q in qs:
                    st.markdown(f"- {q}")
    else:
        st.caption("Click Make Questions in the sidebar to generate a tailored list.")

with prep_tab:
    st.subheader("Paste a Job Description to Prep")
    jd = st.text_area("Job description (paste text here)", height=220)
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Analyze Posting", type="primary"):
            if jd.strip():
                kws = extract_keywords(jd, top_k=14)
                st.markdown("**Key topics / skills detected:**")
                st.write(", ".join(kws) if kws else "(none)")
                extra = smart_questions(company, role, kws, job_desc=jd, tool_examples=st.session_state.get("preset_tools", []))
                st.session_state.questions = extra
                st.success("Updated questions based on the posting.")
            else:
                st.warning("Please paste a job description first.")
    with col2:
        if st.button("Add to My Notes"):
            st.session_state.saved.append({"type":"notes","data":{"company":company, "role":role, "notes":jd}})
            st.toast("Saved to your list.")

# ---------------------------
# Contacts finder
# ---------------------------

def find_contacts(company: str, role: str = "") -> List[Dict[str, str]]:
    if not company:
        return []
    queries = [
        f'site:linkedin.com/in ("Recruiter" OR "Talent Acquisition" OR "University Recruiter") "{company}"',
        f'site:linkedin.com/company {company} people recruiter',
        f'{company} university recruiting email',
        f'{company} HR contact careers',
    ]
    rows: List[Dict[str, str]] = []
    for q in queries:
        for r in ddg_search(q, max_results=6, timelimit="y"):
            rows.append({"title": r.get("title") or "", "url": r.get("href") or "", "snippet": clean_snippet(r.get("body")), "source": categorize_domain(r.get("href") or "")})
    seen = set(); uniq = []
    for row in rows:
        u = row["url"]
        if u in seen: continue
        seen.add(u); uniq.append(row)
    return uniq

with contacts_tab:
    st.subheader("Find Recruiters / HR / University Contacts")
    if st.button("Search Contacts", type="primary"):
        hits = find_contacts(company, role)
        if hits:
            for h in hits:
                with st.container(border=True):
                    st.markdown(f"**[{h['title']}]({h['url']})**")
                    st.caption(h["source"])  
                    st.write(h["snippet"])  
                    st.button("Save Contact", key=f"save_contact_{h['url']}", on_click=lambda row=h: st.session_state.contacts.append(row))
        else:
            st.caption("No contacts found yet — try a different company spelling or add 'university recruiting'.")

    st.divider()
    st.markdown("**Import Contacts (CSV)**")
    up = st.file_uploader("Upload CSV with columns: name,email,title,company,notes", type=["csv"], key="contacts_csv")
    if up is not None:
        try:
            dfc = pd.read_csv(up)
            st.dataframe(dfc, use_container_width=True, hide_index=True)
            if st.button("Add to Saved Contacts"):
                for _, r in dfc.iterrows():
                    st.session_state.contacts.append({k: str(r.get(k, "")) for k in ["name","email","title","company","notes"]})
                st.success("Contacts added.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if st.session_state.contacts:
        st.divider()
        st.markdown("**My Contacts**")
        st.dataframe(pd.DataFrame(st.session_state.contacts), use_container_width=True)

# ---------------------------
# QR Card / vCard
# ---------------------------

def build_vcard(name: str, title: str, email: str, phone: str, org: str, linkedin: str, website: str) -> str:
    lines = [
        "BEGIN:VCARD","VERSION:3.0",f"FN:{name}",
        f"TITLE:{title}" if title else "",
        f"ORG:{org}" if org else "",
        f"EMAIL;TYPE=INTERNET:{email}" if email else "",
        f"TEL;TYPE=CELL:{phone}" if phone else "",
        f"URL:{linkedin}" if linkedin else "",
        f"URL:{website}" if website else "",
        "END:VCARD",
    ]
    return "\n".join([l for l in lines if l])

with qr_tab:
    st.subheader("Create a QR Card for Networking")
    vcf_up = st.file_uploader("Optional: import vCard (.vcf) to prefill", type=["vcf"], key="vcf")
    name = st.text_input("Full name")
    title_i = st.text_input("Title (e.g., Student)")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    org = st.text_input("Organization / School")
    linkedin = st.text_input("LinkedIn URL")
    website = st.text_input("Personal site / portfolio URL")

    if vcf_up is not None:
        try:
            vtxt = vcf_up.read().decode("utf-8", errors="ignore")
            def pick(tag):
                for line in vtxt.splitlines():
                    if line.startswith(tag+":"): return line.split(":",1)[1].strip()
                return ""
            name = name or pick("FN")
            title_i = title_i or pick("TITLE")
            org = org or pick("ORG")
            if not linkedin:
                for line in vtxt.splitlines():
                    if line.startswith("URL:") and "linkedin" in line.lower():
                        linkedin = line.split(":",1)[1].strip(); break
            st.success("vCard fields imported. Adjust as needed.")
        except Exception:
            st.warning("Could not parse vCard. Proceeding with manual fields.")

    mode = st.radio("QR content", ["vCard (contacts app)", "Just my LinkedIn URL"], horizontal=True)
    col_l, col_r = st.columns([1,1])
    with col_l:
        if st.button("Generate QR", type="primary"):
            if segno is None:
                st.error("QR feature requires 'segno'. Make sure it's in requirements.txt and installed.")
            else:
                buf = io.BytesIO()
                if mode.startswith("vCard"):
                    vcf = build_vcard(name, title_i, email, phone, org, linkedin, website)
                    qr = segno.make(vcf)
                else:
                    url = linkedin or website
                    if not url:
                        st.warning("Provide your LinkedIn or a URL.")
                        qr = None
                    else:
                        qr = segno.make(url)
                if qr is not None:
                    qr.save(buf, kind="png", scale=6)
                    st.image(buf.getvalue(), caption="Scan me!", use_column_width=False)
                    st.download_button("⬇️ Download QR (PNG)", data=buf.getvalue(), file_name="my_qr.png", mime="image/png")
    with col_r:
        if st.button("Download vCard (.vcf)"):
            vcf = build_vcard(name, title_i, email, phone, org, linkedin, website)
            st.download_button("⬇️ Save vCard", data=vcf.encode("utf-8"), file_name="my_contact.vcf", mime="text/vcard")

# ---------------------------
# Resume helpers
# ---------------------------

def parse_resume(upload) -> Tuple[str, bytes, str]:
    raw = upload.read()
    name = upload.name.lower()
    ext = ".txt"
    if name.endswith(".pdf"):
        ext = ".pdf"
        if PdfReader is None:
            return "", raw, ext
        try:
            reader = PdfReader(io.BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages), raw, ext
        except Exception:
            return "", raw, ext
    if name.endswith(".docx"):
        ext = ".docx"
        if Document is None:
            return "", raw, ext
        try:
            doc = Document(io.BytesIO(raw))
            txt = "\n".join([p.text for p in doc.paragraphs])
            return txt, raw, ext
        except Exception:
            return "", raw, ext
    try:
        return raw.decode("utf-8", errors="ignore"), raw, ".txt"
    except Exception:
        return "", raw, ".bin"


def save_docx_from_text(text: str, path: str):
    if Document is None:
        return False
    try:
        doc = Document()
        for line in text.splitlines():
            if line.strip().startswith("# "):
                doc.add_heading(line.strip("# ").strip(), level=1)
            elif line.strip().startswith("## "):
                doc.add_heading(line.strip("# ").strip(), level=2)
            else:
                doc.add_paragraph(line)
        doc.save(path)
        return True
    except Exception:
        return False


def append_resume_index(meta: Dict[str, Any]):
    cols = ["timestamp","company","role","job_url","file"]
    row = {k: meta.get(k, "") for k in cols}
    df = pd.DataFrame([row])
    if os.path.exists(INDEX_CSV):
        base = pd.read_csv(INDEX_CSV)
        base = pd.concat([base, df], ignore_index=True)
        base.to_csv(INDEX_CSV, index=False)
    else:
        df.to_csv(INDEX_CSV, index=False)

with saved_tab:
    st.subheader("My Saved Items")
    if not st.session_state.saved and not os.path.exists(INDEX_CSV):
        st.caption("Nothing saved yet.")
    else:
        if st.session_state.saved:
            st.markdown("### Projects & Jobs")
            for i, item in enumerate(st.session_state.saved):
                if item["type"] == "project":
                    p = item["data"]
                    with st.container(border=True):
                        st.markdown(f"**[Project] [{p['title']}]({p['url']})**")
                        st.caption(p["source"])  
                        st.write(p["snippet"])  
                elif item["type"] == "job":
                    j = item["data"]
                    with st.container(border=True):
                        st.markdown(f"**[Job] [{j['title']}]({j['url']})** — {j.get('match',0)}% match")
                        st.caption(j["source"])  
                        st.write(j["snippet"])  
                elif item["type"] == "notes":
                    n = item["data"]
                    with st.container(border=True):
                        st.markdown(f"**[Notes] {n.get('company') or ''} — {n.get('role') or ''}**")
                        st.write(n.get("notes", ""))
        if os.path.exists(INDEX_CSV):
            st.markdown("### Tailored Resumes Databank")
            df = pd.read_csv(INDEX_CSV)
            st.dataframe(df, use_container_width=True, hide_index=True)
            for _, r in df.tail(10).iterrows():
                fp = os.path.join(RESUME_DIR, str(r["file"]))
                if os.path.exists(fp):
                    with open(fp, "rb") as f:
                        st.download_button(f"⬇️ {r['file']}", data=f.read(), file_name=str(r["file"]))

with help_tab:
    st.subheader("Setup & Notes")
    st.markdown(
        """
        **Local setup**
        ```bash
        python -m venv .venv && source .venv/bin/activate
        pip install -r requirements.txt
        streamlit run app.py
        ```
        **Deploy on Streamlit Community Cloud**: push to GitHub, deploy from repo, add secrets if using cloud AI.

        **ATS applications**: most systems require submitting on their domains. This app prepares and opens the link; you finalize the submission there.

        **Databank**: tailored resumes are saved under `data/resumes/` with an `index.csv`. On Streamlit Cloud, storage may be ephemeral; download important files.

        **AI**: works with OpenAI OR any OpenAI‑compatible local server (e.g., Ollama). Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` env vars.
        """
    )

st.markdown("""
<div class="muted small" style="margin-top:1rem;">
  Built with ❤️ using Streamlit + DuckDuckGo + Wikipedia + YAKE + Segno. Optional OpenAI/compatible LLM support.
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    pass
