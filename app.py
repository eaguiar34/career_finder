"""
Career Fair Companion — Streamlit App (Enhanced Jobs Search + Sidebar Bulk)
- Smarter company lookup + ATS awareness (Workday/Greenhouse/Lever/Taleo/SuccessFactors/Ashby/Workable/Jobvite/SmartRecruiters/iCIMS/Eightfold/Recruitee/ADP/Breezy)
- Jobs tab now has its own search bar (so results appear when you type there)
- Sidebar adds a **Bulk** tab to paste/upload URLs and scan careers pages
- Bulk Companies main tab remains for viewing + CSV export
"""
from __future__ import annotations
import os, io, re, json, difflib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st

# ---------- Optional libs ----------
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
    from openai import OpenAI
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
# BeautifulSoup optional
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore
import requests

# ---------- Paths ----------
DATA_DIR = "data"
RESUME_DIR = os.path.join(DATA_DIR, "resumes")
BULK_FILE = os.path.join(DATA_DIR, "companies.txt")
os.makedirs(RESUME_DIR, exist_ok=True)
INDEX_CSV = os.path.join(RESUME_DIR, "index.csv")

# discover preset .txt files for bulk lists
PRESET_TXT_FILES: List[str] = []
try:
    if os.path.isdir(DATA_DIR):
        PRESET_TXT_FILES = [os.path.join(DATA_DIR, f) for f in sorted(os.listdir(DATA_DIR)) if f.endswith('.txt')]
except Exception:
    PRESET_TXT_FILES = [BULK_FILE]

# ---------- Theme values ----------
DEFAULT_PRIMARY = "#7C3AED"  # violet-600
DEFAULT_ACCENT  = "#22D3EE"  # cyan-400
DEFAULT_BG = "#0B1020"
DEFAULT_CARD_BG = "#0f172a"

SCHOOL_THEMES = {
    "Cal State LA": {"primary": "#000000", "accent": "#FFC72C"},
    "UCLA":         {"primary": "#2774AE", "accent": "#FFD100"},
    "USC":          {"primary": "#990000", "accent": "#FFCC00"},
    "UC Berkeley":  {"primary": "#003262", "accent": "#FDB515"},
    "MIT":          {"primary": "#A31F34", "accent": "#8A8B8C"},
}

# Prefer the corporate entity when names are ambiguous
COMPANY_FIXUPS = {
    "jacobs": "Jacobs Solutions",
    "jacobs engineering": "Jacobs Solutions",
    "pwc": "PwC",
    "ernst & young": "EY",
}

def canonical_company(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return COMPANY_FIXUPS.get(key, name)

# Known ATS hosts (expandable)
ATS_HOSTS = [
    "lever.co","greenhouse.io","myworkdayjobs.com","smartrecruiters.com","icims.com",
    "taleo.net","successfactors.com","ashbyhq.com","workable.com","jobvite.com",
    "eightfold.ai","recruitee.com","adp.com","workforcenow.adp.com","breezy.hr",
]

# ---------- Majors (subset for brevity) ----------
MAJOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "(Custom)": {"role": "Intern","interests": ["projects","research","leadership"],"tasks": "","skills": "","tools": ["Excel","Python"]},
    "Civil Engineering": {"role": "Civil Engineering Intern","interests": ["transportation","water","sustainability","Python"],"tasks": "transportation design, hydrology/hydraulics, site visits, CAD/BIM","skills": "Civil 3D, AutoCAD, HEC-RAS, BIM, GIS","tools": ["Civil 3D","AutoCAD","HEC-RAS","MicroStation","BIM","GIS"]},
    "Computer Science / Software": {"role": "Software Engineer Intern","interests": ["backend","web","AI","data"],"tasks": "APIs, web apps, data pipelines","skills": "Python, JavaScript, React, Node, SQL","tools": ["Python","JS/TS","React","Node","SQL"]},
    "Data / Analytics": {"role": "Data Analyst Intern","interests": ["analytics","dashboards","SQL"],"tasks": "dashboards, ETL, experimentation","skills": "SQL, Python, Tableau, Power BI","tools": ["SQL","Python","Tableau","Power BI","R"]},
    "Nursing": {"role": "Student Nurse / Nursing Intern","interests": ["clinical","patient care","EMR"],"tasks": "vitals, charting, patient education","skills": "BLS, EMR/EHR","tools": ["Epic","Cerner","MEDITECH"]},
    # ... (other majors omitted to keep this file compact)
}

# ---------- Page ----------
st.set_page_config(page_title="Career Fair Companion", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
      html, body, [class*="css"] {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
      .hero {{ background: radial-gradient(1200px 400px at 20% -10%, {DEFAULT_ACCENT}33, transparent),
                             radial-gradient(800px 300px at 80% -20%, {DEFAULT_PRIMARY}33, transparent);
               padding: 1rem 1.25rem; border-radius: 18px; border: 1px solid #1f2937; }}
      .app-title {{ background: linear-gradient(90deg, {DEFAULT_PRIMARY}, {DEFAULT_ACCENT});
                    -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 800; }}
      .card {{ border-radius: 16px; padding: 1rem 1.25rem; border: 1px solid #1f2937; background: {DEFAULT_CARD_BG}; box-shadow: 0 8px 24px rgba(0,0,0,0.25); }}
      .muted {{ color: #94a3b8; }} .small {{ font-size: 0.9rem; }}
      .match-pill {{ padding: 2px 8px; border-radius: 999px; background: #0ea5e933; border:1px solid #22d3ee; font-weight:600; }}
      .btn-primary button {{ background: {DEFAULT_PRIMARY} !important; border-color: {DEFAULT_PRIMARY} !important; }}
      .btn-accent button {{ background: {DEFAULT_ACCENT} !important; border-color: {DEFAULT_ACCENT} !important; color: #0b1020 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

def apply_theme(primary: str, accent: str):
    st.markdown(
        f"""
        <style>
          .app-title {{ background: linear-gradient(90deg, {primary}, {accent}); -webkit-background-clip: text; background-clip: text; color: transparent; }}
          .btn-primary button {{ background: {primary} !important; border-color: {primary} !important; }}
          .btn-accent button {{ background: {accent} !important; border-color: {accent} !important; color: #0b1020 !important; }}
          .match-pill {{ background: {accent}22; border-color: {accent}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------- AI helpers ----------

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

# ---------- Search helpers ----------

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
        query = canonical_company(company)
        candidates = [f"{query} (company)", f"{query} (corporation)", query]
        page_title = None
        for q in candidates:
            res = wikipedia.search(q, results=1)
            if res:
                page_title = res[0]
                break
        if not page_title:
            return None
        page = wikipedia.page(page_title, auto_suggest=False)
        if any(w in page.title.lower() for w in ["(patriarch)", "given name", "biblical"]):
            return None
        return clean_snippet(page.summary, 600)
    except Exception:
        return None


def fetch_text(url: str, limit: int = 6000) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (CareerFairComp/1.0)"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return ""
        html = r.text
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            txt = soup.get_text(" ", strip=True)
        else:
            txt = re.sub(r"<[^>]+>", " ", html)
            txt = re.sub(r"\s+", " ", txt).strip()
        return clean_snippet(txt, length=limit)
    except Exception:
        return ""


def categorize_domain(url: str) -> str:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0]
    except Exception:
        return "Other"
    if any(k in host for k in ATS_HOSTS):
        return "ATS"
    if any(k in host for k in ["indeed.com", "linkedin.com/jobs", "glassdoor.com/job"]):
        return "Job Board"
    if any(k in host for k in ["newsroom", "press", "media", "blog."]):
        return "Press/Blog"
    return "Other"


def find_official_site(company: str) -> Optional[str]:
    for r in ddg_search(f"{company} official site", max_results=4, timelimit="y"):
        url = r.get("href") or ""
        if not url:
            continue
        host = re.sub(r"^https?://", "", url).split("/")[0]
        if not any(s in host for s in ["wikipedia.org", "linkedin.com", "twitter.com"]):
            return url
    return None

# try to guess careers pages quickly (ddg + common paths)
COMMON_CAREERS_PATHS = ["/careers", "/careers/jobs", "/jobs", "/join-us", "/work-with-us", "/careers/search", "/en/careers"]

def find_careers_candidates(company: str) -> List[str]:
    cand = []
    site = find_official_site(company)
    if site:
        base = re.sub(r"/$", "", site)
        cand += [base + p for p in COMMON_CAREERS_PATHS]
    # search results
    for r in ddg_search(f"{company} careers", max_results=6, timelimit="y"):
        u = r.get("href");
        if u and u not in cand:
            cand.append(u)
    return cand[:10]

# core: job search aggregator (smarter)

def job_search(company: str, role: str = "", location: str = "", max_results: int = 24) -> List[Dict[str, Any]]:
    brand = canonical_company(company)

    # Seeds across ATS
    seeds = [
        "site:lever.co", "site:greenhouse.io", "site:myworkdayjobs.com", "site:smartrecruiters.com", "site:icims.com",
        "site:taleo.net", "site:successfactors.com", "site:ashbyhq.com", "site:workable.com", "site:jobvite.com",
        "site:eightfold.ai", "site:recruitee.com", "site:workforcenow.adp.com", "site:adp.com", "site:breezy.hr",
    ]
    terms = f"{brand} {role}".strip()
    if not role:
        # encourage intern/new grad queries when role is empty
        terms += " intern OR internship OR graduate"
    if location:
        terms += f" {location}"

    rows: List[Dict[str, Any]] = []
    for s in seeds:
        q = f"{s} {terms}"
        for r in ddg_search(q, max_results=6, timelimit="m"):
            rows.append({"title": r["title"], "url": r["href"], "snippet": clean_snippet(r["body"]), "source": categorize_domain(r["href"])})

    # Also search company careers site/domain directly
    for cand in find_careers_candidates(brand):
        try:
            host = re.sub(r"^https?://", "", cand).split("/")[0]
            q = f"site:{host} {role or 'intern OR internship OR graduate'}"
            if location:
                q += f" {location}"
            for r in ddg_search(q, max_results=6, timelimit="m"):
                rows.append({"title": r["title"], "url": r["href"], "snippet": clean_snippet(r["body"]), "source": categorize_domain(r["href"])})
        except Exception:
            continue

    # De-dup and cap
    seen = set(); uniq = []
    for row in rows:
        u = row["url"]
        if u in seen: continue
        seen.add(u); uniq.append(row)
    return uniq[:max_results]

# ---- Preference search ----

def prefs_to_queries(p: Dict[str, Any]) -> List[str]:
    role = p.get("role","")
    tasks = p.get("tasks","")
    skills = p.get("skills","")
    location = p.get("location","")
    companies = [c for c in (p.get("companies") or []) if c]
    base_terms = f"{role} {skills} {tasks}".strip()
    if not role:
        base_terms += " intern OR internship OR graduate"
    seeds = [
        "site:lever.co","site:greenhouse.io","site:myworkdayjobs.com","site:smartrecruiters.com","site:icims.com",
        "site:taleo.net","site:successfactors.com","site:ashbyhq.com","site:workable.com","site:jobvite.com",
        "site:eightfold.ai","site:recruitee.com","site:workforcenow.adp.com","site:adp.com","site:breezy.hr",
        "careers"
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

# utility scoring

def _norm_tokens(s: str) -> List[str]:
    s = s.lower(); s = re.sub(r"[^a-z0-9\+\-\s]", " ", s)
    return [t for t in s.split() if len(t) > 2]

def score_job(job: Dict[str, Any], role: str, interests: List[str], location: str) -> Tuple[int, str]:
    title = job.get("title") or ""; snip = job.get("snippet") or ""; url = job.get("url") or ""
    body = snip or fetch_text(url)
    role_ratio = difflib.SequenceMatcher(a=(role or '').lower(), b=(title + " " + body).lower()).ratio() if role else 0.0
    body_tokens = set(_norm_tokens(title + " " + body))
    interests_norm = [i.lower().strip() for i in interests if i.strip()]
    overlap = sum(1 for i in interests_norm if any(tok in body_tokens for tok in _norm_tokens(i)))
    interest_ratio = (overlap / max(1, len(interests_norm))) if interests_norm else 0.0
    loc_hit = 1 if (location and location.lower() in (title + " " + body + " " + url).lower()) else 0
    src = job.get("source", "Other"); source_bonus = {"ATS": 1.0, "Job Board": 0.6, "Other": 0.4}.get(src, 0.4)
    score = 40 * role_ratio + 40 * interest_ratio + 10 * loc_hit + 10 * source_bonus
    score = max(0, min(100, int(round(score))))
    reason = f"role~{int(role_ratio*100)}%, interests {overlap}/{len(interests_norm) or 1}, location {'✓' if loc_hit else '—'}, source {src}"
    return score, reason

# ---------- Keywords ----------

def extract_keywords(text: str, top_k: int = 12) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if yake:
        try:
            kw = yake.KeywordExtractor(top=top_k)
            return [k for k, _ in kw.extract_keywords(t)][:top_k]
        except Exception:
            pass
    # fallback: frequent words (very rough)
    toks = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-\+]{2,}", t.lower()) if w not in {"and","the","with","for","from","that","this","will","your","you"}]
    freq = {}
    for x in toks: freq[x] = freq.get(x, 0) + 1
    return [k for k,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)][:top_k]

# ---------- Resume helpers ----------

def parse_resume(upload) -> Tuple[str, bytes, str]:
    raw = upload.read(); name = upload.name.lower(); ext = ".txt"
    if name.endswith(".pdf"):
        ext = ".pdf"
        if PdfReader is None: return "", raw, ext
        try:
            reader = PdfReader(io.BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages), raw, ext
        except Exception: return "", raw, ext
    if name.endswith(".docx"):
        ext = ".docx"
        if Document is None: return "", raw, ext
        try:
            doc = Document(io.BytesIO(raw))
            txt = "\n".join([p.text for p in doc.paragraphs])
            return txt, raw, ext
        except Exception: return "", raw, ext
    try:
        return raw.decode("utf-8", errors="ignore"), raw, ".txt"
    except Exception:
        return "", raw, ".bin"


def save_docx_from_text(text: str, path: str):
    if Document is None: return False
    try:
        doc = Document()
        for line in text.splitlines():
            if line.strip().startswith("# "): doc.add_heading(line.strip("# ").strip(), level=1)
            elif line.strip().startswith("## "): doc.add_heading(line.strip("# ").strip(), level=2)
            else: doc.add_paragraph(line)
        doc.save(path); return True
    except Exception: return False


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

# ---------- Session defaults ----------
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
    "bulk_urls": [],
    "bulk_results": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Sidebar (now with Bulk) ----------
with st.sidebar:
    tabs = st.tabs(["Inputs", "Settings", "Bulk"])

    with tabs[0]:
        st.subheader("🎓 Major Preset + Research Inputs")
        major = st.selectbox("Major (preset)", options=list(MAJOR_PRESETS.keys()), index=list(MAJOR_PRESETS.keys()).index("Civil Engineering"))
        if st.button("Apply Preset"):
            p = MAJOR_PRESETS.get(major, MAJOR_PRESETS["(Custom)"])
            st.session_state.role_input = p.get("role", "Intern")
            st.session_state.interests_input = ", ".join(p.get("interests", []))
            st.session_state.preset_tools = p.get("tools", [])
            st.session_state.pref_defaults = {
                "role": p.get("role", "Intern"),
                "tasks": p.get("tasks", ""),
                "skills": p.get("skills", ""),
            }
        company = st.text_input("Company name", placeholder="e.g., Jacobs, AECOM, Google")
        role = st.text_input("Target role/title", value=MAJOR_PRESETS["Civil Engineering"]["role"], key="role_sidebar")
        location = st.text_input("Location filter (optional)", placeholder="e.g., Los Angeles, CA")
        interests_str = st.text_input("Interests / keywords (comma‑separated)", value=", ".join(MAJOR_PRESETS["Civil Engineering"]["interests"]))
        interests = [s.strip() for s in interests_str.split(",") if s.strip()]

        colA, colB = st.columns(2)
        with colA: go_research = st.button("Research", type="primary")
        with colB: go_questions = st.button("Make Questions")
        colC, colD = st.columns(2)
        with colC: go_jobs = st.button("Find Jobs")
        with colD: go_export = st.button("Export Sheet")

    with tabs[1]:
        st.subheader("⚙️ Settings")
        school = st.selectbox("School theme", options=["(Default)"] + list(SCHOOL_THEMES.keys()), index=0)
        if school != "(Default)":
            t = SCHOOL_THEMES.get(school, {})
            primary = t.get("primary", DEFAULT_PRIMARY); accent = t.get("accent", DEFAULT_ACCENT)
        else:
            primary = DEFAULT_PRIMARY; accent = DEFAULT_ACCENT
        custom_primary = st.color_picker("Primary", value=primary)
        custom_accent  = st.color_picker("Accent", value=accent)
        apply_theme(custom_primary or primary, custom_accent or accent)
        st.markdown("---")
        st.subheader("🤖 AI Settings")
        use_ai = st.toggle("Use AI (OpenAI or compatible)", value=False)
        st.text_input("Model (optional)", value=os.environ.get("OPENAI_MODEL", ""))
        st.text_input("Base URL (optional)", value=os.environ.get("OPENAI_BASE_URL", ""))

    with tabs[2]:
        st.subheader("📦 Bulk Companies (quick)")
        st.caption("Paste URLs or load a preset list; scan careers pages. Full table lives on the main tab.")
        txt_sb = st.text_area("Paste URLs (one per line)", height=100)
        preset_labels = [f"data/{os.path.basename(p)}" for p in PRESET_TXT_FILES] or ["(none)"]
        preset_choice = st.selectbox("Load preset list", preset_labels, index=0)
        if st.button("Load preset in sidebar") and PRESET_TXT_FILES:
            chosen_fp = PRESET_TXT_FILES[preset_labels.index(preset_choice)]
            try:
                with open(chosen_fp, "r", encoding="utf-8") as f:
                    st.session_state.bulk_urls = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
                st.success(f"Loaded {len(st.session_state.bulk_urls)} URLs from {os.path.basename(chosen_fp)}")
            except Exception as e:
                st.error(f"Could not load preset: {e}")
        if st.button("Scan (sidebar)"):
            urls = []
            if txt_sb.strip():
                urls += [u.strip() for u in txt_sb.splitlines() if u.strip() and not u.strip().startswith('#')]
            if st.session_state.get("bulk_urls"):
                urls += [u for u in st.session_state.bulk_urls if u]
            urls = list(dict.fromkeys(urls))[:50]
            if not urls:
                st.warning("No URLs provided. Paste or load a preset.")
            else:
                items = []
                for u in urls:
                    dom = extract_domain(u)
                    hits = ddg_find_careers(dom)
                    if not hits:
                        items.append({"domain": dom, "careers_url": "(none)", "title": "", "snippet": ""})
                    else:
                        for h in hits[:2]:
                            items.append({"domain": dom, "careers_url": h["url"], "title": h["title"], "snippet": h["snippet"]})
                st.session_state.bulk_results = items
                st.success(f"Scanned {len(urls)} domains. See 'Bulk Companies' tab for details.")

# ---------- Header ----------
st.markdown(
    f"""
    <div class="hero">
      <div style="display:flex; align-items:center; gap:12px;">
        <div style="font-size:2.3rem;">🎓</div>
        <div>
          <div class="app-title" style="font-size:2rem;">Career Fair Companion</div>
          <div class="muted small">Research companies fast. Ask sharper questions. Find roles. Tailor resumes. Swap QR cards.</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Research / Questions / Jobs triggers from sidebar ----------
if 'go_research' not in globals():
    go_research = go_questions = go_jobs = go_export = False  # guards for editors

if DDGS is None:
    st.warning("Search library (duckduckgo-search) is unavailable. Ensure it's in requirements.txt and installed.")

if 'company' not in locals():
    company, role, location, interests = "", "", "", []

if go_research and company:
    with st.spinner("Researching…"):
        cn = canonical_company(company)
        st.session_state.wiki = wiki_summary(cn)
        if not st.session_state.wiki:
            site = find_official_site(cn)
            if site:
                st.session_state.wiki = fetch_text(site, limit=500)
        st.session_state.projects = []  # omitted here for brevity
    st.success("Company research updated.")

if go_questions:
    st.session_state.questions = {
        "Role & Impact": [
            f"How does {role or 'this role'} define success in the first 90 days?",
            "Which projects are top priority this quarter?",
            "What metrics matter most for early‑career hires?",
        ],
        "Team & Mentorship": [
            "What does onboarding look like?",
            "How often are 1:1s and feedback cycles?",
            "Any examples of cross‑team collaboration for interns/new grads?",
        ],
        "Projects & Methods": [
            "Which tools are standard here?",
            "How do you scope projects on tight timelines?",
            f"For someone into {', '.join(interests[:5]) or 'the core skills for this role'}, what would you recommend studying?",
        ],
        "Culture & Growth": [
            "How do interns find mentors and stretch projects?",
            "Are there rotations or formal training?",
            "How do you support learning (certs, conferences)?",
        ],
        "Recruiter‑Specific": [
            "Which teams should I watch based on my interests?",
            "Common pitfalls you see in applications?",
            "What should I include in a follow‑up email after the fair?",
        ],
    }
    st.success("Questions ready.")

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

# ---------- Tabs ----------
overview_tab, jobs_tab, bulk_tab, apply_tab, saved_tab = st.tabs(["Overview","Jobs","Bulk Companies","Apply","Saved"])  # slimmer for this edition

with overview_tab:
    st.subheader("Welcome")
    st.write("Use the sidebar to enter a company and role, or search from the Jobs tab.")

with jobs_tab:
    st.subheader("Job Listings (ATS + official domains)")
    # NEW: in-tab search bar
    c1, c2, c3 = st.columns([1.2,1,1])
    company_in = c1.text_input("Company", value=locals().get('company', ''), placeholder="e.g., Jacobs, AECOM, Google", key="jobs_company")
    role_in = c2.text_input("Role keywords", value=locals().get('role',''), placeholder="e.g., Civil Engineering Intern", key="jobs_role")
    loc_in = c3.text_input("Location (optional)", value=locals().get('location',''), placeholder="City, State", key="jobs_loc")
    if st.button("Search Jobs", type="primary"):
        if not company_in.strip():
            st.warning("Please enter a company.")
        else:
            with st.spinner("Searching across ATS and the company's domains…"):
                rows = job_search(company_in, role=role_in, location=loc_in)
                scored = []
                interests = []
                for j in rows:
                    pct, why = score_job(j, role_in, interests, loc_in)
                    j2 = {**j, "match": pct, "why": why}
                    scored.append(j2)
                st.session_state.jobs = sorted(scored, key=lambda x: x.get("match", 0), reverse=True)
    st.divider()
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
    else:
        st.info("No results yet. Enter a company above and click **Search Jobs**.")

# -------- Bulk Companies tab (full table) --------

def extract_domain(url: str) -> str:
    try:
        url = url.strip()
        if not re.match(r"^https?://", url):
            url = "https://" + url
        host = re.sub(r"^https?://", "", url).split("/")[0]
        return host.lower()
    except Exception:
        return url


def ddg_find_careers(domain: str) -> List[Dict[str, str]]:
    queries = [
        f"site:{domain} careers",
        f"site:{domain} jobs",
        f"site:{domain} internships",
        f"site:{domain} early careers",
    ]
    hits = []
    for q in queries:
        for r in ddg_search(q, max_results=5, timelimit="y"):
            hits.append({"title": r.get("title"), "url": r.get("href"), "snippet": clean_snippet(r.get("body"))})
    seen = set(); uniq = []
    for h in hits:
        u = h["url"]
        if u in seen: continue
        seen.add(u); uniq.append(h)
    return uniq

with bulk_tab:
    st.subheader("Bulk Companies — scan careers pages")
    st.caption("Paste/upload URLs, or load a preset list from data/*.txt. Use the sidebar's Bulk tab for a quick mini-scan.")
    txt = st.text_area("Paste URLs (one per line)", height=120)
    up = st.file_uploader("Or upload .txt / .csv (column: url)", type=["txt","csv"])
    preset_labels = [f"data/{os.path.basename(p)}" for p in PRESET_TXT_FILES]
    preset_choice = st.selectbox("Or load a preset list", preset_labels, index=0) if preset_labels else None
    if st.button("Load preset") and preset_choice:
        chosen_fp = PRESET_TXT_FILES[preset_labels.index(preset_choice)]
        try:
            with open(chosen_fp, "r", encoding="utf-8") as f:
                st.session_state.bulk_urls = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            st.success(f"Loaded {len(st.session_state.bulk_urls)} URLs from {os.path.basename(chosen_fp)}")
        except Exception as e:
            st.error(f"Could not load preset: {e}")

    col1, col2 = st.columns(2)
    urls: List[str] = []
    if txt.strip():
        urls += [u.strip() for u in txt.splitlines() if u.strip() and not u.strip().startswith('#')]
    if st.session_state.get("bulk_urls"):
        urls += [u for u in st.session_state.bulk_urls if u]
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                dfu = pd.read_csv(up)
                if "url" in dfu.columns:
                    urls += [str(x) for x in dfu["url"].dropna().tolist()]
                else:
                    urls += [str(x) for x in dfu.iloc[:,0].dropna().tolist()]
            else:
                raw = up.read().decode("utf-8", errors="ignore")
                urls += [u.strip() for u in raw.splitlines() if u.strip() and not u.strip().startswith('#')]
        except Exception as e:
            st.error(f"Could not parse file: {e}")
    if not urls and os.path.exists(BULK_FILE):
        try:
            with open(BULK_FILE, "r", encoding="utf-8") as f:
                urls = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            st.info("Loaded companies from data/companies.txt")
        except Exception:
            pass
    urls = list(dict.fromkeys(urls))

    with col1:
        if st.button("Scan for Careers", type="primary"):
            items = []
            for u in urls[:80]:
                dom = extract_domain(u)
                hits = ddg_find_careers(dom)
                if not hits:
                    items.append({"domain": dom, "careers_url": "(none)", "title": "", "snippet": ""})
                else:
                    for h in hits[:3]:
                        items.append({"domain": dom, "careers_url": h["url"], "title": h["title"], "snippet": h["snippet"]})
            st.session_state.bulk_results = items
            st.success("Bulk results updated.")
    with col2:
        if st.session_state.bulk_results:
            dfb = pd.DataFrame(st.session_state.bulk_results)
            st.download_button("⬇️ Download CSV", data=dfb.to_csv(index=False).encode("utf-8"), file_name="bulk_careers.csv", mime="text/csv")

    if st.session_state.bulk_results:
        dfb = pd.DataFrame(st.session_state.bulk_results)
        st.dataframe(dfb, use_container_width=True, hide_index=True)

with apply_tab:
    st.subheader("Apply — Tailor Resume & Open Application")
    if not st.session_state.jobs:
        st.info("Search jobs first, then pick a posting here.")
    else:
        options = [f"{j['title']} — {j.get('match',0)}%" for j in st.session_state.jobs]
        idx = st.selectbox("Choose a job", options=range(len(options)), format_func=lambda i: options[i])
        job = st.session_state.jobs[idx]
        st.link_button("Open official application", url=job["url"])  # submit on employer site

with saved_tab:
    st.subheader("Saved (recent tailored resumes show here if generated)")
    if os.path.exists(INDEX_CSV):
        df = pd.read_csv(INDEX_CSV)
        st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("""
<div class="muted small" style="margin-top:1rem;">
  Built with ❤️ using Streamlit + DuckDuckGo + Wikipedia + YAKE + Segno. Optional OpenAI/compatible LLM support.
</div>
""", unsafe_allow_html=True)
