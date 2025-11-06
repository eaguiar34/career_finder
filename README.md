# Career Fair Companion (Streamlit)

A free, student‑friendly app to **research companies**, **generate recruiter questions**, **surface jobs with a % match**, **tailor resumes (opt‑in)**, **save tailored versions**, **find recruiter/HR contacts**, and **share your info via QR**. Built with **Streamlit + Python**, and optionally powered by **OpenAI or any OpenAI‑compatible local server** (Ollama/LM Studio).

---

## ✨ Features

* **Multi‑major presets**: Engineering (Civil/Mechanical/ECE), CS/Software, Data/Analytics, Business/Finance, Marketing/Comms, Design/UX, Life Sciences, Social Sciences, Accounting **+ Nursing, Education, Architecture, Environmental Science**. One‑click preset fills role, interests, tools, and assistant defaults.
* **Company research**: Wikipedia snapshot (optionally AI‑summarized) + recent projects/press via DuckDuckGo.
* **Questions for recruiters**: Smart buckets (Role & Impact, Team & Mentorship, Projects & Methods, Culture & Growth, Recruiter‑Specific). Heuristic by default, LLM‑powered if configured.
* **Job search with % match**: Finds links across ATS (Lever/Greenhouse/Workday/SmartRecruiters/iCIMS) & boards. Ranks by **% match** to your role, interests, location, and source quality.
* **Preference‑driven search**: Tell the assistant what you want (role, tasks, skills, location, pay, target companies) to find cross‑company matches.
* **Apply tab**: Open the official application link; upload/preview your resume; **opt‑in AI tailoring** (or keyword‑highlight fallback); download tailored DOCX/TXT; **databank** of generated resumes.
* **Contacts finder**: Quick searches for recruiters/HR/university recruiting + CSV import; keep a saved list.
* **Networking QR**: Build a **vCard** or share your **LinkedIn URL** as a QR code.
* **School themes**: Choose your school (e.g., Cal State LA, UCLA, USC, UC Berkeley, MIT) or pick custom colors for a little **color pop**.
* **Privacy by default**: No server DB. Everything is in session + files. On Streamlit Cloud, storage is **ephemeral**—download what you need.

> ⚠️ Applications are still completed on the employer’s site (ATS). The app just helps you prepare, tailor, and launch the link.

---

## 🚀 Quickstart (Local)

**Requirements**: Python 3.10+ recommended.

```bash
git clone <your-repo>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Optional AI (two ways)

1. **OpenAI Cloud**

   * Set env vars or use Streamlit **Secrets**:

   ```bash
   export OPENAI_API_KEY="sk-..."
   export OPENAI_MODEL="gpt-4o-mini"   # optional
   ```

   Or create `.streamlit/secrets.toml`:

   ```toml
   openai_api_key = "sk-..."
   openai_model   = "gpt-4o-mini"
   # openai_base_url = "https://..."  # only if using a compatible host
   ```

2. **FREE local LLM (OpenAI‑compatible)**

   * Install a local server and model:

   ```bash
   # Ollama example
   brew install ollama   # macOS; see https://ollama.com for other OS
   ollama run llama3.1:8b

   # Then run the app with these vars:
   export OPENAI_BASE_URL="http://localhost:11434/v1"
   export OPENAI_API_KEY="ollama"         # any string works for Ollama
   export OPENAI_MODEL="llama3.1:8b-instruct"
   streamlit run app.py
   ```

   * LM Studio works similarly; point `OPENAI_BASE_URL` at its local server.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push to GitHub: include `app.py`, `requirements.txt`, and commit.
2. Go to **share.streamlit.io** → New app → choose repo/branch → **file = app.py** → Deploy.
3. In **Secrets** (⚙️ → Settings → Secrets):

   * `openai_api_key = YOUR_KEY` (only if you want cloud AI)
   * optional: `openai_model = gpt-4o-mini`
   * optional: `openai_base_url = https://...` (if using a hosted OpenAI‑compatible API)

**Notes**

* Streamlit Cloud cannot connect to your localhost (Ollama/LM Studio). Use a hosted API for cloud.
* The app writes to `data/resumes/` and `index.csv` for the resume databank. On Cloud this is **ephemeral**; download files you want to keep.

---

## 🔢 How the % Match works (high level)

The app computes a score from job text + your inputs:

* **Role similarity** (string similarity between your desired role and job text)
* **Interests/skills overlap** (keywords in the posting)
* **Location** match (if you set one) and **remote/hybrid** hints
* **Source quality** bonus (ATS sites get a bump)

Two scoring paths are used:

* **Company view** (`Find Jobs` for one company): role + interests + location + ATS bonus
* **Preference view** (assistant search): uses your preferences (role/tasks/skills/location/pay)

These are compact heuristics, tuned for speed and clarity over perfect precision. Customize in `score_job()` and `score_job_vs_prefs()`.

---

## 📝 Resume Tailoring (opt‑in)

* Upload PDF/DOCX/TXT; preview extracted text.
* Click **Fetch job page text** to pull context from the posting.
* Turn on **Use AI** if you have a model configured, then **Tailor now**. If AI is off, the app highlights posting keywords in your resume.
* Tailored resumes save under `data/resumes/` and show up in the databank table (also downloadable).

**Ethics**: Tailoring strengthens alignment but must remain truthful. Edit the output to reflect your real experience.

---

## 🔎 Sources & Fair Use

* Uses DuckDuckGo Search to find press, projects, ATS listings, and public recruiter profiles.
* Keep employer and platform **Terms of Service** in mind; follow robots/no‑scrape rules and rate limits.
* Wikipedia content is summarized with attribution implied by link; confirm details on official sites when accuracy matters.

---

## 🎨 Theming & Presets

* Change the look with **School** themes or custom colors in the sidebar.
* Add or tweak majors by editing `MAJOR_PRESETS` in `app.py` (role, interests, tasks, skills, tools).

---

## 🧩 Requirements (reference)

Create `requirements.txt` with (adjust versions as needed):

```txt
streamlit>=1.38
pandas>=2.2
requests>=2.32
beautifulsoup4>=4.12
duckduckgo-search>=6.1
wikipedia>=1.4
yake @ git+https://github.com/LIAAD/yake#egg=yake   # optional; or pin a wheel if available
segno>=1.6
PyPDF2>=3.0
python-docx>=1.1
openai>=1.51
```

> Tip: if `yake` fails on your platform, comment it out; the app falls back to a simple keyword extractor.

---

## 🛠️ Troubleshooting

* **No AI replies**: check `OPENAI_API_KEY` (or local server `OPENAI_BASE_URL`) and network egress in Cloud.
* **DDG rate limits**: try fewer/max results, add delays, or rerun.
* **PDF extraction empty**: some PDFs are image‑based; export a text‑based PDF or upload DOCX/TXT.
* **Cloud storage**: download your tailored files; Cloud filesystems are not guaranteed long‑term.

---

## 📄 License

MIT — free to use, modify, and share. Please keep the license in your repo.

---

## 🧭 Roadmap ideas

* Export recruiter questions to a 1‑pager PDF
* True cloud persistence (Drive/S3/Firestore)
* Built‑in LinkedIn API/Graph integrations (respecting terms)
* Per‑major weight tuning for % match
* Interview prep drills (STAR bullets from resume)
