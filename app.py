import re
from io import BytesIO

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# CONFIG + SIMPLE STYLING
# -----------------------
st.set_page_config(page_title="AI Match", page_icon="🧠", layout="wide")

PRIMARY = "#2563EB"   # blue
BG = "#F7F8FA"        # light gray background
CARD = "#FFFFFF"      # card background

st.markdown(
    f"""
    <style>
      .stApp {{ background: {BG}; }}
      .card {{
        background: {CARD};
        padding: 18px 18px;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
        margin-bottom: 14px;
      }}
      .title {{
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 6px;
      }}
      .subtitle {{
        font-size: 16px;
        color: rgba(0,0,0,0.65);
        margin-bottom: 12px;
      }}
      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(37,99,235,0.10);
        color: {PRIMARY};
        font-weight: 600;
        margin-right: 8px;
        font-size: 12px;
      }}
      .btn-primary button {{
        background: {PRIMARY} !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# NLP PIPELINE (WORKING AI)
# -----------------------
def extract_text(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text).strip()

    if name.endswith(".docx"):
        doc = Document(BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()

    if name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore").strip()

    raise ValueError("Upload PDF, DOCX, or TXT only.")


def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^a-z0-9+\#\.\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tfidf_similarity_score(cv_text: str, jd_text: str) -> float:
    cv = clean_text(cv_text)
    jd = clean_text(jd_text)

    if len(cv) < 30 or len(jd) < 30:
        return 0.0

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    m = vec.fit_transform([cv, jd])
    sim = cosine_similarity(m[0:1], m[1:2])[0][0]
    return max(0.0, min(1.0, float(sim))) * 100.0


SKILLS = {
    "java", "python", "sql", "git", "linux", "docker", "aws", "azure",
    "api", "rest", "rest api", "html", "css", "javascript",
    "ai", "artificial intelligence", "machine learning", "nlp", "data analysis",
    "computer science", "programming",
    "oop", "object oriented programming", "data structures", "algorithms",
    "teamwork", "problem solving", "time management", "communication",
    "word", "excel", "powerpoint", "microsoft office",
    "english", "arabic",
}


def find_skills(text: str) -> set:
    t = clean_text(text)
    found = set()
    for s in SKILLS:
        if re.search(r"\b" + re.escape(s) + r"\b", t):
            found.add(s)
    return found


def skill_match_score(cv_text: str, jd_text: str) -> float:
    cv_sk = find_skills(cv_text)
    jd_sk = find_skills(jd_text)
    if not jd_sk:
        return 0.0
    return (len(cv_sk & jd_sk) / len(jd_sk)) * 100.0


def compute_all(cv_text: str, jd_text: str):
    sim = tfidf_similarity_score(cv_text, jd_text)
    sk = skill_match_score(cv_text, jd_text)

    # recruiter-friendly weights
    final = sim if sk == 0.0 else (0.25 * sim + 0.75 * sk)

    cv_sk = find_skills(cv_text)
    jd_sk = find_skills(jd_text)
    matched = sorted(cv_sk & jd_sk)
    missing = sorted(jd_sk - cv_sk)

    return final, sim, sk, matched, missing


# -----------------------
# PAGES (LIKE YOUR FIGMA)
# -----------------------
def page_home():
    st.markdown('<div class="pill">AI-Powered</div><div class="pill">Instant Results</div><div class="pill">Skill Gap Analysis</div>', unsafe_allow_html=True)

    st.markdown('<div class="title">AI-Powered CV Screening<br/> & Job Matching</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload your CV, paste a job description, and get a match score with missing skills — instantly.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="card"><b>What it does</b><br/><br/>• Calculates Match Score (%)<br/>• Extracts matched & missing skills<br/>• Gives recommendations to improve your CV</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><b>Who it’s for</b><br/><br/>• HR recruiters (fast screening)<br/>• Job seekers (CV improvement)<br/>• Universities (career support)</div>', unsafe_allow_html=True)

    st.markdown("### Ready?")
    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    if st.button("Analyze Your CV →", use_container_width=True):
        st.session_state.page = "Dashboard"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def page_dashboard():
    st.markdown('<div class="title">CV Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload your CV and compare it with a job description.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Inputs")
        cv_file = st.file_uploader("Upload CV (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
        jd = st.text_area("Job Description", height=220)
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        run = st.button("✨ Analyze Match", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Results")

        if run:
            if not cv_file:
                st.error("Upload a CV first.")
            elif not jd.strip():
                st.error("Paste a job description first.")
            else:
                try:
                    cv_text = extract_text(cv_file.name, cv_file.read())
                    final, sim, sk, matched, missing = compute_all(cv_text, jd)

                    st.metric("Match Score", f"{final:.0f}%")
                    st.progress(min(100, max(0, int(final))) / 100)

                    colA, colB = st.columns(2)
                    with colA:
                        st.write("✅ Matched Skills")
                        st.write(", ".join(matched) if matched else "None detected")
                    with colB:
                        st.write("⚠️ Missing Skills")
                        st.write(", ".join(missing) if missing else "None detected")

                    st.write("### Recommendations")
                    if missing:
                        for s in missing[:10]:
                            st.write(f"- Add or highlight: {s}")
                    else:
                        st.write("Your CV aligns well with this job description.")

                    with st.expander("Score breakdown"):
                        st.write(f"TF-IDF Similarity: {sim:.1f}%")
                        st.write(f"Skill Match: {sk:.1f}%")
                        st.write("Final = 25% Similarity + 75% Skill Match")

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Upload CV + paste JD, then click Analyze.")
        st.markdown('</div>', unsafe_allow_html=True)


def page_about():
    st.markdown('<div class="title">About</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">This project is an AI-powered CV screening and job matching system. It compares CV text with job descriptions using NLP techniques, computes a match score, and identifies missing skills to help both recruiters and job seekers.</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><b>Method</b><br/><br/>• Text extraction (PDF/DOCX)<br/>• Text preprocessing<br/>• TF-IDF vectorization<br/>• Cosine similarity scoring<br/>• Skill gap analysis</div>', unsafe_allow_html=True)


def page_contact():
    st.markdown('<div class="title">Contact</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><b>Office</b><br/>Abha, Aseer Region<br/>Kingdom of Saudi Arabia</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><b>Email</b><br/>alhuaithy90@gmail.com</div>', unsafe_allow_html=True)


# -----------------------
# NAV (LIKE A WEBSITE)
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("AI Match")
choice = st.sidebar.radio(
    "Navigation",
    ["Home", "Dashboard", "About", "Contact"],
    index=["Home", "Dashboard", "About", "Contact"].index(st.session_state.page),
)
st.session_state.page = choice

if choice == "Home":
    page_home()
elif choice == "Dashboard":
    page_dashboard()
elif choice == "About":
    page_about()
else:
    page_contact()
