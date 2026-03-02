import re
from io import BytesIO

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="AI Match", page_icon="🧠", layout="wide")

PRIMARY = "#2563EB"
BG = "#F7F8FA"
CARD = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "#475569"
GREEN_BG = "#DCFCE7"
GREEN_TXT = "#166534"
ORANGE_BG = "#FFEDD5"
ORANGE_TXT = "#9A3412"


# --------- CSS (fix invisible text + make it look like SaaS) ----------
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{display:none;}}
    header, footer, #MainMenu {{visibility:hidden;}}

    .stApp {{ background: {BG}; color: {TEXT}; }}

    .wrap {{ max-width: 1120px; margin: 0 auto; padding: 14px 10px 60px; }}

    .topbar {{
        background: rgba(247,248,250,0.92);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(15,23,42,0.08);
        position: sticky; top: 0; z-index: 99;
        padding: 14px 0;
        margin-bottom: 18px;
    }}
    .topbar-inner {{
        max-width: 1120px; margin: 0 auto; padding: 0 10px;
        display:flex; align-items:center; justify-content:space-between;
    }}
    .brand {{ display:flex; align-items:center; gap:10px; font-weight:900; }}
    .logo {{
        width:34px; height:34px; border-radius:10px;
        background:{PRIMARY}; color:white; display:flex; align-items:center; justify-content:center;
        font-weight:900;
    }}

    .navbtns .stButton>button {{
        border-radius: 12px !important;
        height: 40px !important;
        font-weight: 800 !important;
        border: 1px solid rgba(37,99,235,0.25) !important;
        background: white !important;
        color: {PRIMARY} !important;
    }}
    .navbtns .stButton>button:hover {{
        border: 1px solid rgba(37,99,235,0.45) !important;
    }}

    .hero-title {{ font-size: 46px; font-weight: 950; line-height:1.05; margin: 8px 0; }}
    .hero-sub {{ color: {MUTED}; font-size: 15px; max-width: 640px; }}

    .card {{
        background:{CARD};
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(2,6,23,0.06);
        padding: 18px;
    }}

    .pill {{
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(37,99,235,0.10);
        color: {PRIMARY};
        font-weight: 800;
        font-size: 12px;
        margin-right: 8px;
    }}

    /* Inputs (fix invisible text + nicer corners) */
    textarea {{
        color: {TEXT} !important;
        background: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid rgba(15,23,42,0.10) !important;
    }}
    input {{
        color: {TEXT} !important;
    }}

    /* Primary button */
    .primary .stButton>button {{
        background:{PRIMARY} !important;
        color:white !important;
        border:none !important;
        border-radius: 14px !important;
        font-weight: 900 !important;
        height: 44px !important;
    }}

    /* Skill chips */
    .chip {{
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 12px;
        margin: 6px 6px 0 0;
        border: 1px solid rgba(15,23,42,0.08);
    }}
    .chip-green {{ background:{GREEN_BG}; color:{GREEN_TXT}; }}
    .chip-orange {{ background:{ORANGE_BG}; color:{ORANGE_TXT}; }}

    /* Circle score */
    .ring {{
        width: 160px; height: 160px; border-radius: 999px;
        display:flex; align-items:center; justify-content:center;
        margin: 6px 0 8px;
        background:
            conic-gradient({PRIMARY} var(--p), rgba(15,23,42,0.10) 0);
    }}
    .ring-inner {{
        width: 128px; height: 128px; border-radius: 999px;
        background: white;
        display:flex; align-items:center; justify-content:center;
        flex-direction: column;
        border: 1px solid rgba(15,23,42,0.08);
    }}
    .score {{ font-size: 38px; font-weight: 950; line-height: 1; }}
    .score-sub {{ color:{MUTED}; font-weight: 700; font-size: 12px; margin-top: 4px; }}

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- AI / NLP ----------------
def extract_text(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join([(p.extract_text() or "") for p in reader.pages]).strip()
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
    "java","python","sql","git","linux","docker","aws","azure",
    "api","rest","rest api","html","css","javascript","typescript","react",
    "ai","artificial intelligence","machine learning","nlp","data analysis",
    "computer science","programming","oop","object oriented programming","data structures","algorithms",
    "microsoft office","word","excel","powerpoint",
    "teamwork","collaboration","problem solving","time management","communication","english","arabic"
}

def find_skills(text: str) -> set:
    t = clean_text(text)
    found = set()
    for s in SKILLS:
        if re.search(r"\b" + re.escape(s) + r"\b", t):
            found.add(s)
    return found


def compute_all(cv_text: str, jd_text: str):
    sim = tfidf_similarity_score(cv_text, jd_text)
    cv_sk = find_skills(cv_text)
    jd_sk = find_skills(jd_text)
    sk = 0.0 if not jd_sk else (len(cv_sk & jd_sk) / len(jd_sk)) * 100.0
    final = sim if sk == 0.0 else (0.25 * sim + 0.75 * sk)
    matched = sorted(cv_sk & jd_sk)
    missing = sorted(jd_sk - cv_sk)
    return final, sim, sk, matched, missing


# ---------------- NAV ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go(p):
    st.session_state.page = p
    st.rerun()

st.markdown(
    """
    <div class="topbar">
      <div class="topbar-inner">
        <div class="brand"><div class="logo">🧠</div>AI Match</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

nav = st.columns([1, 1, 1, 1, 6])
for i, (label, key) in enumerate([("Home","Home"),("Dashboard","Dashboard"),("About","About"),("Contact","Contact")]):
    with nav[i]:
        st.markdown('<div class="navbtns">', unsafe_allow_html=True)
        if st.button(label, use_container_width=True):
            go(key)
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- PAGES ----------------
def home():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown('<span class="pill">AI-Powered Recruitment</span>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">AI-Powered CV<br/>Screening & Job<br/>Matching</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Upload your CV and compare it instantly with job descriptions using NLP. Get match scores, missing skills, and recommendations in seconds.</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            st.markdown('<div class="primary">', unsafe_allow_html=True)
            if st.button("Try Now →", use_container_width=True):
                go("Dashboard")
            st.markdown('</div>', unsafe_allow_html=True)
        with b2:
            if st.button("Learn More", use_container_width=True):
                go("About")

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<span class="pill">Free to use</span><span class="pill">Instant results</span><span class="pill">AI-powered</span>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<b>Quick Preview</b><div class='smallmuted'>Example output</div>", unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.metric("Match Score", "87%")
        st.markdown("<div class='smallmuted'>Analysis ~2 seconds</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/><br/>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; font-weight:950;'>Why Choose AI Match?</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#475569; margin-top:-6px;'>Streamline hiring with intelligent CV analysis and matching</p>", unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    feats = [
        ("AI-Powered Analysis","Advanced algorithms analyze CVs and job descriptions."),
        ("Instant Results","Get comprehensive results and insights in seconds."),
        ("Accurate Matching","Skill match + similarity scoring for better decisions."),
        ("Secure & Private","Your data is processed securely with privacy protection.")
    ]
    for col, (t, d) in zip([f1,f2,f3,f4], feats):
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<b>{t}</b><br/><span style='color:#475569; font-size:13px;'>{d}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def dashboard():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight:950; margin-bottom:4px;'>CV Analysis Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<div style='color:#475569; margin-bottom:16px;'>Upload your CV and paste a job description to get instant AI-powered insights.</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<b>Upload CV</b>", unsafe_allow_html=True)
        cv_file = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("<b>Job Description</b>", unsafe_allow_html=True)
        jd = st.text_area("Paste the job description here...", height=220)

        st.markdown('<div class="primary">', unsafe_allow_html=True)
        run = st.button("✨ Analyze Match", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<b>Results</b>", unsafe_allow_html=True)

        if run:
            if not cv_file:
                st.error("Upload a CV first.")
            elif not jd.strip():
                st.error("Paste a job description first.")
            else:
                cv_text = extract_text(cv_file.name, cv_file.read())
                final, sim, sk, matched, missing = compute_all(cv_text, jd)

                p = max(0, min(100, int(final)))
                st.markdown(
                    f"""
                    <div class="ring" style="--p:{p}%;">
                      <div class="ring-inner">
                        <div class="score">{p}%</div>
                        <div class="score-sub">Good match</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<div style='font-weight:900; margin-top:10px;'>Matched Skills</div>", unsafe_allow_html=True)
                if matched:
                    chips = "".join([f"<span class='chip chip-green'>{s}</span>" for s in matched])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.write("None detected")

                st.markdown("<div style='font-weight:900; margin-top:14px;'>Missing Skills</div>", unsafe_allow_html=True)
                if missing:
                    chips = "".join([f"<span class='chip chip-orange'>{s}</span>" for s in missing])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.write("None detected")

                st.markdown("<div style='font-weight:900; margin-top:18px;'>AI Recommendations</div>", unsafe_allow_html=True)
                if missing:
                    for s in missing[:8]:
                        st.write(f"- Consider adding/highlighting: {s}")
                else:
                    st.write("Your CV aligns well with this job description.")

                with st.expander("Score breakdown"):
                    st.write(f"TF-IDF Similarity: {sim:.1f}%")
                    st.write(f"Skill Match: {sk:.1f}%")
                    st.write("Final = 25% Similarity + 75% Skill Match")
        else:
            st.info("Ready to analyze. Upload a CV and paste a job description, then click Analyze.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def about():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight:950;'>About</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
        <b>AI Match</b> compares a candidate CV with a job description to produce a match score and skill gap analysis.
        <br/><br/>
        <b>Methods</b><br/>
        • Text extraction (PDF/DOCX)<br/>
        • TF-IDF vectorization + Cosine similarity<br/>
        • Keyword-based skill extraction<br/>
        • Combined scoring (skills-weighted)
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


def contact():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight:950;'>Get In Touch</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
          <b>Office</b><br/>
          Abha, Aseer Region<br/>
          Kingdom of Saudi Arabia<br/><br/>
          <b>Email</b><br/>
          support@aimatch.com
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.page == "Home":
    home()
elif st.session_state.page == "Dashboard":
    dashboard()
elif st.session_state.page == "About":
    about()
else:
    contact()
