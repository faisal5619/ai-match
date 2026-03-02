import re
from io import BytesIO

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Page setup
# =========================
st.set_page_config(page_title="AI Match", page_icon="🧠", layout="wide")

PRIMARY = "#2563EB"
BG = "#F7F8FA"
CARD = "#FFFFFF"
TEXT_MUTED = "rgba(0,0,0,0.65)"

# Hide Streamlit sidebar and default menu/footer
st.markdown(
    f"""
    <style>
      .stApp {{ background: {BG}; }}
      [data-testid="stSidebar"] {{ display: none; }}
      #MainMenu {{ visibility: hidden; }}
      footer {{ visibility: hidden; }}
      header {{ visibility: hidden; }}

      .container {{
        max-width: 1080px;
        margin: 0 auto;
        padding: 10px 10px 60px 10px;
      }}

      .navbar {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(247,248,250,0.9);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0,0,0,0.06);
        padding: 14px 0;
      }}

      .nav-inner {{
        max-width: 1080px;
        margin: 0 auto;
        padding: 0 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}

      .brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 800;
        font-size: 16px;
      }}

      .logo {{
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: {PRIMARY};
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 900;
      }}

      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(37,99,235,0.10);
        color: {PRIMARY};
        font-weight: 700;
        font-size: 12px;
      }}

      .title {{
        font-size: 44px;
        font-weight: 900;
        margin: 10px 0 6px 0;
        line-height: 1.08;
      }}

      .subtitle {{
        font-size: 16px;
        color: {TEXT_MUTED};
        margin-bottom: 18px;
        max-width: 640px;
      }}

      .card {{
        background: {CARD};
        padding: 18px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.05);
      }}

      .section-title {{
        font-weight: 800;
        font-size: 18px;
        margin-bottom: 12px;
      }}

      .feature {{
        padding: 14px;
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.06);
        background: white;
      }}

      .smallmuted {{ color: {TEXT_MUTED}; font-size: 13px; }}

      .btn-primary button {{
        background: {PRIMARY} !important;
        color: white !important;
        border-radius: 14px !important;
        border: none !important;
        font-weight: 800 !important;
        height: 44px !important;
      }}

      .btn-ghost button {{
        background: transparent !important;
        color: {PRIMARY} !important;
        border-radius: 14px !important;
        border: 1px solid rgba(37,99,235,0.35) !important;
        font-weight: 800 !important;
        height: 44px !important;
      }}

      /* Make inputs look more “SaaS” */
      textarea, input {{
        border-radius: 14px !important;
      }}

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# NLP / AI logic
# =========================
def extract_text(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()

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


def skill_match_score(cv_text: str, jd_text: str) -> float:
    cv_sk = find_skills(cv_text)
    jd_sk = find_skills(jd_text)
    if not jd_sk:
        return 0.0
    return (len(cv_sk & jd_sk) / len(jd_sk)) * 100.0


def compute_all(cv_text: str, jd_text: str):
    sim = tfidf_similarity_score(cv_text, jd_text)
    sk = skill_match_score(cv_text, jd_text)
    final = sim if sk == 0.0 else (0.25 * sim + 0.75 * sk)
    cv_sk = find_skills(cv_text)
    jd_sk = find_skills(jd_text)
    matched = sorted(cv_sk & jd_sk)
    missing = sorted(jd_sk - cv_sk)
    return final, sim, sk, matched, missing


# =========================
# Navigation state
# =========================
if "page" not in st.session_state:
    st.session_state.page = "Home"


def go(page_name: str):
    st.session_state.page = page_name
    st.rerun()


# =========================
# Top navbar (Figma-like)
# =========================
st.markdown(
    """
    <div class="navbar">
      <div class="nav-inner">
        <div class="brand">
          <div class="logo">🧠</div>
          <div>AI Match</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Buttons row (acts like navbar links)
nav = st.columns([1,1,1,1,6])
with nav[0]:
    if st.button("Home", use_container_width=True):
        go("Home")
with nav[1]:
    if st.button("Dashboard", use_container_width=True):
        go("Dashboard")
with nav[2]:
    if st.button("About", use_container_width=True):
        go("About")
with nav[3]:
    if st.button("Contact", use_container_width=True):
        go("Contact")


# =========================
# Pages
# =========================
def page_home():
    st.markdown('<div class="container">', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown('<span class="pill">AI-Powered Recruitment</span>', unsafe_allow_html=True)
        st.markdown('<div class="title">AI-Powered CV<br/>Screening & Job<br/>Matching</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Upload your CV and compare it instantly with job descriptions using NLP. Get match scores, missing skills, and recommendations in seconds.</div>',
            unsafe_allow_html=True
        )

        b1, b2 = st.columns([1, 1])
        with b1:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            if st.button("Try Now →", use_container_width=True):
                go("Dashboard")
            st.markdown('</div>', unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
            if st.button("Learn More", use_container_width=True):
                go("About")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<span class="pill">Free to use</span> <span class="pill">Instant results</span> <span class="pill">AI-powered</span>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Quick Preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="smallmuted">Example output</div>', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.metric("Match Score", "87%")
        st.markdown("<div class='smallmuted'>Analysis: ~2 seconds</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br/><br/>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; font-weight:900;'>Why Choose AI Match?</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:rgba(0,0,0,0.65); margin-top:-6px;'>Streamline hiring with intelligent CV analysis and matching</p>", unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    for col, title, desc in [
        (f1, "AI-Powered Analysis", "Analyze CVs and job descriptions to find the best match."),
        (f2, "Instant Results", "Get match scores and insights in seconds."),
        (f3, "Accurate Matching", "Skill matching + similarity scoring for better decisions."),
        (f4, "Secure & Private", "Your data stays private during analysis.")
    ]:
        with col:
            st.markdown('<div class="feature">', unsafe_allow_html=True)
            st.markdown(f"<b>{title}</b><br/><span class='smallmuted'>{desc}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_dashboard():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight:900;'>CV Analysis Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p class='smallmuted'>Upload your CV and paste a job description to get instant AI-powered insights.</p>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<b>Upload CV</b>", unsafe_allow_html=True)
        cv_file = st.file_uploader("PDF, DOC, or DOCX", type=["pdf", "docx", "txt"])
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("<b>Job Description</b>", unsafe_allow_html=True)
        jd = st.text_area("Paste the job description here...", height=220)
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        run = st.button("✨ Analyze Match", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<b>Results</b>", unsafe_allow_html=True)

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

                    a, b = st.columns(2)
                    with a:
                        st.markdown("✅ **Matched Skills**")
                        st.write(", ".join(matched) if matched else "None detected")
                    with b:
                        st.markdown("⚠️ **Missing Skills**")
                        st.write(", ".join(missing) if missing else "None detected")

                    st.markdown("### AI Recommendations")
                    if missing:
                        for s in missing[:8]:
                            st.write(f"- Consider adding/highlighting: {s}")
                    else:
                        st.write("Your CV aligns well with this job description.")

                    with st.expander("Score breakdown"):
                        st.write(f"TF-IDF Similarity: {sim:.1f}%")
                        st.write(f"Skill Match: {sk:.1f}%")
                        st.write("Final = 25% Similarity + 75% Skill Match")

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Ready to analyze. Upload a CV, paste a job description, then click Analyze.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_about():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight:900;'>About</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
          <b>AI Match</b> is an AI-powered CV screening and job matching system.
          It compares candidate CVs with job descriptions using NLP, then provides:
          <br/><br/>
          • Match Score (%)<br/>
          • Matched & Missing Skills<br/>
          • Recommendations to improve CV alignment<br/><br/>
          <b>Methodology</b><br/>
          • Text extraction (PDF/DOCX)<br/>
          • TF-IDF vectorization + cosine similarity<br/>
          • Skill gap analysis (keyword-based)<br/>
          • Combined scoring for recruiter-friendly results
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


def page_contact():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight:900;'>Get In Touch</h2>", unsafe_allow_html=True)
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
    page_home()
elif st.session_state.page == "Dashboard":
    page_dashboard()
elif st.session_state.page == "About":
    page_about()
else:
    page_contact()
