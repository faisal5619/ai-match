import re
from io import BytesIO

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Match", page_icon="🧠", layout="wide")

PRIMARY = "#2563EB"
BG = "#F7F8FA"
CARD = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "#475569"
BORDER = "rgba(15,23,42,0.10)"
SHADOW = "0 12px 30px rgba(2,6,23,0.06)"

GREEN_BG = "#DCFCE7"
GREEN_TXT = "#166534"
ORANGE_BG = "#FFEDD5"
ORANGE_TXT = "#9A3412"

# ---------------- CSS ----------------
st.markdown(
    f"""
    <style>
    #MainMenu {{visibility:hidden;}}
    header {{visibility:hidden;}}
    footer {{visibility:hidden;}}

    .stApp {{
        background: #F7F8FA;
        color: #0F172A;
    }}

    .wrap {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px 20px 60px;
    }}

    .nav {{
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(15,23,42,0.08);
    }}

    .nav-inner {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 14px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 900;
        font-size: 18px;
    }}

    .logo {{
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: #2563EB;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 900;
        box-shadow: 0 8px 18px rgba(37,99,235,0.25);
    }}

    .navlinks .stButton > button {{
        background: transparent !important;
        border: none !important;
        color: #64748B !important;
        font-weight: 700 !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        height: auto !important;
        box-shadow: none !important;
    }}

    .navlinks .stButton > button:hover {{
        background: rgba(37,99,235,0.08) !important;
        color: #2563EB !important;
    }}

    .active .stButton > button {{
        background: rgba(37,99,235,0.12) !important;
        color: #2563EB !important;
    }}

    .card {{
        background: white;
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        box-shadow: 0 10px 28px rgba(2,6,23,0.06);
        padding: 22px;
    }}

    .title {{
        font-size: 42px;
        font-weight: 950;
        margin: 8px 0 10px;
        line-height: 1.12;
        letter-spacing: -0.02em;
    }}

    .subtitle {{
        color: #64748B;
        font-size: 16px;
        margin-bottom: 18px;
        max-width: 760px;
        line-height: 1.7;
    }}

    .pill {{
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(37,99,235,0.10);
        color: #2563EB;
        font-weight: 800;
        font-size: 12px;
        margin-right: 8px;
        margin-bottom: 6px;
    }}

    textarea {{
        color: #0F172A !important;
        background: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid rgba(15,23,42,0.10) !important;
    }}

    input {{
        color: #0F172A !important;
    }}

    .primary .stButton > button {{
        background: linear-gradient(90deg, #2563EB, #4F46E5) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 900 !important;
        height: 46px !important;
        box-shadow: 0 10px 24px rgba(79,70,229,0.18) !important;
    }}

    .chip {{
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 12px;
        margin: 6px 6px 0 0;
        border: 1px solid rgba(15,23,42,0.06);
    }}

    .chip-green {{
        background: #DCFCE7;
        color: #166534;
    }}

    .chip-orange {{
        background: #FFEDD5;
        color: #9A3412;
    }}

    .ring {{
        width: 170px;
        height: 170px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 8px auto 14px;
        background: conic-gradient(#F59E0B var(--p), rgba(148,163,184,0.22) 0);
    }}

    .ring-inner {{
        width: 128px;
        height: 128px;
        border-radius: 999px;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        border: 1px solid rgba(15,23,42,0.08);
    }}

    .score {{
        font-size: 40px;
        font-weight: 950;
        line-height: 1;
        color: #B45309;
    }}

    .score-sub {{
        color: #64748B;
        font-weight: 800;
        font-size: 13px;
        margin-top: 6px;
    }}

    .h {{
        font-weight: 900;
        margin-top: 12px;
        margin-bottom: 8px;
        font-size: 18px;
    }}

    .upload-box {{
        border: 2px dashed rgba(148,163,184,0.35);
        border-radius: 18px;
        padding: 28px 18px;
        text-align: center;
        background: #FAFBFC;
        margin-bottom: 14px;
    }}

    .upload-note {{
        color: #64748B;
        font-size: 13px;
        margin-top: 6px;
    }}

    .muted {{
        color: #64748B;
        font-size: 14px;
    }}

    .section-gap {{
        margin-top: 18px;
    }}

    .big-pill {{
        display: inline-block;
        padding: 9px 16px;
        border-radius: 999px;
        background: rgba(37,99,235,0.10);
        color: #2563EB;
        font-weight: 800;
        font-size: 14px;
        margin-right: 10px;
        margin-bottom: 8px;
    }}

    .feature-box {{
        background: white;
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        box-shadow: 0 10px 28px rgba(2,6,23,0.06);
        padding: 22px;
        min-height: 150px;
    }}

    .preview-big {{
        background: linear-gradient(135deg,#EEF2FF,#F8FAFC);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 24px;
        box-shadow: 0 10px 28px rgba(2,6,23,0.06);
        padding: 28px;
        min-height: 360px;
    }}

    .dotted-box {{
        border: 2px dashed rgba(148,163,184,0.35);
        border-radius: 18px;
        padding: 20px;
        background: #FAFBFC;
    }}

    .inside-box-title {{
        font-size: 18px;
        font-weight: 900;
        margin-bottom: 10px;
        color: #0F172A;
    }}

    .about-big {{
        font-size: 17px;
        line-height: 1.9;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- NLP ----------------
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
    "java", "python", "sql", "git", "linux", "docker", "aws", "azure",
    "api", "rest", "rest api", "html", "css", "javascript", "typescript", "react",
    "ai", "artificial intelligence", "machine learning", "nlp", "data analysis",
    "computer science", "programming", "oop", "object oriented programming", "data structures", "algorithms",
    "microsoft office", "word", "excel", "powerpoint",
    "teamwork", "collaboration", "problem solving", "time management", "communication", "english", "arabic"
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

# ---------------- NAV (THIS FIXES YOUR PROBLEM) ----------------
PAGES = ["Home", "Dashboard", "About", "Contact"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

# If URL has ?page=... use it, otherwise keep session_state (so uploads don't kick you back Home)
qp = st.query_params.get("page")
if qp in PAGES:
    st.session_state.page = qp

def set_page(p: str):
    st.session_state.page = p
    st.query_params["page"] = p  # keep URL in sync (important!)
    st.rerun()

# ---------------- NAVBAR UI ----------------
st.markdown('<div class="nav"><div class="nav-inner">', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="brand">
        <div class="logo">🧠</div>
        <div>AI Match</div>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("</div></div>", unsafe_allow_html=True)

# Buttons row under navbar (stable + doesn’t break on reruns)
st.markdown('<div class="wrap"><div class="navlinks">', unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns([1, 1, 1, 1])

def nav_btn(label, target):
    cls = "active" if st.session_state.page == target else ""
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    if st.button(label, use_container_width=True, key=f"nav_{target}"):
        set_page(target)
    st.markdown("</div>", unsafe_allow_html=True)

with b1:
    nav_btn("⌂ Home", "Home")
with b2:
    nav_btn("⌘ Dashboard", "Dashboard")
with b3:
    nav_btn("ⓘ About", "About")
with b4:
    nav_btn("✉ Contact", "Contact")

st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------- PAGES ----------------
def page_home():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<span class="big-pill">AI-Powered Recruitment</span>', unsafe_allow_html=True)
        st.markdown('<div class="title" style="font-size:58px;">AI-Powered CV<br/>Screening & Job<br/>Matching</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle" style="font-size:18px; max-width:620px;">Upload your CV and compare it instantly with job descriptions using NLP. Get match scores, missing skills, and recommendations in seconds.</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns([1, 1], gap="small")
        with c1:
            st.markdown('<div class="primary">', unsafe_allow_html=True)
            if st.button("Try Now →", use_container_width=True, key="home_try"):
                set_page("Dashboard")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="primary">', unsafe_allow_html=True)
            if st.button("Learn More", use_container_width=True, key="home_learn"):
                set_page("About")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            '<span class="big-pill">Free to use</span><span class="big-pill">Instant results</span><span class="big-pill">AI-powered</span>',
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            """<div style="background: linear-gradient(135deg,#EEF2FF,#F8FAFC); border: 1px solid rgba(15,23,42,0.08); border-radius: 24px; box-shadow: 0 10px 28px rgba(2,6,23,0.06); padding: 28px; min-height: 360px;">
<div style="width:130px; height:130px; border-radius:24px; background:white; display:flex; align-items:center; justify-content:center; margin:0 auto 28px auto; font-size:58px; box-shadow:0 14px 28px rgba(15,23,42,0.08);">🧠</div>
<div style="display:flex; gap:14px;">
<div style="background:white; border:1px solid rgba(15,23,42,0.08); border-radius:16px; padding:16px; min-height:90px; box-shadow:0 10px 20px rgba(2,6,23,0.05); flex:1;">
<div style="color:#64748B; font-size:12px; font-weight:700;">Match Score</div>
<div style="color:#16A34A; font-size:34px; font-weight:900;">87%</div>
</div>
<div style="background:white; border:1px solid rgba(15,23,42,0.08); border-radius:16px; padding:16px; min-height:90px; box-shadow:0 10px 20px rgba(2,6,23,0.05); flex:1;">
<div style="color:#64748B; font-size:12px; font-weight:700;">Analysis</div>
<div style="color:#0F172A; font-size:18px; font-weight:800;">2 seconds</div>
</div>
</div>
</div>""",
            unsafe_allow_html=True
        )



    st.markdown("<br/><br/>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; font-weight:950; font-size:44px;'>Why Choose AI Match?</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#475569; margin-top:-6px; font-size:16px;'>Streamline hiring with intelligent CV analysis and matching</p>", unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4, gap="large")
    feats = [
        ("AI-Powered Analysis", "Advanced algorithms analyze CVs and job descriptions."),
        ("Instant Results", "Get comprehensive results and insights in seconds."),
        ("Accurate Matching", "Skill match + similarity scoring for better decisions."),
        ("Secure & Private", "Your data is processed securely with privacy protection.")
    ]

    for col, (t, d) in zip([f1, f2, f3, f4], feats):
        with col:
            st.markdown(
                f"""
                <div class="feature-box">
                    <div style="font-size:20px; font-weight:900; margin-bottom:10px;">{t}</div>
                    <div style="color:#475569; font-size:14px; line-height:1.8;">{d}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


def page_dashboard():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("<div class='title' style='font-size:40px;'>CV Analysis Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle' style='font-size:17px;'>Upload your CV and paste a job description to get instant AI-powered insights.</div>", unsafe_allow_html=True)

    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""
    if "cv_name" not in st.session_state:
        st.session_state.cv_name = None
    if "cv_bytes" not in st.session_state:
        st.session_state.cv_bytes = None

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="dotted-box" style="padding:22px;">
                <div style="font-size:18px; font-weight:900; margin-bottom:12px; color:#0F172A;">Upload CV</div>
            """,
            unsafe_allow_html=True
        )

        up = st.file_uploader(
            "Click to upload or drag and drop",
            type=["pdf", "docx", "txt"],
            key="cv_uploader",
            label_visibility="collapsed"
        )

        if up is not None:
            st.session_state.cv_name = up.name
            st.session_state.cv_bytes = up.read()

        if st.session_state.cv_name:
            st.markdown(
                f"""
                <div style="
                    margin-top:10px;
                    margin-bottom:14px;
                    padding:12px 14px;
                    background:white;
                    border:1px solid rgba(15,23,42,0.08);
                    border-radius:14px;
                    font-weight:700;
                    color:#0F172A;
                ">
                    Uploaded CV: {st.session_state.cv_name}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='font-size:18px; font-weight:900; margin-top:8px; margin-bottom:10px; color:#0F172A;'>Job Description</div>", unsafe_allow_html=True)

        st.session_state.jd_text = st.text_area(
            "Paste the job description here...",
            height=260,
            value=st.session_state.jd_text,
            key="jd_area"
        )

        st.markdown('<div class="primary">', unsafe_allow_html=True)
        run = st.button("✨ Analyze Match", use_container_width=True, key="analyze_btn")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='inside-box-title'>Match Score</div>", unsafe_allow_html=True)

        if run:
            if not st.session_state.cv_bytes or not st.session_state.cv_name:
                st.error("Upload a CV first.")
            elif not st.session_state.jd_text.strip():
                st.error("Paste a job description first.")
            else:
                cv_text = extract_text(st.session_state.cv_name, st.session_state.cv_bytes)
                final, sim, sk, matched, missing = compute_all(cv_text, st.session_state.jd_text)
                p = max(0, min(100, int(final)))

                st.markdown(
                    f"""
                    <div class="ring" style="--p:{p}%;">
                        <div class="ring-inner">
                            <div class="score">{p}%</div>
                            <div class="score-sub">Good Match</div>
                        </div>
                    </div>
                    <div style="text-align:center; color:#64748B; font-size:14px; margin-bottom:18px;">
                        Your CV matches {p}% of job requirements
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<div class='inside-box-title'>Matched Skills</div>", unsafe_allow_html=True)
                if matched:
                    st.markdown("".join([f"<span class='chip chip-green'>{s}</span>" for s in matched]), unsafe_allow_html=True)
                else:
                    st.write("None detected")

                st.markdown("<div class='inside-box-title' style='margin-top:20px;'>Missing Skills</div>", unsafe_allow_html=True)
                if missing:
                    st.markdown("".join([f"<span class='chip chip-orange'>{s}</span>" for s in missing]), unsafe_allow_html=True)
                else:
                    st.write("None detected")

                st.markdown("<div class='inside-box-title' style='margin-top:20px;'>AI Recommendations</div>", unsafe_allow_html=True)
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
            st.markdown(
                """
                <div style="text-align:center; padding:90px 20px;">
                    <div style="
                        width:120px;
                        height:70px;
                        margin:0 auto 18px;
                        border-radius:999px;
                        background:#F1F5F9;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:38px;
                        color:#94A3B8;
                    ">↗</div>
                    <div style="font-size:28px; font-weight:900; color:#334155; margin-bottom:10px;">
                        Ready to Analyze
                    </div>
                    <div style="color:#64748B; font-size:15px; max-width:340px; margin:0 auto; line-height:1.7;">
                        Upload your CV and paste a job description, then click "Analyze Match" to see your results
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_about():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("<div class='title' style='font-size:46px;'>About</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card about-big">
            <b>AI Match</b> compares a candidate CV with a job description to produce a match score and skill gap analysis.
            <br/><br/>
            <b>Methods</b><br/>
            • Text extraction (PDF/DOCX/TXT)<br/>
            • TF-IDF vectorization + Cosine similarity<br/>
            • Keyword-based skill extraction<br/>
            • Combined scoring (skills-weighted)
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

def page_contact():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("<div class='title' style='font-size:42px; text-align:center;'>Get In Touch</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle' style='text-align:center; max-width:700px; margin:0 auto 30px;'>Have questions or feedback? We'd love to hear from you. Send us a message and we'll respond as soon as possible.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.3], gap="large")

    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="inside-box-title">Email</div>
                <div>support@aimatch.com</div>
                <div>info@aimatch.com</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card">
                <div class="inside-box-title">Phone</div>
                <div>+966 12 345 6789</div>
                <div>Mon-Fri 9am-6pm EST</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card">
                <div class="inside-box-title">Office</div>
                <div>123 Tech Street</div>
                <div>Abha, Aseer</div>
                <div>Kingdom of Saudi Arabia</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='inside-box-title'>Send Us a Message</div>", unsafe_allow_html=True)

        st.text_input("Name", key="c_name")
        st.text_input("Email", key="c_email")
        st.text_input("Subject", key="c_subject")
        st.text_area("Message", height=160, key="c_msg")

        st.markdown('<div class="primary">', unsafe_allow_html=True)
        st.button("Send Message", use_container_width=True, key="send_msg")
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption("Demo UI only (no email sending yet).")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ROUTER ----------------
if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Dashboard":
    page_dashboard()
elif st.session_state.page == "About":
    page_about()
elif st.session_state.page == "Contact":
    page_contact()
else:
    st.session_state.page = "Home"
    page_home()
