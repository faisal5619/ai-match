import re
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import BytesIO
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from agents.cv_agent import analyze_cv
from agents.job_agent import load_jobs
from agents.match_agent import analyze_all_jobs
from agents.recommendation_agent import generate_recommendations
from database.db import init_db, insert_resume, insert_match
from auth_db import init_auth_db, register_user, login_user, save_user_cv, load_user_cv, delete_user_cv, load_user_profile, save_user_profile, fetch_github_skills

# ---------------- INIT DB ----------------
init_db()
init_auth_db()

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Match", page_icon="🧠", layout="wide")

def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

AI_ICON = img_to_base64("ai_icon.png")

# ---------------- SESSION DEFAULTS ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "login"
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- HELPERS ----------------
def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))

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
    "computer science", "programming", "oop", "object oriented programming",
    "data structures", "algorithms", "microsoft office", "word", "excel", "powerpoint",
    "teamwork", "collaboration", "problem solving", "time management",
    "communication", "english", "arabic"
}

def find_skills(text: str) -> set:
    t = clean_text(text)
    found = set()
    for s in SKILLS:
        if re.search(r"\b" + re.escape(s) + r"\b", t):
            found.add(s)
    return found

# ---------------- EMAIL ----------------
def send_interview_email(to_email, candidate_name, score, job_title, company):
    sender_email = st.secrets["EMAIL_ADDRESS"]
    sender_password = st.secrets["EMAIL_PASSWORD"]
    subject = "Interview Invitation - AI Match"
    body = f"""
Dear {candidate_name},

Thank you for using AI Match.

We are pleased to inform you that your CV achieved a matching score of {score:.2f}% for the position of {job_title} at {company}.

Since your score is above our required threshold, you have been shortlisted for the next stage of the recruitment process.

Our team will contact you soon regarding the interview details.

Best regards,
AI Match Recruitment Team
"""
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, None
    except Exception as e:
        return False, str(e)

def send_recruitment_email(to_email, candidate_name, candidate_email, score, job_title, company, matched_skills, missing_skills, cv_name):
    sender_email = st.secrets["EMAIL_ADDRESS"]
    sender_password = st.secrets["EMAIL_PASSWORD"]
    subject = "New Shortlisted Candidate - AI Match"
    matched_text = ", ".join(matched_skills) if matched_skills else "None"
    missing_text = ", ".join(missing_skills) if missing_skills else "None"
    body = f"""
Dear Recruitment Team,

A new candidate has been shortlisted through AI Match.

Candidate Name: {candidate_name}
Candidate Email: {candidate_email}
CV File: {cv_name}

Job Title: {job_title}
Company: {company}
Match Score: {score:.2f}%

Matched Skills: {matched_text}
Missing Skills: {missing_text}

Please review the candidate for the next recruitment stage.

Best regards,
AI Match System
"""
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, None
    except Exception as e:
        return False, str(e)

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
        background: transparent;
        backdrop-filter: none;
        border-bottom: none;
        box-shadow: none;
    }}

    .nav-inner {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 10px 20px;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: 14px;
        font-weight: 900;
        font-size: 26px;
    }}

    .logo {{
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: #2563EB;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 22px rgba(37,99,235,0.30);
        overflow: hidden;
    }}

    .logo-img {{
        width: 30px;
        height: 30px;
        object-fit: contain;
        display: block;
    }}

    .navlinks .stButton > button {{
        background: #FFFFFF !important;
        border: 1px solid rgba(15,23,42,0.06) !important;
        color: #475569 !important;
        font-weight: 700 !important;
        padding: 10px 12px !important;
        border-radius: 12px !important;
        height: auto !important;
        box-shadow: 0 8px 18px rgba(2,6,23,0.06) !important;
        transition: all 0.2s ease !important;
    }}

    .navlinks .stButton > button:hover {{
        background: rgba(37,99,235,0.08) !important;
        color: #2563EB !important;
        border: 1px solid rgba(37,99,235,0.10) !important;
    }}

    .active .stButton > button {{
        background: rgba(37,99,235,0.14) !important;
        color: #2563EB !important;
        border: 1px solid rgba(37,99,235,0.18) !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        box-shadow: none !important;
    }}

    .card {{
        background: #FFFFFF;
        border: none;
        border-radius: 20px;
        box-shadow: 0 14px 34px rgba(2,6,23,0.10), 0 3px 10px rgba(2,6,23,0.05);
        padding: 24px;
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

    textarea {{
        color: #0F172A !important;
        background: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid rgba(15,23,42,0.10) !important;
    }}

    input {{
        color: #0F172A !important;
    }}

    button[kind="primary"] {{
        background: #2563EB !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 900 !important;
        height: 38px !important;
        box-shadow: 0 10px 24px rgba(37,99,235,0.25) !important;
        transition: all 0.2s ease !important;
    }}

    button[kind="primary"]:hover {{
        background: #1D4ED8 !important;
        color: white !important;
        box-shadow: 0 14px 28px rgba(29,78,216,0.35) !important;
        transform: translateY(-1px);
    }}

    button[kind="secondary"] {{
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid rgba(15,23,42,0.10) !important;
        border-radius: 14px !important;
        font-weight: 900 !important;
        height: 38px !important;
        box-shadow: 0 10px 24px rgba(2,6,23,0.08) !important;
        transition: all 0.2s ease !important;
    }}

    button[kind="secondary"]:hover {{
        background: #F8FAFC !important;
        color: #0F172A !important;
        border: 1px solid rgba(15,23,42,0.14) !important;
        box-shadow: 0 14px 28px rgba(2,6,23,0.12) !important;
        transform: translateY(-1px);
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

    .chip-green {{ background: #DCFCE7; color: #166534; }}
    .chip-orange {{ background: #FFEDD5; color: #9A3412; }}

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

    .inside-box-title {{
        font-size: 18px;
        font-weight: 900;
        margin-bottom: 10px;
        color: #0F172A;
    }}

    .section-gap {{ margin-top: 18px; }}
    .muted {{ color: #64748B; font-size: 14px; }}

    /* ── AUTH STYLES ── */
    .auth-logo {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        margin-bottom: 30px;
        margin-top: 40px;
    }}
    .auth-logo-icon {{
        width: 54px;
        height: 54px;
        border-radius: 16px;
        background: #2563EB;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 26px rgba(37,99,235,0.35);
        overflow: hidden;
    }}
    .auth-logo-icon img {{
        width: 34px;
        height: 34px;
        object-fit: contain;
    }}
    .auth-logo-name {{
        font-size: 30px;
        font-weight: 950;
        color: #0F172A;
        letter-spacing: -0.02em;
    }}
    .auth-title {{
        font-size: 24px;
        font-weight: 900;
        color: #0F172A;
        text-align: center;
        margin-bottom: 4px;
    }}
    .auth-subtitle {{
        font-size: 14px;
        color: #64748B;
        text-align: center;
        margin-bottom: 22px;
        line-height: 1.6;
    }}
    .auth-tabs {{
        display: flex;
        background: #F1F5F9;
        border-radius: 14px;
        padding: 4px;
        margin-bottom: 22px;
        gap: 4px;
    }}
    .auth-tab {{
        flex: 1;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 14px;
        color: #64748B;
    }}
    .auth-tab.active {{
        background: #FFFFFF;
        color: #2563EB;
        box-shadow: 0 4px 12px rgba(2,6,23,0.08);
    }}
    .auth-divider {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 16px 0;
        color: #94A3B8;
        font-size: 13px;
        font-weight: 600;
    }}
    .auth-divider::before, .auth-divider::after {{
        content: "";
        flex: 1;
        height: 1px;
        background: rgba(15,23,42,0.08);
    }}

    /* Auth tab buttons — override default secondary style */
    [data-testid="column"] button[kind="secondary"] {{
        background: #F1F5F9 !important;
        color: #64748B !important;
        border: none !important;
        box-shadow: none !important;
        font-weight: 800 !important;
        border-radius: 12px !important;
    }}
    [data-testid="column"] button[kind="secondary"]:hover {{
        background: #E2E8F0 !important;
        color: #475569 !important;
        box-shadow: none !important;
        transform: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# ════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE
# ════════════════════════════════════════════════════════════════════════════
def page_auth():
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:

        # Logo
        st.markdown(
            f"""
            <div class="auth-logo">
                <div class="auth-logo-icon">
                    <img src="data:image/png;base64,{AI_ICON}" />
                </div>
                <div class="auth-logo-name">AI Match</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tab switcher — clean single row
        login_active = st.session_state.auth_tab == "login"
        tc1, tc2 = st.columns(2)
        with tc1:
            if st.button(
                "Sign In",
                key="tab_login",
                use_container_width=True,
                type="primary" if login_active else "secondary"
            ):
                st.session_state.auth_tab = "login"
                st.rerun()
        with tc2:
            if st.button(
                "Create Account",
                key="tab_reg",
                use_container_width=True,
                type="primary" if not login_active else "secondary"
            ):
                st.session_state.auth_tab = "register"
                st.rerun()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── LOGIN ──────────────────────────────────────────────────────────
        if st.session_state.auth_tab == "login":
            st.markdown(
                '<div class="auth-title">Welcome back 👋</div>'
                '<div class="auth-subtitle">Sign in to your AI Match account to continue</div>',
                unsafe_allow_html=True
            )

            login_email    = st.text_input("Email address", key="login_email", placeholder="you@example.com")
            login_password = st.text_input("Password", key="login_password", type="password", placeholder="Your password")

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if st.button("Sign In →", key="login_btn", type="primary", use_container_width=True):
                if not login_email.strip():
                    st.error("Please enter your email address.")
                elif not is_valid_email(login_email):
                    st.error("Please enter a valid email address.")
                elif not login_password:
                    st.error("Please enter your password.")
                else:
                    ok, result = login_user(login_email, login_password)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.user_name = result
                        st.session_state.user_email = login_email.strip().lower()
                        st.session_state.cv_loaded_from_db = False
                        st.rerun()
                    else:
                        st.error(result)

            st.markdown('<div class="auth-divider">New to AI Match?</div>', unsafe_allow_html=True)

            if st.button("Create a free account", key="go_register", use_container_width=True):
                st.session_state.auth_tab = "register"
                st.rerun()

        # ── REGISTER ───────────────────────────────────────────────────────
        else:
            st.markdown(
                '<div class="auth-title">Create your account ✨</div>'
                '<div class="auth-subtitle">Join AI Match and find your best job matches instantly</div>',
                unsafe_allow_html=True
            )

            reg_name     = st.text_input("Full name", key="reg_name", placeholder="Faisal Alhudaithy")
            reg_email    = st.text_input("Email address", key="reg_email", placeholder="you@example.com")
            reg_password = st.text_input("Password", key="reg_password", type="password", placeholder="At least 8 characters")
            reg_confirm  = st.text_input("Confirm password", key="reg_confirm", type="password", placeholder="Repeat your password")

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if st.button("Create Account →", key="register_btn", type="primary", use_container_width=True):
                if not reg_name.strip():
                    st.error("Please enter your full name.")
                elif not reg_email.strip():
                    st.error("Please enter your email address.")
                elif not is_valid_email(reg_email):
                    st.error("Please enter a valid email address.")
                elif len(reg_password) < 8:
                    st.error("Password must be at least 8 characters.")
                elif reg_password != reg_confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, err = register_user(reg_name, reg_email, reg_password)
                    if ok:
                        st.success("Account created successfully! You can now sign in.")
                        st.session_state.auth_tab = "login"
                        st.rerun()
                    else:
                        st.error(err)

            st.markdown('<div class="auth-divider">Already have an account?</div>', unsafe_allow_html=True)

            if st.button("Sign in instead", key="go_login", use_container_width=True):
                st.session_state.auth_tab = "login"
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  SHOW AUTH WALL IF NOT LOGGED IN  — stops everything below from running
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    page_auth()
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
#  NAV / ROUTING  (only reached when logged in)
# ════════════════════════════════════════════════════════════════════════════
PAGES = ["Home", "Dashboard", "Profile", "About", "Contact"]

qp = st.query_params.get("page")
if qp in PAGES:
    st.session_state.page = qp

def set_page(p: str):
    st.session_state.page = p
    st.query_params["page"] = p
    st.rerun()

# ---------------- NAVBAR ----------------
first_name = st.session_state.user_name.split()[0] if st.session_state.user_name else "User"
avatar_letter = first_name[0].upper()
current_page = st.session_state.page

def nav_active(page):
    return "nav-active" if current_page == page else ""

st.markdown(
    f"""
    <style>
    .navbar {{
        display: flex;
        align-items: center;
        background: #FFFFFF;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(2,6,23,0.07);
        padding: 10px 20px;
        margin-bottom: 24px;
        gap: 8px;
    }}
    .nav-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 900;
        font-size: 20px;
        color: #0F172A;
        margin-right: 20px;
        flex-shrink: 0;
    }}
    .nav-logo-box {{
        width: 38px; height: 38px;
        border-radius: 11px;
        background: #2563EB;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 12px rgba(37,99,235,0.30);
        overflow: hidden;
        flex-shrink: 0;
    }}
    .nav-logo-box img {{ width: 24px; height: 24px; object-fit: contain; }}
    .nav-spacer {{ flex: 1; }}
    .nav-avatar {{
        width: 34px; height: 34px;
        border-radius: 50%;
        background: #2563EB;
        color: white;
        font-weight: 900; font-size: 14px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 3px 10px rgba(37,99,235,0.28);
        flex-shrink: 0;
    }}
    .nav-uname {{
        font-weight: 800; font-size: 13px; color: #0F172A; margin-right: 4px;
    }}
    </style>

    """,
    unsafe_allow_html=True
)

# Actual clickable nav — one clean row of columns
nc1, nc2, nc3, nc4, nc5, nc6, nc7, nc8 = st.columns([2, 0.85, 1.1, 0.85, 1.0, 0.5, 1.1, 0.85], vertical_alignment="center")

with nc1:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:36px;height:36px;border-radius:11px;background:#2563EB;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(37,99,235,0.28);overflow:hidden;">
                <img src="data:image/png;base64,{AI_ICON}" style="width:22px;height:22px;object-fit:contain;" />
            </div>
            <span style="font-weight:900;font-size:19px;color:#0F172A;">AI Match</span>
        </div>
        """, unsafe_allow_html=True
    )

def nav_btn(col, label, target):
    with col:
        is_active = current_page == target
        bg = "rgba(37,99,235,0.10)" if is_active else "transparent"
        color = "#2563EB" if is_active else "#475569"
        fw = "800" if is_active else "700"
        st.markdown(
            f'<div style="margin:-4px 0;">',
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<style>
            div[data-testid="stButton"] button[kind="secondary"]#btn_{target} {{
                background: {bg} !important; color: {color} !important; font-weight: {fw} !important;
            }}
            </style>""", unsafe_allow_html=True
        )
        if st.button(label, use_container_width=True, key=f"nav_{target}"):
            set_page(target)
        st.markdown("</div>", unsafe_allow_html=True)

nav_btn(nc2, "Home", "Home")
nav_btn(nc3, "Dashboard", "Dashboard")
nav_btn(nc4, "About", "About")
nav_btn(nc5, "Contact", "Contact")

with nc6:
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:center;height:38px;width:34px;border-radius:50%;background:#2563EB;color:white;font-weight:900;font-size:14px;box-shadow:0 3px 10px rgba(37,99,235,0.28);margin:0 auto;">{avatar_letter}</div>',
        unsafe_allow_html=True
    )

nav_btn(nc7, f"{first_name} · Profile", "Profile")

with nc8:
    st.markdown('<style>.logout-btn button{background:#FEF2F2!important;color:#DC2626!important;border:1px solid rgba(220,38,38,0.2)!important;font-weight:700!important;border-radius:10px!important;box-shadow:none!important;}</style>', unsafe_allow_html=True)
    st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
    if st.button("Logout", use_container_width=True, key="nav_logout"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.user_email = ""
        st.session_state.page = "Home"
        st.session_state.auth_tab = "login"
        st.session_state.cv_name = None
        st.session_state.cv_bytes = None
        st.session_state.cv_loaded_from_db = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  PAGES
# ════════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<span class="big-pill">AI-Powered Recruitment</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="title" style="font-size:58px;">AI-Powered CV<br/>Screening & Job<br/>Matching</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="subtitle" style="font-size:18px; max-width:620px;">Upload your CV and compare it instantly with job descriptions using NLP. Get match scores, missing skills, and recommendations in seconds.</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns([0.28, 0.28, 0.44], gap="small")
        with c1:
            if st.button("Try Now →", key="home_try", type="primary", use_container_width=True):
                set_page("Dashboard")
        with c2:
            if st.button("Learn More", key="home_learn", type="secondary", use_container_width=True):
                set_page("About")

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            '<span class="big-pill">Free to use</span><span class="big-pill">Instant results</span><span class="big-pill">AI-powered</span>',
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            f"""
<div style="background: linear-gradient(135deg,#E0EAFF,#F5F8FF); border: 1px solid rgba(15,23,42,0.08); border-radius: 24px; box-shadow: 0 10px 28px rgba(2,6,23,0.06); padding: 28px; min-height: 420px;">
<div style="width:200px; height:200px; border-radius:32px; background:white; display:flex; align-items:center; justify-content:center; margin:0 auto 28px auto; box-shadow:0 14px 28px rgba(15,23,42,0.08);">
    <img src="data:image/png;base64,{AI_ICON}" style="width:130px; height:130px; object-fit:contain; filter:drop-shadow(0 12px 24px rgba(37,99,235,0.18));" />
</div>
<div style="display:flex; gap:16px; margin-top:10px;">
<div style="background:white; border-radius:18px; padding:16px; flex:1; box-shadow:0 12px 22px rgba(2,6,23,0.08); display:flex; align-items:center; gap:12px;">
<div style="width:36px; height:36px; border-radius:10px; background:#16A34A; display:flex; align-items:center; justify-content:center; color:white; font-size:18px;">✓</div>
<div>
<div style="font-size:12px; color:#64748B; font-weight:700;">Match Score</div>
<div style="font-size:26px; font-weight:900; color:#16A34A;">87%</div>
</div>
</div>
<div style="background:white; border-radius:18px; padding:16px; flex:1; box-shadow:0 12px 22px rgba(2,6,23,0.08); display:flex; align-items:center; gap:12px;">
<div style="width:36px; height:36px; border-radius:10px; background:#4F46E5; display:flex; align-items:center; justify-content:center; color:white; font-size:18px;">⚡</div>
<div>
<div style="font-size:12px; color:#64748B; font-weight:700;">Analysis</div>
<div style="font-size:16px; font-weight:900;">2 seconds</div>
</div>
</div>
</div>
</div>
""",
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
    st.markdown("<div class='subtitle' style='font-size:17px;'>Upload your CV and let the system automatically find the best matching jobs for you.</div>", unsafe_allow_html=True)

    # ── Session defaults ──
    if "cv_name" not in st.session_state:
        st.session_state.cv_name = None
    if "cv_bytes" not in st.session_state:
        st.session_state.cv_bytes = None
    if "candidate_name" not in st.session_state:
        st.session_state.candidate_name = st.session_state.user_name
    if "candidate_email" not in st.session_state:
        st.session_state.candidate_email = st.session_state.user_email
    if "cv_loaded_from_db" not in st.session_state:
        st.session_state.cv_loaded_from_db = False

    # ── Auto-load saved CV from DB on first visit ──
    if not st.session_state.cv_loaded_from_db:
        saved_name, saved_bytes = load_user_cv(st.session_state.user_email)
        if saved_name and saved_bytes:
            st.session_state.cv_name = saved_name
            st.session_state.cv_bytes = saved_bytes
        st.session_state.cv_loaded_from_db = True

    left, right = st.columns([1, 1], gap="large")

    with left:
        # ── Quick Preferences ──
        with st.container(border=True):
            st.markdown('<div style="font-size:18px; font-weight:900; margin-bottom:12px; color:#0F172A;">⚡ Job Preferences</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:12px; color:#64748B; margin-bottom:10px;">Adjust your preferences to improve match results</div>', unsafe_allow_html=True)

            dash_prefs = load_user_profile(st.session_state.user_email)

            location_options = ["", "Riyadh", "Jeddah", "Abha", "Dammam", "Medina", "Remote", "Any"]
            loc_index = location_options.index(dash_prefs["preferred_location"]) if dash_prefs["preferred_location"] in location_options else 0

            job_type_options = ["", "Full-time", "Part-time", "Internship", "Remote", "Freelance"]
            jt_index = job_type_options.index(dash_prefs["job_type"]) if dash_prefs["job_type"] in job_type_options else 0

            field_options = ["", "Artificial Intelligence", "Web Development", "Data Science", "Cybersecurity",
                             "Mobile Development", "Cloud Computing", "Software Engineering", "DevOps", "Other"]
            fi_index = field_options.index(dash_prefs["field_of_interest"]) if dash_prefs["field_of_interest"] in field_options else 0

            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                dash_location = st.selectbox("📍 Location", location_options, index=loc_index,
                    key="dash_location", format_func=lambda x: "Any" if x == "" else x)
            with pc2:
                dash_job_type = st.selectbox("💼 Job Type", job_type_options, index=jt_index,
                    key="dash_job_type", format_func=lambda x: "Any" if x == "" else x)
            with pc3:
                dash_field = st.selectbox("🎯 Field", field_options, index=fi_index,
                    key="dash_field", format_func=lambda x: "Any" if x == "" else x)

            if st.button("Save Preferences", key="dash_save_prefs", use_container_width=True):
                save_user_profile(st.session_state.user_email, dash_prefs["full_name"], dash_location, dash_job_type, dash_field)
                st.toast("Preferences saved!", icon="✅")
                st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div style="font-size:18px; font-weight:900; margin-bottom:12px; color:#0F172A;">Upload CV</div>', unsafe_allow_html=True)

            st.text_input("Full Name", key="candidate_name")
            st.text_input("Email Address", key="candidate_email")

            up = st.file_uploader(
                "Click to upload or drag and drop",
                type=["pdf", "docx", "txt"],
                key="cv_uploader",
                label_visibility="collapsed"
            )

            if up is not None:
                new_bytes = up.read()
                st.session_state.cv_name = up.name
                st.session_state.cv_bytes = new_bytes
                # Save new CV to database immediately
                save_user_cv(st.session_state.user_email, up.name, new_bytes)
                st.toast("CV saved to your account!", icon="✅")

            if st.session_state.cv_name:
                # Show green banner if loaded from DB, white if just uploaded
                is_saved = st.session_state.cv_loaded_from_db and up is None
                banner_bg = "#DCFCE7" if is_saved else "white"
                banner_border = "rgba(22,163,74,0.20)" if is_saved else "rgba(15,23,42,0.08)"
                banner_label = "💾 Saved CV: " if is_saved else "📄 Uploaded CV: "
                st.markdown(
                    f"""
                    <div style="margin-top:10px; margin-bottom:14px; padding:12px 14px; background:{banner_bg}; border:1px solid {banner_border}; border-radius:14px; font-weight:700; color:#0F172A;">
                        {banner_label}{st.session_state.cv_name}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
            run = st.button("✨ Analyze Match", use_container_width=True, key="analyze_btn", type="primary")
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown("<div class='inside-box-title'>Match Score</div>", unsafe_allow_html=True)

            if run:
                if not st.session_state.candidate_name.strip():
                    st.error("Please enter your full name.")
                elif not st.session_state.candidate_email.strip():
                    st.error("Please enter your email address.")
                elif not is_valid_email(st.session_state.candidate_email):
                    st.error("Please enter a valid email address.")
                elif not st.session_state.cv_bytes or not st.session_state.cv_name:
                    st.error("Upload a CV first.")
                else:
                    cv_data = analyze_cv(st.session_state.cv_name, st.session_state.cv_bytes)

                    # ── Enrich skills with GitHub ──
                    github_username = load_user_profile(st.session_state.user_email).get("github_username", "")
                    if github_username.strip():
                        with st.spinner("🐙 Fetching your GitHub skills..."):
                            github_skills = fetch_github_skills(github_username)
                        if github_skills:
                            # Merge GitHub skills into CV skills
                            merged_skills = list(set(cv_data["skills"] + github_skills))
                            cv_data["skills"] = merged_skills
                            # Also append to text for TF-IDF matching
                            cv_data["text"] += " " + " ".join(github_skills)
                            st.toast(f"🐙 Added {len(github_skills)} skills from GitHub!", icon="✅")

                    resume_id = insert_resume(st.session_state.cv_name, cv_data["text"], cv_data["skills"])
                    jobs = load_jobs()
                    match_results = analyze_all_jobs(cv_data, jobs)

                    # ── Apply preference boosts ──
                    prefs = load_user_profile(st.session_state.user_email)
                    pref_location = prefs.get("preferred_location", "").lower().strip()
                    pref_job_type = prefs.get("job_type", "").lower().strip()
                    pref_field    = prefs.get("field_of_interest", "").lower().strip()

                    FIELD_KEYWORDS = {
                        "artificial intelligence": ["ai", "artificial intelligence", "machine learning", "nlp", "deep learning"],
                        "web development": ["web", "frontend", "backend", "html", "css", "javascript", "react"],
                        "data science": ["data", "analyst", "analytics", "sql", "python", "visualization"],
                        "cybersecurity": ["security", "cyber", "penetration", "firewall", "soc"],
                        "mobile development": ["mobile", "android", "ios", "flutter", "swift"],
                        "cloud computing": ["cloud", "aws", "azure", "devops", "kubernetes"],
                        "software engineering": ["software", "developer", "engineer", "programming"],
                        "devops": ["devops", "ci/cd", "docker", "kubernetes", "pipeline"],
                    }

                    def preference_boost(result):
                        score = result["final_score"]
                        job_text = (result.get("title","") + " " + result.get("description","") + " " + result.get("location","")).lower()

                        # Location boost: +10 if matches
                        if pref_location and pref_location not in ["any", "remote", ""]:
                            if pref_location in job_text:
                                score += 10

                        # Job type boost: +8 if matches
                        if pref_job_type and pref_job_type not in [""]:
                            if pref_job_type in job_text:
                                score += 8

                        # Field boost: +12 if keyword matches
                        if pref_field:
                            keywords = FIELD_KEYWORDS.get(pref_field.lower(), [pref_field])
                            if any(kw in job_text for kw in keywords):
                                score += 12

                        return min(score, 100)  # cap at 100

                    if pref_location or pref_job_type or pref_field:
                        for r in match_results:
                            r["boosted_score"] = preference_boost(r)
                        match_results.sort(key=lambda x: x["boosted_score"], reverse=True)
                    else:
                        for r in match_results:
                            r["boosted_score"] = r["final_score"]

                    recommendations = generate_recommendations(match_results, top_n=3)

                    # Show active preferences as pills
                    active_prefs = []
                    if pref_location and pref_location != "any":
                        active_prefs.append(f"📍 {prefs['preferred_location']}")
                    if pref_job_type:
                        active_prefs.append(f"💼 {prefs['job_type']}")
                    if pref_field:
                        active_prefs.append(f"🎯 {prefs['field_of_interest']}")

                    if active_prefs:
                        pills_html = " ".join([f'<span class="big-pill" style="font-size:12px;padding:6px 12px;">{p}</span>' for p in active_prefs])
                        st.markdown(
                            f'<div style="margin-bottom:12px;"><div style="font-size:12px;font-weight:700;color:#64748B;margin-bottom:6px;">Applied preferences:</div>{pills_html}</div>',
                            unsafe_allow_html=True
                        )

                    if github_username.strip() and github_skills:
                        gh_pills = " ".join([f'<span class="chip chip-green">{s}</span>' for s in github_skills[:8]])
                        st.markdown(
                            f'<div style="margin-bottom:14px;"><div style="font-size:12px;font-weight:700;color:#166534;margin-bottom:6px;">🐙 Skills from GitHub:</div>{gh_pills}</div>',
                            unsafe_allow_html=True
                        )

                    if match_results:
                        top_result = match_results[0]
                        p = max(0, min(100, int(top_result["final_score"])))

                        email_sent = False
                        email_error = None

                        if top_result["final_score"] >= 60:
                            email_sent, email_error = send_interview_email(
                                to_email=st.session_state.candidate_email,
                                candidate_name=st.session_state.candidate_name,
                                score=top_result["final_score"],
                                job_title=top_result["title"],
                                company=top_result["company"]
                            )
                            send_recruitment_email(
                                to_email=st.secrets["RECRUITMENT_EMAIL"],
                                candidate_name=st.session_state.candidate_name,
                                candidate_email=st.session_state.candidate_email,
                                score=top_result["final_score"],
                                job_title=top_result["title"],
                                company=top_result["company"],
                                matched_skills=top_result["matched_skills"],
                                missing_skills=top_result["missing_skills"],
                                cv_name=st.session_state.cv_name
                            )

                        st.markdown(
                            f"""
                            <div class="ring" style="--p:{p}%;">
                                <div class="ring-inner">
                                    <div class="score">{p}%</div>
                                    <div class="score-sub">Top Match</div>
                                </div>
                            </div>
                            <div style="text-align:center; color:#64748B; font-size:14px; margin-bottom:18px;">
                                Best job match found automatically from available jobs
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        if top_result["final_score"] >= 60:
                            if email_sent:
                                st.success(f"Congratulations {st.session_state.candidate_name}! Your score is above 60%, and an interview email has been sent to {st.session_state.candidate_email}.")
                            else:
                                st.error(f"Email failed: {email_error}")
                        else:
                            st.info("Your score is below 60%, so no interview email was sent.")

                        st.markdown("<div class='inside-box-title'>Top Recommended Jobs</div>", unsafe_allow_html=True)

                        for rec in recommendations:
                            st.markdown(
                                f"""
                                <div style="border:1px solid rgba(15,23,42,0.08); border-radius:16px; padding:16px; margin-bottom:14px; background:white;">
                                    <div style="font-size:18px; font-weight:900; color:#0F172A;">{rec['title']}</div>
                                    <div style="color:#475569; font-size:14px; margin-bottom:6px;">{rec['company']} • {rec['location']}</div>
                                    <div style="margin-bottom:8px;">
                                        <span class="chip chip-green">{rec['fit_label']}</span>
                                        <span class="chip chip-green">{rec['final_score']}%</span>
                                    </div>
                                    <div style="color:#334155; font-size:14px; line-height:1.7; margin-bottom:8px;">{rec['reason']}</div>
                                    <div style="color:#9A3412; font-size:14px; line-height:1.7;">{rec['tip']}</div>
                                    <div style="margin-top:10px;"><a href="{rec['url']}" target="_blank">View Job</a></div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.markdown("<div class='inside-box-title' style='margin-top:20px;'>Detailed Match Breakdown</div>", unsafe_allow_html=True)

                        for result in match_results[:3]:
                            insert_match(
                                resume_id=resume_id,
                                job_title=result["title"],
                                company=result["company"],
                                location=result["location"],
                                similarity_score=result["similarity_score"],
                                skill_score=result["skill_score"],
                                final_score=result["final_score"],
                                matched_skills=result["matched_skills"],
                                missing_skills=result["missing_skills"]
                            )
                            with st.expander(f"{result['title']} at {result['company']} — {result['final_score']}%"):
                                st.write(f"**Location:** {result['location']}")
                                st.write(f"**Similarity Score:** {result['similarity_score']}%")
                                st.write(f"**Skill Score:** {result['skill_score']}%")
                                st.write(f"**Matched Skills:** {', '.join(result['matched_skills']) if result['matched_skills'] else 'None'}")
                                st.write(f"**Missing Skills:** {', '.join(result['missing_skills']) if result['missing_skills'] else 'None'}")
                    else:
                        st.warning("No jobs were available for matching.")
            else:
                st.markdown(
                    """
                    <div style="text-align:center; padding:90px 20px;">
                        <div style="width:120px; height:70px; margin:0 auto 18px; border-radius:999px; background:#F1F5F9; display:flex; align-items:center; justify-content:center; font-size:38px; color:#94A3B8;">↗</div>
                        <div style="font-size:28px; font-weight:900; color:#334155; margin-bottom:10px;">Ready to Analyze</div>
                        <div style="color:#64748B; font-size:15px; max-width:340px; margin:0 auto; line-height:1.7;">Upload your CV and click "Analyze Match" to see automatic job recommendations</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)


def page_about():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("<div class='title' style='font-size:46px;'>About</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.write("**AI Match** is an intelligent multi-agent system that analyzes CVs and automatically matches them with relevant job opportunities.")
        st.markdown("### How the system works")
        st.write("• CV Agent extracts text and detects skills from uploaded resumes")
        st.write("• Job Agent retrieves job listings from the web using web scraping")
        st.write("• Vector Agent creates semantic embeddings and performs similarity search using FAISS")
        st.write("• Match Agent evaluates similarity and skill compatibility")
        st.write("• Recommendation Agent explains results and suggests improvements")
        st.markdown("### Technologies used")
        st.write("• Natural Language Processing (NLP)")
        st.write("• Sentence Transformers for semantic embeddings")
        st.write("• FAISS vector database for similarity search")
        st.write("• Web scraping for real-time job retrieval")
        st.write("• Streamlit for the interactive web interface")
        st.write("The platform automatically analyzes candidate profiles, retrieves relevant job postings, and recommends the most suitable opportunities.")

    st.markdown("</div>", unsafe_allow_html=True)


def page_contact():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("<div class='title' style='font-size:42px; text-align:center;'>Get In Touch</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle' style='text-align:center; max-width:700px; margin:0 auto 30px;'>Have questions or feedback? We'd love to hear from you.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.3], gap="large")

    with c1:
        st.markdown('<div class="card"><div class="inside-box-title">Email</div><div>support@aimatch.com</div><div>info@aimatch.com</div></div>', unsafe_allow_html=True)
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="inside-box-title">Phone</div><div>+966 12 345 6789</div><div>Mon-Fri 9am-6pm EST</div></div>', unsafe_allow_html=True)
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="inside-box-title">Office</div><div>123 Tech Street</div><div>Abha, Aseer</div><div>Kingdom of Saudi Arabia</div></div>', unsafe_allow_html=True)

    with c2:
        st.text_input("Name", key="c_name")
        st.text_input("Email", key="c_email")
        st.text_input("Subject", key="c_subject")
        st.text_area("Message", height=160, key="c_msg")
        st.button("Send Message", use_container_width=True, key="send_msg")
        st.caption("Demo UI only (no email sending yet).")

    st.markdown("</div>", unsafe_allow_html=True)




def page_profile():
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("<div class='title' style='font-size:40px;'>My Profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Manage your personal info, job preferences, and saved CV.</div>", unsafe_allow_html=True)

    email = st.session_state.user_email
    profile = load_user_profile(email)

    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: Basic Info + Job Preferences ──
    with left:
        with st.container(border=True):
            st.markdown('<div class="inside-box-title">👤 Basic Information</div>', unsafe_allow_html=True)

            new_name  = st.text_input("Full Name", value=profile["full_name"], key="prof_name")
            st.text_input("Email Address", value=email, disabled=True, key="prof_email")
            new_github = st.text_input("GitHub Username", value=profile.get("github_username", ""),
                key="prof_github", placeholder="e.g. faisal5619")
            if new_github.strip():
                st.markdown(
                    f'<div style="font-size:12px; color:#2563EB; margin-top:-8px; margin-bottom:4px;">🔗 github.com/{new_github.strip()}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="inside-box-title" style="margin-top:16px;">💼 Job Preferences</div>', unsafe_allow_html=True)

            location_options = ["", "Riyadh", "Jeddah", "Abha", "Dammam", "Medina", "Remote", "Any"]
            loc_index = location_options.index(profile["preferred_location"]) if profile["preferred_location"] in location_options else 0
            new_location = st.selectbox(
                "Preferred Job Location",
                options=location_options,
                index=loc_index,
                key="prof_location",
                format_func=lambda x: "Select a location..." if x == "" else x
            )

            job_type_options = ["", "Full-time", "Part-time", "Internship", "Remote", "Freelance"]
            jt_index = job_type_options.index(profile["job_type"]) if profile["job_type"] in job_type_options else 0
            new_job_type = st.selectbox(
                "Preferred Job Type",
                options=job_type_options,
                index=jt_index,
                key="prof_job_type",
                format_func=lambda x: "Select a job type..." if x == "" else x
            )

            field_options = ["", "Artificial Intelligence", "Web Development", "Data Science", "Cybersecurity",
                             "Mobile Development", "Cloud Computing", "Software Engineering", "DevOps", "Other"]
            fi_index = field_options.index(profile["field_of_interest"]) if profile["field_of_interest"] in field_options else 0
            new_field = st.selectbox(
                "Field of Interest",
                options=field_options,
                index=fi_index,
                key="prof_field",
                format_func=lambda x: "Select a field..." if x == "" else x
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("💾 Save Profile", key="save_profile_btn", type="primary", use_container_width=True):
                if not new_name.strip():
                    st.error("Full name cannot be empty.")
                else:
                    save_user_profile(email, new_name, new_location, new_job_type, new_field, new_github)
                    st.session_state.user_name = new_name
                    st.toast("Profile saved successfully!", icon="✅")
                    st.rerun()

    # ── RIGHT: CV Management ──
    with right:
        with st.container(border=True):
            st.markdown('<div class="inside-box-title">📄 My CV</div>', unsafe_allow_html=True)

            saved_name, saved_bytes = load_user_cv(email)

            if saved_name:
                st.markdown(
                    f"""
                    <div style="padding:16px; background:#DCFCE7; border:1px solid rgba(22,163,74,0.20);
                                border-radius:14px; margin-bottom:16px;">
                        <div style="font-weight:900; color:#166534; font-size:15px; margin-bottom:4px;">✅ CV Saved</div>
                        <div style="color:#166534; font-size:13px;">📎 {saved_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col_replace, col_delete = st.columns(2)
                with col_replace:
                    st.markdown("<div style='font-weight:700; font-size:13px; color:#475569; margin-bottom:6px;'>Replace CV</div>", unsafe_allow_html=True)
                    new_cv = st.file_uploader("Replace", type=["pdf", "docx", "txt"], key="profile_cv_upload", label_visibility="collapsed")
                    if new_cv is not None:
                        new_bytes = new_cv.read()
                        save_user_cv(email, new_cv.name, new_bytes)
                        st.session_state.cv_name = new_cv.name
                        st.session_state.cv_bytes = new_bytes
                        st.session_state.cv_loaded_from_db = False
                        st.toast("CV updated successfully!", icon="✅")
                        st.rerun()

                with col_delete:
                    st.markdown("<div style='font-weight:700; font-size:13px; color:#475569; margin-bottom:6px;'>Remove CV</div>", unsafe_allow_html=True)
                    if st.button("🗑️ Delete CV", key="delete_cv_btn", use_container_width=True):
                        delete_user_cv(email)
                        st.session_state.cv_name = None
                        st.session_state.cv_bytes = None
                        st.session_state.cv_loaded_from_db = False
                        st.toast("CV deleted.", icon="🗑️")
                        st.rerun()

            else:
                st.markdown(
                    """
                    <div style="padding:20px; background:#F1F5F9; border-radius:14px; text-align:center; margin-bottom:16px;">
                        <div style="font-size:32px; margin-bottom:8px;">📭</div>
                        <div style="font-weight:800; color:#475569; margin-bottom:4px;">No CV saved yet</div>
                        <div style="font-size:13px; color:#94A3B8;">Upload your CV from the Dashboard or below</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                new_cv = st.file_uploader("Upload your CV", type=["pdf", "docx", "txt"], key="profile_cv_upload")
                if new_cv is not None:
                    new_bytes = new_cv.read()
                    save_user_cv(email, new_cv.name, new_bytes)
                    st.session_state.cv_name = new_cv.name
                    st.session_state.cv_bytes = new_bytes
                    st.session_state.cv_loaded_from_db = False
                    st.toast("CV uploaded and saved!", icon="✅")
                    st.rerun()

        # Profile summary card
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="inside-box-title">📋 Profile Summary</div>', unsafe_allow_html=True)
            gh = profile.get("github_username", "")
            items = [
                ("Name", profile["full_name"] or "—"),
                ("Email", email),
                ("GitHub", f"@{gh}" if gh else "—"),
                ("Location", profile["preferred_location"] or "—"),
                ("Job Type", profile["job_type"] or "—"),
                ("Field", profile["field_of_interest"] or "—"),
                ("CV", saved_name if saved_name else "—"),
            ]
            for label, value in items:
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; padding:10px 0;
                                border-bottom:1px solid rgba(15,23,42,0.06); font-size:14px;">
                        <span style="color:#64748B; font-weight:700;">{label}</span>
                        <span style="color:#0F172A; font-weight:800;">{value}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Dashboard":
    page_dashboard()
elif st.session_state.page == "Profile":
    page_profile()
elif st.session_state.page == "About":
    page_about()
elif st.session_state.page == "Contact":
    page_contact()
else:
    st.session_state.page = "Home"
    page_home()
