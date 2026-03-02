import re
from io import BytesIO

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# TEXT EXTRACTION
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


# -----------------------
# CLEANING
# -----------------------

def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^a-z0-9+\#\.\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -----------------------
# SIMILARITY SCORE (TF-IDF)
# -----------------------

def tfidf_similarity_score(cv_text: str, jd_text: str) -> float:
    cv = clean_text(cv_text)
    jd = clean_text(jd_text)

    if len(cv) < 30 or len(jd) < 30:
        return 0.0

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    m = vec.fit_transform([cv, jd])
    sim = cosine_similarity(m[0:1], m[1:2])[0][0]

    return max(0.0, min(1.0, float(sim))) * 100.0


# -----------------------
# SKILL DETECTION
# -----------------------

SKILLS = {
    # Programming
    "java", "python", "c", "c++", "c#",
    "javascript", "typescript", "html", "css",

    # AI / Data
    "ai", "artificial intelligence", "machine learning",
    "deep learning", "nlp", "data analysis",

    # Tools
    "git", "linux", "docker", "aws", "azure",
    "api", "rest", "rest api",

    # Education / Domain
    "computer science", "software", "programming",
    "object oriented programming", "oop",
    "data structures", "algorithms",

    # Office
    "microsoft office", "word", "excel", "powerpoint",

    # Soft skills
    "teamwork", "collaboration",
    "problem solving", "time management",
    "communication", "english", "arabic"
}


def find_skills(text: str) -> set:
    t = clean_text(text)
    found = set()
    for s in SKILLS:
        pattern = r"\b" + re.escape(s) + r"\b"
        if re.search(pattern, t):
            found.add(s)
    return found


def skill_match_score(cv_text: str, jd_text: str) -> float:
    cv_sk = find_skills(cv_text)
    jd_sk = find_skills(jd_text)

    if not jd_sk:
        return 0.0

    return (len(cv_sk & jd_sk) / len(jd_sk)) * 100.0


# -----------------------
# FINAL SCORING (UPDATED WEIGHTS)
# -----------------------

def final_score(cv_text: str, jd_text: str):
    sim = tfidf_similarity_score(cv_text, jd_text)
    sk = skill_match_score(cv_text, jd_text)

    # NEW WEIGHTS (more recruiter-friendly)
    # 25% semantic similarity
    # 75% skill matching
    if sk == 0.0:
        final = sim
    else:
        final = 0.25 * sim + 0.75 * sk

    cv_sk = find_skills(cv_text)
    jd_sk = find_skills(jd_text)
    matched = sorted(cv_sk & jd_sk)
    missing = sorted(jd_sk - cv_sk)

    return final, sim, sk, matched, missing


# -----------------------
# STREAMLIT UI
# -----------------------

st.set_page_config(page_title="AI Match", page_icon="🧠", layout="wide")

st.title("CV Analysis Dashboard")
st.caption("Upload your CV and paste a job description to get instant matching results.")

left, right = st.columns([1, 1])

with left:
    cv_file = st.file_uploader("Upload CV (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    jd = st.text_area("Job Description", height=250)
    run = st.button("✨ Analyze Match", use_container_width=True)

with right:
    st.subheader("Results")

    if run:
        if not cv_file:
            st.error("Upload a CV first.")
        elif not jd.strip():
            st.error("Paste a job description first.")
        else:
            try:
                cv_text = extract_text(cv_file.name, cv_file.read())
                final, sim, sk, matched, missing = final_score(cv_text, jd)

                st.metric("Match Score", f"{final:.0f}%")

                with st.expander("Score Breakdown"):
                    st.write(f"TF-IDF Similarity: **{sim:.1f}%**")
                    st.write(f"Skill Match Score: **{sk:.1f}%**")
                    st.write("Final = 25% Similarity + 75% Skill Match")

                st.write("✅ Matched Skills:", ", ".join(matched) if matched else "None detected")
                st.write("⚠️ Missing Skills:", ", ".join(missing) if missing else "None detected")

                st.write("### Recommendations")
                if missing:
                    st.write("If you truly have these, add them clearly to your CV:")
                    for s in missing[:10]:
                        st.write(f"- {s}")
                else:
                    st.write("Your CV aligns well with this job description.")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Ready to analyze. Upload a CV, paste a job description, then click Analyze.")
