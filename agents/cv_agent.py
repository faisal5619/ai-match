import re
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document


SKILLS = {
    "java", "python", "sql", "git", "linux", "docker",
    "aws", "azure", "api", "rest", "html", "css",
    "javascript", "typescript", "react",
    "ai", "artificial intelligence", "machine learning",
    "nlp", "data analysis", "computer science",
    "programming", "oop", "data structures", "algorithms",
    "microsoft office", "excel", "powerpoint",
    "teamwork", "communication", "problem solving",
    "english", "arabic"
}


def extract_text(filename, file_bytes):
    """
    Extract text from uploaded CV file
    """

    name = filename.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(
            [(page.extract_text() or "") for page in reader.pages]
        )

    if name.endswith(".docx"):
        doc = Document(BytesIO(file_bytes))
        return "\n".join(
            [p.text for p in doc.paragraphs if p.text]
        )

    if name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file format")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9+\#\.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text):
    """
    Detect skills inside CV
    """

    cleaned = clean_text(text)

    found = set()

    for skill in SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", cleaned):
            found.add(skill)

    return sorted(found)


def analyze_cv(filename, file_bytes):
    """
    Main CV Agent function
    """

    raw_text = extract_text(filename, file_bytes)

    skills = extract_skills(raw_text)

    return {
        "text": raw_text,
        "skills": skills
    }
