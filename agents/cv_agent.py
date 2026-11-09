import re
from io import BytesIO
from urllib.parse import quote
import requests
from pypdf import PdfReader
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


def fetch_github_skills(github_username: str) -> list:
    """
    Fetches public repos from GitHub API and extracts programming languages.
    Returns a list of skill strings. Completely free, no auth needed.
    """
    if not github_username.strip():
        return []
    try:
        # quote() keeps whatever was typed inside the username part of the address.
        url = f"https://api.github.com/users/{quote(github_username.strip(), safe='')}/repos?per_page=30&sort=updated"
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "AI-Match-App"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return []
        repos = r.json()
        languages = set()
        topics = set()
        for repo in repos:
            if repo.get("language"):
                languages.add(repo["language"].lower())
            for topic in repo.get("topics", []):
                topics.add(topic.lower())
        # Map to skills the system understands
        skill_map = {
            "python": "python", "javascript": "javascript", "typescript": "typescript",
            "java": "java", "html": "html", "css": "css", "sql": "sql",
            "shell": "linux", "dockerfile": "docker", "react": "react",
            "jupyter notebook": "data analysis", "r": "data analysis",
        }
        found = []
        for lang in languages:
            if lang in skill_map:
                found.append(skill_map[lang])
            else:
                found.append(lang)
        for topic in topics:
            if topic in skill_map:
                found.append(skill_map[topic])
        return list(set(found))
    except Exception:
        return []
