from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from agents.job_agent import get_job_text
from agents.cv_agent import clean_text, extract_skills
from database.vector_agent import build_job_index, search_jobs


def tfidf_similarity_score(cv_text, job_text):
    cv = clean_text(cv_text)
    job = clean_text(job_text)

    if len(cv) < 30 or len(job) < 30:
        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )

    matrix = vectorizer.fit_transform([cv, job])
    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    return max(0.0, min(1.0, float(similarity))) * 100.0


def skill_match_score(cv_skills, job_text):
    job_skills = set(extract_skills(job_text))
    cv_skills = set(cv_skills)

    if not job_skills:
        return 0.0, [], []

    matched = sorted(cv_skills & job_skills)
    missing = sorted(job_skills - cv_skills)

    score = (len(matched) / len(job_skills)) * 100.0
    return score, matched, missing


def calculate_final_score(similarity_score, skill_score):
    if skill_score == 0:
        return similarity_score
    return (0.25 * similarity_score) + (0.75 * skill_score)


def analyze_job_match(cv_data, job):
    cv_text = cv_data["text"]
    cv_skills = cv_data["skills"]

    job_text = get_job_text(job)

    similarity = tfidf_similarity_score(cv_text, job_text)
    skill_score, matched, missing = skill_match_score(cv_skills, job_text)
    final_score = calculate_final_score(similarity, skill_score)

    return {
        "title": job["title"],
        "company": job["company"],
        "location": job["location"],
        "description": job["description"],
        "url": job["url"],
        "source": job["source"],
        "similarity_score": round(similarity, 2),
        "skill_score": round(skill_score, 2),
        "final_score": round(final_score, 2),
        "matched_skills": matched,
        "missing_skills": missing,
    }


def analyze_all_jobs(cv_data, jobs):
    if not jobs:
        return []

    index, _ = build_job_index(jobs)
    candidate_jobs = search_jobs(cv_data["text"], jobs, index, top_k=min(5, len(jobs)))

    results = []

    for job in candidate_jobs:
        result = analyze_job_match(cv_data, job)
        results.append(result)

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results
