import sqlite3
from datetime import datetime

DB_NAME = "ai_match.db"


def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        raw_text TEXT,
        skills TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        company TEXT,
        location TEXT,
        description TEXT,
        url TEXT,
        source TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume_id INTEGER,
        job_title TEXT,
        company TEXT,
        location TEXT,
        similarity_score REAL,
        skill_score REAL,
        final_score REAL,
        matched_skills TEXT,
        missing_skills TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_resume(file_name, raw_text, skills):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO resumes (file_name, raw_text, skills, created_at)
    VALUES (?, ?, ?, ?)
    """, (
        file_name,
        raw_text,
        ", ".join(skills),
        datetime.now().isoformat()
    ))

    conn.commit()
    resume_id = cur.lastrowid
    conn.close()
    return resume_id


def insert_job(title, company, location, description, url, source):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO jobs (title, company, location, description, url, source, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        title,
        company,
        location,
        description,
        url,
        source,
        datetime.now().isoformat()
    ))

    conn.commit()
    job_id = cur.lastrowid
    conn.close()
    return job_id


def insert_match(
    resume_id,
    job_title,
    company,
    location,
    similarity_score,
    skill_score,
    final_score,
    matched_skills,
    missing_skills
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO matches (
        resume_id,
        job_title,
        company,
        location,
        similarity_score,
        skill_score,
        final_score,
        matched_skills,
        missing_skills,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        resume_id,
        job_title,
        company,
        location,
        similarity_score,
        skill_score,
        final_score,
        ", ".join(matched_skills),
        ", ".join(missing_skills),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()
