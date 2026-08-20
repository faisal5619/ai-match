import sqlite3
import hashlib

DB_PATH = "aimatch.db"

# A public account so visitors can try the app without registering.
#
# It is seeded here in code rather than created by hand through the sign-up
# form, because Streamlit Community Cloud gives each app a temporary filesystem:
# when the app sleeps or restarts, aimatch.db is wiped and every registered
# account disappears with it. Anything created by hand would work for a day and
# then quietly vanish. Seeding on startup means the demo account always exists.
DEMO_EMAIL = "demo@aimatch.app"
DEMO_PASSWORD = "demo1234"
DEMO_NAME = "Demo User"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_auth_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_cvs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            cv_filename TEXT NOT NULL,
            cv_bytes BLOB NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES users(email)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            email TEXT PRIMARY KEY,
            full_name TEXT,
            preferred_location TEXT DEFAULT '',
            job_type TEXT DEFAULT '',
            field_of_interest TEXT DEFAULT '',
            github_username TEXT DEFAULT '',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES users(email)
        )
    """)
    # Add github_username column if it doesn't exist (for existing databases)
    try:
        c.execute("ALTER TABLE user_profiles ADD COLUMN github_username TEXT DEFAULT ''")
        conn.commit()
    except Exception:
        pass
    conn.commit()

    # Seed the demo account. INSERT OR IGNORE means this is safe to run on every
    # startup: if the account already exists, nothing happens.
    c.execute(
        "INSERT OR IGNORE INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
        (DEMO_NAME, DEMO_EMAIL, hash_password(DEMO_PASSWORD))
    )
    c.execute(
        "INSERT OR IGNORE INTO user_profiles (email, full_name) VALUES (?, ?)",
        (DEMO_EMAIL, DEMO_NAME)
    )
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(full_name: str, email: str, password: str):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
            (full_name.strip(), email.strip().lower(), hash_password(password))
        )
        # Create empty profile row for new user
        c.execute(
            "INSERT OR IGNORE INTO user_profiles (email, full_name) VALUES (?, ?)",
            (email.strip().lower(), full_name.strip())
        )
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally:
        conn.close()


def login_user(email: str, password: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT full_name, password_hash FROM users WHERE email = ?",
        (email.strip().lower(),)
    )
    row = c.fetchone()
    conn.close()

    if not row:
        return False, "No account found with this email."

    full_name, stored_hash = row
    if stored_hash != hash_password(password):
        return False, "Incorrect password."

    return True, full_name


def save_user_cv(email: str, cv_filename: str, cv_bytes: bytes):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM user_cvs WHERE email = ?", (email.strip().lower(),))
    c.execute(
        "INSERT INTO user_cvs (email, cv_filename, cv_bytes) VALUES (?, ?, ?)",
        (email.strip().lower(), cv_filename, cv_bytes)
    )
    conn.commit()
    conn.close()


def delete_user_cv(email: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM user_cvs WHERE email = ?", (email.strip().lower(),))
    conn.commit()
    conn.close()


def load_user_cv(email: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT cv_filename, cv_bytes FROM user_cvs WHERE email = ? ORDER BY uploaded_at DESC LIMIT 1",
        (email.strip().lower(),)
    )
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], bytes(row[1])
    return None, None


def load_user_profile(email: str):
    """Returns dict with profile fields, or defaults if not found."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT full_name, preferred_location, job_type, field_of_interest, github_username FROM user_profiles WHERE email = ?",
        (email.strip().lower(),)
    )
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "full_name": row[0] or "",
            "preferred_location": row[1] or "",
            "job_type": row[2] or "",
            "field_of_interest": row[3] or "",
            "github_username": row[4] or "",
        }
    return {
        "full_name": "",
        "preferred_location": "",
        "job_type": "",
        "field_of_interest": "",
        "github_username": "",
    }


def save_user_profile(email: str, full_name: str, preferred_location: str, job_type: str, field_of_interest: str, github_username: str = ""):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_profiles (email, full_name, preferred_location, job_type, field_of_interest, github_username, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(email) DO UPDATE SET
            full_name = excluded.full_name,
            preferred_location = excluded.preferred_location,
            job_type = excluded.job_type,
            field_of_interest = excluded.field_of_interest,
            github_username = excluded.github_username,
            updated_at = CURRENT_TIMESTAMP
    """, (email.strip().lower(), full_name.strip(), preferred_location, job_type, field_of_interest, github_username.strip()))
    conn.commit()
    conn.close()


def fetch_github_skills(github_username: str) -> list:
    """
    Fetches public repos from GitHub API and extracts programming languages.
    Returns a list of skill strings. Completely free, no auth needed.
    """
    if not github_username.strip():
        return []
    try:
        import requests
        url = f"https://api.github.com/users/{github_username.strip()}/repos?per_page=30&sort=updated"
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
