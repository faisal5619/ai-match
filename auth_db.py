import sqlite3
import hashlib

DB_PATH = "aimatch.db"


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
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES users(email)
        )
    """)
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
        "SELECT full_name, preferred_location, job_type, field_of_interest FROM user_profiles WHERE email = ?",
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
        }
    return {
        "full_name": "",
        "preferred_location": "",
        "job_type": "",
        "field_of_interest": "",
    }


def save_user_profile(email: str, full_name: str, preferred_location: str, job_type: str, field_of_interest: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_profiles (email, full_name, preferred_location, job_type, field_of_interest, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(email) DO UPDATE SET
            full_name = excluded.full_name,
            preferred_location = excluded.preferred_location,
            job_type = excluded.job_type,
            field_of_interest = excluded.field_of_interest,
            updated_at = CURRENT_TIMESTAMP
    """, (email.strip().lower(), full_name.strip(), preferred_location, job_type, field_of_interest))
    conn.commit()
    conn.close()
