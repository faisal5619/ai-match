import sqlite3
import hashlib
import os

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
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(full_name: str, email: str, password: str):
    """
    Returns (True, None) on success.
    Returns (False, error_message) on failure.
    """
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
            (full_name.strip(), email.strip().lower(), hash_password(password))
        )
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally:
        conn.close()


def login_user(email: str, password: str):
    """
    Returns (True, full_name) on success.
    Returns (False, error_message) on failure.
    """
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
