"""Simple username/password authentication for Streamlit.

- Users are stored in a local SQLite file (users.db).
- Passwords are never stored: only a salted PBKDF2-SHA256 hash.
- Usernames are restricted to letters, digits and underscore, so they are
  safe to use as folder names for per-user data.
"""

import hashlib
import hmac
import os
import re
import sqlite3

import streamlit as st

DB_PATH = os.getenv("USERS_DB", "users.db")
PBKDF2_ROUNDS = 200_000
USERNAME_RE = re.compile(r"^[a-z0-9_]{3,30}$")
MIN_PASSWORD_LENGTH = 8


# ---------- storage ----------

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               username TEXT PRIMARY KEY,
               salt BLOB NOT NULL,
               password_hash BLOB NOT NULL
           )"""
    )
    return conn


def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)


def create_user(username, password):
    """Return (ok, message)."""
    username = username.strip().lower()

    if not USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, digits or underscore."
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters."

    salt = os.urandom(16)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
                (username, salt, _hash_password(password, salt)),
            )
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created."


def verify_user(username, password):
    """Return the normalized username if credentials are valid, else None."""
    username = username.strip().lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()

    if row is None:
        # Hash anyway so timing doesn't reveal whether the user exists.
        _hash_password(password, b"\x00" * 16)
        return None

    salt, stored_hash = row
    if hmac.compare_digest(_hash_password(password, salt), stored_hash):
        return username
    return None


# ---------- Streamlit UI ----------

def require_login():
    """Show the login / sign-up page until the user is signed in.

    Returns the username once authenticated; otherwise stops the script.
    """
    if st.session_state.get("user"):
        return st.session_state["user"]

    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        st.title("📚 Chat with PDF")
        login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

        with login_tab:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", type="primary")
            if submitted:
                user = verify_user(username, password)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        with signup_tab:
            with st.form("signup_form"):
                new_username = st.text_input("Choose a username")
                new_password = st.text_input("Choose a password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                created = st.form_submit_button("Create account")
            if created:
                if new_password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, message = create_user(new_username, new_password)
                    if ok:
                        st.success(f"{message} You can now log in from the other tab.")
                    else:
                        st.error(message)

    st.stop()


def logout():
    st.session_state.clear()
    st.rerun()