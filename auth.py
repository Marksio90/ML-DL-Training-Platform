
"""
auth.py — Secure email login for Streamlit app using streamlit-authenticator.

Instrukcja:
1) Dodaj do `.streamlit/secrets.toml` sekcję [auth] z listą użytkowników (emaile) i hashami haseł (bcrypt).
2) W requirements.txt musi być: streamlit-authenticator, bcrypt.
3) W app.py wywołaj require_login() JAK NAJWCZEŚNIEJ (przed całą logiką UI).

Plik `secrets.example.toml` pokazuje gotowy wzór konfiguracji.
"""

from __future__ import annotations

import os
import streamlit as st

try:
    import streamlit_authenticator as stauth
except Exception as e:
    st.error("Brak pakietu `streamlit-authenticator`. Dodaj go do requirements.txt i zainstaluj.")
    st.stop()


def _build_config_from_secrets() -> dict:
    """Zbuduj config dla streamlit-authenticator z `st.secrets`.

    Oczekiwany układ w `.streamlit/secrets.toml`:
    [auth]
    cookie_name = "tmiv_auth"
    cookie_key = "SUPER-SECRET-CHANGE-ME"
    cookie_expiry_days = 7
    emails = ["user1@example.com", "user2@example.com"]
    names = ["Użytkownik 1", "Użytkownik 2"]  # opcjonalnie, domyślnie = email
    hashed_passwords = [
      "$2b$12$CwTycUXWue0Thq9StjUM0uJ8JmGdQzjKc4w4F7oW3sHnQ6wqYbG1e",
      "..."
    ]

    Zwraca dict zgodny z API stauth.Authenticate().
    """
    auth = st.secrets.get("auth", {})
    emails = list(auth.get("emails", []))
    names = list(auth.get("names", [])) if "names" in auth else emails
    hashed = list(auth.get("hashed_passwords", []))

    if not (emails and hashed and len(emails) == len(hashed)):
        return {}

    # cookie
    cookie_name = auth.get("cookie_name", "tmiv_auth")
    cookie_key = auth.get("cookie_key", "CHANGE-ME")
    cookie_expiry_days = int(auth.get("cookie_expiry_days", 7))

    # mapujemy do struktury expected by streamlit-authenticator
    users = {}
    for i, email in enumerate(emails):
        nm = names[i] if i < len(names) else email
        users[email] = {"name": nm, "password": hashed[i]}

    config = {
        "credentials": {"usernames": users},
        "cookie": {
            "name": cookie_name,
            "key": cookie_key,
            "expiry_days": cookie_expiry_days,
        },
        "preauthorized": {"emails": []},  # opcjonalnie
    }
    return config


def _demo_config() -> dict:
    """Fallback do konta demo (gdy brak .secrets). HASŁO: demo1234

    Używaj WYŁĄCZNIE lokalnie. Na produkcji koniecznie skonfiguruj `.streamlit/secrets.toml`!
    """
    # hash hasła "demo1234", wygenerowany przez stauth.Hasher
    demo_hash = "$2b$12$fk9gX5J3a5XfRrY9Z1iCxe3zKZr1C/9XoFv5S1lzdE8kq1bRk8TNi"
    return {
        "credentials": {
            "usernames": {
                "demo@local": {"name": "Demo User", "password": demo_hash}
            }
        },
        "cookie": {"name": "tmiv_auth", "key": "DEMO-ONLY-CHANGE", "expiry_days": 1},
        "preauthorized": {"emails": []},
    }


def _make_authenticator():
    cfg = _build_config_from_secrets()
    if not cfg:
        st.warning(
            "🔒 Nie wykryto poprawnej konfiguracji w `.streamlit/secrets.toml` → aktywowano konto **demo@local / demo1234**. "
            "Skonfiguruj docelowe konta w secrets, aby włączyć produkcyjne logowanie e‑mailem."
        )
        cfg = _demo_config()

    names = [v["name"] for v in cfg["credentials"]["usernames"].values()]
    usernames = list(cfg["credentials"]["usernames"].keys())
    passwords = [v["password"] for v in cfg["credentials"]["usernames"].values()]

    authenticator = stauth.Authenticate(
        {"usernames": cfg["credentials"]["usernames"]},
        cfg["cookie"]["name"],
        cfg["cookie"]["key"],
        cfg["cookie"]["expiry_days"],
    )
    return authenticator


def require_login() -> dict:
    """Renderuje formularz logowania i blokuje aplikację do czasu poprawnej autoryzacji.

    Zwraca słownik z informacjami o użytkowniku: {"email": ..., "name": ...}
    """
    authenticator = _make_authenticator()
    # Formularz logowania (w miejscu, gdzie zostanie wywołane)
    name, auth_status, username = authenticator.login(location="main", max_login_attempts=3)

    if auth_status is False:
        st.error("Błędny e‑mail lub hasło.")
        st.stop()
    if auth_status is None:
        st.info("Zaloguj się, aby korzystać z aplikacji.")
        st.stop()

    # Zalogowany
    st.session_state["_current_user"] = {"email": username, "name": name}
    return st.session_state["_current_user"]


def auth_sidebar():
    """Sekcja w sidebarze: status i przycisk wylogowania."""
    authenticator = _make_authenticator()
    with st.sidebar:
        st.caption("🔐 **Sesja**")
        user = st.session_state.get("_current_user", {})
        if user:
            st.write(f"Zalogowano jako: **{user.get('name', user.get('email'))}**")
        authenticator.logout("Wyloguj", "sidebar")
