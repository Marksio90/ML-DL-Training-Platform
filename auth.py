"""
auth.py — Email login + self-signup (Streamlit + streamlit-authenticator)

Funkcje:
- require_login(): wymusza logowanie (zakładka Logowanie) i opcjonalnie Rejestracja
- Rejestracja nowych kont (zakładka Rejestracja) — zapisywane do .streamlit/users.json
- auth_sidebar(): pokazuje status + przycisk Wyloguj
- current_user(): zwraca zalogowanego usera (lub None)

Konfiguracja (w .streamlit/secrets.toml):
[auth]
cookie_name        = "tmiv_auth"
cookie_key         = "ULTRA-SECRET-CHANGE-ME"   # mocny losowy sekret
cookie_expiry_days = 7
emails             = ["admin@example.com"]      # opcjonalni użytkownicy "zaszyci" w secrets
names              = ["Admin"]                  # jw.
hashed_passwords   = ["$2b$12$...."]            # jw., hashe bcrypt

# Rejestracja (opcjonalnie)
allow_self_signup  = true                       # domyślnie: false
invite_code        = "MY-INVITE-2025"           # opcjonalny kod zaproszenia (puste = brak)
min_password_len   = 10                         # domyślnie: 10

Uwaga: Użytkownicy zarejestrowani w UI zapisywani są do .streamlit/users.json
(oddzielnie od secrets). Aplikacja przy starcie scala zbiory: secrets + users.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import streamlit as st

try:
    import streamlit_authenticator as stauth
except Exception:
    st.error("Brak pakietu `streamlit-authenticator`. Dodaj go do requirements.txt i zainstaluj.")
    st.stop()

# Wyjątek przy braku secrets.toml (w nowszych Streamlitach)
try:
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except Exception:  # starsze wersje
    class StreamlitSecretNotFoundError(Exception):
        pass


# --------------------------- ŚCIEŻKI/STAŁE ---------------------------

APP_ROOT = Path.cwd()
STREAMLIT_DIR = APP_ROOT / ".streamlit"
STREAMLIT_DIR.mkdir(parents=True, exist_ok=True)
USERS_FILE = STREAMLIT_DIR / "users.json"  # dynamiczny sklepik użytkowników (rejestracja)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --------------------------- POMOCNICZE ---------------------------

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _merge_users(*sources: list[dict]) -> dict:
    """Łączy wielu dostawców userów do struktury: {email: {"name":..., "password":...}}.
    Ostatni wygrywa przy konflikcie (np. aktualizacja hasła)."""
    merged: dict[str, dict] = {}
    for src in sources:
        for email, rec in src.items():
            email_lc = str(email).strip().lower()
            merged[email_lc] = {"name": rec.get("name", email_lc), "password": rec.get("password", "")}
    return merged

def _hash_password(plain: str) -> str:
    return stauth.Hasher([plain]).generate()[0]


# --------------------------- KONFIGURACJA ---------------------------

def _load_signup_policy() -> dict:
    # Domyślne ustawienia rejestracji (gdy brak w secrets)
    try:
        auth_sec = st.secrets.get("auth", {})
    except StreamlitSecretNotFoundError:
        auth_sec = {}
    except Exception:
        auth_sec = {}

    return {
        "allow_self_signup": bool(auth_sec.get("allow_self_signup", False)),
        "invite_code": str(auth_sec.get("invite_code", "") or ""),
        "min_password_len": int(auth_sec.get("min_password_len", 10)),
    }

def _users_from_secrets() -> dict:
    """Użytkownicy zdefiniowani w secrets.toml (emaile normalizowane do lowercase)."""
    try:
        auth = st.secrets.get("auth", {})
    except StreamlitSecretNotFoundError:
        return {}
    except Exception:
        return {}

    emails = [str(e).strip().lower() for e in auth.get("emails", [])]
    names = list(auth.get("names", [])) if "names" in auth else emails
    hashed = list(auth.get("hashed_passwords", []))
    if not emails or not hashed or len(emails) != len(hashed):
        return {}

    users = {}
    for i, email in enumerate(emails):
        nm = names[i] if i < len(names) else email
        users[email] = {"name": nm, "password": hashed[i]}
    return users

def _cookie_settings() -> tuple[str, str, int]:
    try:
        auth = st.secrets.get("auth", {})
    except StreamlitSecretNotFoundError:
        auth = {}
    except Exception:
        auth = {}
    name = str(auth.get("cookie_name", "tmiv_auth"))
    key = str(auth.get("cookie_key", "CHANGE-ME"))
    try:
        days = int(auth.get("cookie_expiry_days", 7))
    except Exception:
        days = 7
    return name, key, days

def _demo_config() -> dict:
    """Tryb demo (gdy brak configu) — hasło: demo1234 (NIE NA PROD!)."""
    demo_hash = "$2b$12$fk9gX5J3a5XfRrY9Z1iCxe3zKZr1C/9XoFv5S1lzdE8kq1bRk8TNi"
    return {"demo@local": {"name": "Demo User", "password": demo_hash}}

def _load_dynamic_users() -> dict:
    data = _read_json(USERS_FILE)
    users = data.get("users", {})
    # normalize keys to lowercase
    norm = {}
    for e, v in users.items():
        email = str(e).strip().lower()
        norm[email] = {"name": v.get("name", email), "password": v.get("password", "")}
    return norm

def _save_dynamic_user(email: str, name: str, hashed_password: str) -> None:
    email = str(email).strip().lower()
    data = _read_json(USERS_FILE)
    users = data.get("users", {})
    users[email] = {"name": name, "password": hashed_password}
    data["users"] = users
    _write_json(USERS_FILE, data)

def _get_config() -> dict:
    """Zwraca pełną konfigurację auth (scala secrets + users.json) i cache'uje w session_state."""
    cache_key = "_auth_config"
    if cache_key in st.session_state and isinstance(st.session_state[cache_key], dict):
        return st.session_state[cache_key]

    # użytkownicy z secrets + dynamiczni z pliku
    secrets_users = _users_from_secrets()
    dynamic_users = _load_dynamic_users()
    users = _merge_users(secrets_users, dynamic_users)

    if not users:  # totalny brak -> demo
        st.warning(
            "🔒 Brak skonfigurowanych użytkowników — aktywuję konto **demo@local / demo1234**. "
            "Na produkcji skonfiguruj `.streamlit/secrets.toml` lub dodaj konto w Rejestracji."
        )
        users = _demo_config()

    c_name, c_key, c_days = _cookie_settings()
    cfg = {
        "credentials": {"usernames": users},
        "cookie": {"name": c_name, "key": c_key, "expiry_days": c_days},
    }
    st.session_state[cache_key] = cfg
    return cfg

def _make_authenticator() -> stauth.Authenticate:
    cfg = _get_config()
    return stauth.Authenticate(
        {"usernames": cfg["credentials"]["usernames"]},
        cfg["cookie"]["name"],
        cfg["cookie"]["key"],
        cfg["cookie"]["expiry_days"],
    )


# --------------------------- UI: REJESTRACJA ---------------------------

def _render_signup_ui(policy: dict) -> None:
    st.subheader("🆕 Rejestracja nowego konta")
    with st.form("signup_form", clear_on_submit=False):
        email = st.text_input("E-mail", placeholder="twoj@adres.pl")
        display_name = st.text_input("Imię/Nazwa wyświetlana", placeholder="Jan Kowalski")
        pwd1 = st.text_input("Hasło", type="password")
        pwd2 = st.text_input("Powtórz hasło", type="password")
        invite = st.text_input("Kod zaproszenia (jeśli wymagany)", placeholder="(opcjonalnie)")

        accepted = st.checkbox("Akceptuję regulamin i politykę prywatności")
        submit = st.form_submit_button("Utwórz konto")

    if not submit:
        return

    # Walidacje
    if not accepted:
        st.error("Musisz zaakceptować regulamin.")
        return

    if not EMAIL_RE.match(email or ""):
        st.error("Podaj poprawny adres e-mail.")
        return

    min_len = int(policy.get("min_password_len", 10))
    if not pwd1 or len(pwd1) < min_len:
        st.error(f"Hasło musi mieć co najmniej {min_len} znaków.")
        return

    if pwd1 != pwd2:
        st.error("Hasła nie są identyczne.")
        return

    inv_required = bool(policy.get("invite_code"))
    if inv_required and (invite or "").strip() != str(policy["invite_code"]).strip():
        st.error("Niepoprawny kod zaproszenia.")
        return

    # Unikalność
    existing = _get_config()["credentials"]["usernames"]
    email_lc = str(email).strip().lower()
    if email_lc in existing:
        st.error("Konto z takim e-mailem już istnieje.")
        return

    # Zapis
    try:
        hashed = _hash_password(pwd1)
        _save_dynamic_user(email=email_lc, name=display_name or email_lc, hashed_password=hashed)
        # Inwaliduj cache i odśwież UI
        st.session_state.pop("_auth_config", None)
        st.success("Konto utworzone ✅ Możesz się zalogować.")
        st.rerun()
    except Exception as e:
        st.exception(e)
        st.error("Nie udało się utworzyć konta.")


# --------------------------- PUBLIC API ---------------------------

def require_login() -> dict:
    """Renderuje zakładki Logowanie/Rejestracja (jeśli włączona) i zwraca zalogowanego usera.
    Klucz: NIE robimy st.stop() wewnątrz pierwszego taba, tylko dopiero po wyrenderowaniu obu.
    """
    # Jeśli już zalogowany – zwróć od razu
    if "_current_user" in st.session_state and st.session_state["_current_user"]:
        return st.session_state["_current_user"]

    policy = _load_signup_policy()
    tabs = ["🔑 Logowanie"]
    allow_signup = bool(policy.get("allow_self_signup", False))
    if allow_signup:
        tabs.append("🆕 Rejestracja")

    t = st.tabs(tabs)

    # --- LOGOWANIE ---
    with t[0]:
        authenticator = _make_authenticator()
        name, auth_status, username = authenticator.login(
            location="main",
            max_login_attempts=3,
            key="login_box",
        )

        if auth_status is False:
            st.error("Błędny e-mail lub hasło.")
            # nie przerywamy — pozwólmy wyrenderować też rejestrację
        elif auth_status is None:
            st.info("Zaloguj się, aby korzystać z aplikacji.")
            if allow_signup:
                st.caption("Nie masz konta? Przejdź do zakładki **Rejestracja**.")
            # nie przerywamy — wyrenderujemy drugi tab
        else:
            # Zalogowany
            st.session_state["_current_user"] = {"email": username, "name": name}

    # --- REJESTRACJA ---
    if allow_signup:
        with t[1]:
            _render_signup_ui(policy)

    # Jeśli nadal brak usera — zatrzymaj appkę po wyrenderowaniu UI
    if not st.session_state.get("_current_user"):
        st.stop()

    return st.session_state["_current_user"]


def auth_sidebar() -> None:
    """Sidebar: status sesji + wylogowanie."""
    authenticator = _make_authenticator()
    with st.sidebar:
        st.caption("🔐 **Sesja**")
        user = st.session_state.get("_current_user", {})
        if user:
            st.write(f"Zalogowano jako: **{user.get('name', user.get('email'))}**")
        authenticator.logout("Wyloguj", "sidebar")


def current_user() -> dict | None:
    """Zwraca użytkownika z session_state lub None."""
    return st.session_state.get("_current_user")
