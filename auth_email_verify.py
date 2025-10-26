
"""
auth_email_verify.py — Email login + self-signup + e-mail verification code (Streamlit)

Funkcje:
- require_login(): zakładki Logowanie / Rejestracja / (Weryfikacja kodem dla nowych kont)
- auth_sidebar(): status sesji + Wyloguj
- current_user(): odczyt zalogowanego użytkownika

Magazyn danych (plikowy, prosty):
- .streamlit/users.json            — zarejestrowani i zweryfikowani użytkownicy
- .streamlit/users_pending.json    — oczekujący na weryfikację (kod + TTL)
- .streamlit/verify_log.jsonl      — dziennik wysyłek kodów (prosty rate-limit)

Konfiguracja w `.streamlit/secrets.toml`:

[auth]
cookie_name        = "tmiv_auth"
cookie_key         = "ULTRA-SECRET-CHANGE-ME"
cookie_expiry_days = 7
allow_self_signup  = true
invite_code        = ""              # opcjonalny kod zaproszenia
min_password_len   = 10

# (opcjonalnie) wstępni użytkownicy (hash bcrypt):
emails           = ["admin@example.com"]
names            = ["Admin"]
hashed_passwords = ["$2b$12$...."]

[email]  # USTAW, aby wysyłać kody
host     = "smtp.yourprovider.com"
port     = 587
username = "apikey_or_user"
password = "secret"
sender   = "TMIV <no-reply@yourdomain>"
use_tls  = true                      # true -> STARTTLS (587), false -> SSL (465)

Uwaga: To jest implementacja plikowa (demo/dev). Na PROD rozważ bazę i provider transakcyjny (Mailgun, Sendgrid).
"""

from __future__ import annotations

import json
import re
import smtplib
import ssl
import time
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
import random
import string

import streamlit as st

try:
    import streamlit_authenticator as stauth
except Exception:
    st.error("Brak pakietu `streamlit-authenticator`. Dodaj go do requirements.txt i zainstaluj.")
    st.stop()

try:
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except Exception:
    class StreamlitSecretNotFoundError(Exception):
        pass

# --------------------------- ŚCIEŻKI ---------------------------

APP_ROOT = Path.cwd()
ST_DIR = APP_ROOT / ".streamlit"
ST_DIR.mkdir(parents=True, exist_ok=True)

USERS_FILE = ST_DIR / "users.json"
PENDING_FILE = ST_DIR / "users_pending.json"
VERIFY_LOG = ST_DIR / "verify_log.jsonl"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# --------------------------- I/O ---------------------------

def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# --------------------------- Utils ---------------------------

def _hash_password(plain: str) -> str:
    return stauth.Hasher([plain]).generate()[0]

def _now_ts() -> int:
    return int(time.time())

def _gen_code(n: int = 6) -> str:
    return "".join(random.choices(string.digits, k=n))

def _policy() -> dict:
    try:
        auth = st.secrets.get("auth", {})
    except StreamlitSecretNotFoundError:
        auth = {}
    except Exception:
        auth = {}
    return {
        "allow_self_signup": bool(auth.get("allow_self_signup", False)),
        "invite_code": str(auth.get("invite_code", "") or ""),
        "min_password_len": int(auth.get("min_password_len", 10)),
        "code_ttl_sec": int(auth.get("code_ttl_sec", 900)),  # 15 min
        "resend_cooldown_sec": int(auth.get("resend_cooldown_sec", 60)),
        "max_verify_attempts": int(auth.get("max_verify_attempts", 5)),
    }

def _smtp_cfg() -> dict | None:
    try:
        sec = st.secrets["email"]
    except Exception:
        return None
    cfg = {
        "host": sec.get("host"),
        "port": int(sec.get("port", 587)),
        "username": sec.get("username"),
        "password": sec.get("password"),
        "sender": sec.get("sender", "no-reply@localhost"),
        "use_tls": bool(sec.get("use_tls", True)),
    }
    if not cfg["host"] or not cfg["port"] or not cfg["username"] or not cfg["password"]:
        return None
    return cfg

def _send_code(email: str, code: str) -> bool:
    cfg = _smtp_cfg()
    if not cfg:
        st.error("E-mail SMTP nie skonfigurowany w `.streamlit/secrets.toml` [email].")
        return False

    msg = EmailMessage()
    msg["Subject"] = "Twój kod weryfikacyjny"
    msg["From"] = cfg["sender"]
    msg["To"] = email
    msg.set_content(f"Twój kod weryfikacyjny: {code}\nKod wygasa za 15 minut.")

    try:
        if cfg["use_tls"]:
            with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(cfg["username"], cfg["password"])
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ssl.create_default_context()) as server:
                server.login(cfg["username"], cfg["password"])
                server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Nie udało się wysłać e-maila z kodem: {e}")
        return False

# --------------------------- Użytkownicy ---------------------------

def _users() -> dict:
    raw = _read_json(USERS_FILE, {"users": {}})
    out = {}
    for e, rec in raw.get("users", {}).items():
        email = str(e).strip().lower()
        out[email] = {"name": rec.get("name", email), "password": rec.get("password", "")}
    return out

def _save_user(email: str, name: str, hashed_password: str) -> None:
    email = str(email).strip().lower()
    raw = _read_json(USERS_FILE, {"users": {}})
    raw["users"][email] = {"name": name, "password": hashed_password}
    _write_json(USERS_FILE, raw)

def _pending() -> dict:
    # { email: { name, hashed_password, code, exp_ts, attempts } }
    return _read_json(PENDING_FILE, {"pending": {}})

def _save_pending(email: str, payload: dict) -> None:
    email = str(email).strip().lower()
    data = _pending()
    data["pending"][email] = payload
    _write_json(PENDING_FILE, data)

def _del_pending(email: str) -> None:
    email = str(email).strip().lower()
    data = _pending()
    if email in data["pending"]:
        del data["pending"][email]
        _write_json(PENDING_FILE, data)

# --------------------------- Secrets users + cookie ---------------------------

def _users_from_secrets() -> dict:
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

# --------------------------- Authenticator ---------------------------

def _auth_config() -> dict:
    if "_auth_config" in st.session_state:
        return st.session_state["_auth_config"]

    merged = {}
    merged.update(_users_from_secrets())
    merged.update(_users())  # verified

    if not merged:
        # Konto demo tylko, jeśli pusto
        demo_hash = "$2b$12$fk9gX5J3a5XfRrY9Z1iCxe3zKZr1C/9XoFv5S1lzdE8kq1bRk8TNi"
        merged["demo@local"] = {"name": "Demo User", "password": demo_hash}
        st.warning("🔒 Brak skonfigurowanych użytkowników — aktywne konto demo@local / demo1234 (tylko lokalnie).")

    c_name, c_key, c_days = _cookie_settings()
    cfg = {
        "credentials": {"usernames": merged},
        "cookie": {"name": c_name, "key": c_key, "expiry_days": c_days},
    }
    st.session_state["_auth_config"] = cfg
    return cfg

def _make_authenticator() -> stauth.Authenticate:
    cfg = _auth_config()
    return stauth.Authenticate(
        {"usernames": cfg["credentials"]["usernames"]},
        cfg["cookie"]["name"],
        cfg["cookie"]["key"],
        cfg["cookie"]["expiry_days"],
    )

# --------------------------- UI: Rejestracja + Weryfikacja ---------------------------

def _render_signup_ui(policy: dict) -> None:
    st.subheader("🆕 Rejestracja nowego konta")
    with st.form("signup_form", clear_on_submit=False):
        email = st.text_input("E-mail", placeholder="twoj@adres.pl")
        display_name = st.text_input("Imię/Nazwa wyświetlana", placeholder="Jan Kowalski")
        pwd1 = st.text_input("Hasło", type="password")
        pwd2 = st.text_input("Powtórz hasło", type="password")
        invite = st.text_input("Kod zaproszenia (jeśli wymagany)", placeholder="(opcjonalnie)")
        accepted = st.checkbox("Akceptuję Regulamin i Politykę Prywatności")
        submit = st.form_submit_button("Utwórz konto")

    if not submit:
        return

    # Walidacje
    email_lc = (email or "").strip().lower()
    if not EMAIL_RE.match(email_lc):
        st.error("Podaj poprawny adres e-mail.")
        return

    if email_lc in _auth_config()["credentials"]["usernames"]:
        st.error("Konto z takim e-mailem już istnieje.")
        return

    min_len = int(policy.get("min_password_len", 10))
    if not pwd1 or len(pwd1) < min_len:
        st.error(f"Hasło musi mieć co najmniej {min_len} znaków.")
        return

    if pwd1 != pwd2:
        st.error("Hasła nie są identyczne.")
        return

    inv = str(policy.get("invite_code", "")).strip()
    if inv and inv != (invite or "").strip():
        st.error("Niepoprawny kod zaproszenia.")
        return

    if not accepted:
        st.error("Musisz zaakceptować Regulamin i Politykę Prywatności.")
        return

    # Utwórz rekord pending + wyślij kod
    code = _gen_code(6)
    ttl = int(policy.get("code_ttl_sec", 900))
    exp_ts = _now_ts() + ttl
    pending = _pending()

    # Rate-limit: 1 wysyłka / resend_cooldown_sec
    cooldown = int(policy.get("resend_cooldown_sec", 60))
    last_send = 0
    try:
        # poszukaj ostatniego logu dla emaila
        with VERIFY_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                if j.get("email") == email_lc:
                    last_send = max(last_send, int(j.get("ts", 0)))
    except Exception:
        pass

    if _now_ts() - last_send < cooldown:
        st.error("Zbyt częste próby wysyłki kodu. Spróbuj ponownie za chwilę.")
        return

    # Zapis pending
    hashed = _hash_password(pwd1)
    _save_pending(email_lc, {
        "name": display_name or email_lc,
        "hashed_password": hashed,
        "code": code,
        "exp_ts": exp_ts,
        "attempts": 0,
    })

    if _send_code(email_lc, code):
        _append_jsonl(VERIFY_LOG, {"email": email_lc, "ts": _now_ts(), "event": "code_sent"})
        st.success("Kod weryfikacyjny został wysłany na e-mail. Sprawdź skrzynkę i wprowadź kod w zakładce **Weryfikacja** poniżej.")
        st.session_state["_show_verify"] = True
    else:
        st.error("Nie udało się wysłać kodu. Skontaktuj się z administratorem.")

def _render_verify_ui(policy: dict) -> None:
    st.subheader("✅ Weryfikacja konta (kod e-mail)")
    with st.form("verify_form"):
        email = st.text_input("E-mail (taki sam jak przy rejestracji)")
        code = st.text_input("Kod weryfikacyjny", max_chars=6)
        submit = st.form_submit_button("Potwierdź")

    if not submit:
        return

    email_lc = (email or "").strip().lower()
    pend = _pending().get("pending", {})
    rec = pend.get(email_lc)
    if not rec:
        st.error("Nie znaleziono oczekującej rejestracji dla tego e-maila.")
        return

    if _now_ts() > int(rec.get("exp_ts", 0)):
        st.error("Kod wygasł. Zarejestruj się ponownie, aby otrzymać nowy kod.")
        _del_pending(email_lc)
        return

    # Próby
    rec_attempts = int(rec.get("attempts", 0)) + 1
    if rec_attempts > int(policy.get("max_verify_attempts", 5)):
        st.error("Przekroczono liczbę prób. Zarejestruj się ponownie.")
        _del_pending(email_lc)
        return

    if (code or "").strip() != str(rec.get("code", "")):
        # zapisz próbę
        rec["attempts"] = rec_attempts
        _save_pending(email_lc, rec)
        st.error("Niepoprawny kod.")
        return

    # Sukces: przenieś do `users.json`
    _save_user(email_lc, rec.get("name", email_lc), rec.get("hashed_password", ""))
    _del_pending(email_lc)
    st.session_state.pop("_auth_config", None)  # odśwież cache
    st.success("Konto zweryfikowane ✅ Możesz się zalogować.")
    st.experimental_rerun()

# --------------------------- Public API ---------------------------

def require_login() -> dict:
    """Zakładki: Logowanie / Rejestracja / (Weryfikacja), a na końcu zwrot zalogowanego usera."""
    if "_current_user" in st.session_state and st.session_state["_current_user"]:
        return st.session_state["_current_user"]

    pol = _policy()
    tabs = ["🔑 Logowanie"]
    allow_signup = bool(pol.get("allow_self_signup", False))
    if allow_signup:
        tabs += ["🆕 Rejestracja", "✅ Weryfikacja"]

    t = st.tabs(tabs)

    # LOGOWANIE
    with t[0]:
        authenticator = _make_authenticator()
        name, auth_status, username = authenticator.login(location="main", max_login_attempts=3, key="login_box")
        if auth_status is True:
            st.session_state["_current_user"] = {"email": username, "name": name}
        elif auth_status is False:
            st.error("Błędny e-mail lub hasło.")
        else:
            st.info("Zaloguj się, aby korzystać z aplikacji.")
            if allow_signup:
                st.caption("Nie masz konta? Rejestracja i weryfikacja w kolejnych zakładkach.")

    if allow_signup:
        with t[1]:
            _render_signup_ui(pol)
        with t[2]:
            _render_verify_ui(pol)

    if not st.session_state.get("_current_user"):
        st.stop()
    return st.session_state["_current_user"]

def auth_sidebar() -> None:
    authenticator = _make_authenticator()
    with st.sidebar:
        st.caption("🔐 **Sesja**")
        user = st.session_state.get("_current_user", {})
        if user:
            st.write(f"Zalogowano jako: **{user.get('name', user.get('email'))}**")
        authenticator.logout("Wyloguj", "sidebar")

def current_user() -> dict | None:
    return st.session_state.get("_current_user")
