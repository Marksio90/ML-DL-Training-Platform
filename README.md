# ML Training Platform
- Biblioteka ML/DL + Mentor RAG + Interview Prep + Fiszki
- Uruchom lokalnie: `pip install -r requirements.txt && streamlit run app.py`

## 🔒 Logowanie + Rejestracja z weryfikacją e‑mail

- Moduł: `auth_email_verify.py` (zakładki **Logowanie / Rejestracja / Weryfikacja**).
- Nowe konta trafiają do `.streamlit/users_pending.json` i wymagają **kodu e‑mail** (TTL 15 min).
- Po potwierdzeniu są przenoszone do `.streamlit/users.json`.
- Skonfiguruj SMTP w `.streamlit/secrets.toml` → sekcja `[email]`.
- Regulamin i Polityka: `docs/Terms.md`, `docs/Privacy.md` (podlinkuj w UI, np. w stopce).

Szybki start (DEV):
1. Skopiuj `.streamlit/secrets.example.toml` do `.streamlit/secrets.toml` i uzupełnij SMTP.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
