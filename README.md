# ML Training Platform
- Biblioteka ML/DL + Mentor RAG + Interview Prep + Fiszki
- Uruchom lokalnie: `pip install -r requirements.txt && streamlit run app.py`

## 🔒 Bezpieczne logowanie e‑mailem

W aplikacji włączono logowanie przy użyciu **adresu e‑mail + hasła (bcrypt)** dzięki `streamlit-authenticator`.

### Konfiguracja
1. Zainstaluj zależności (dodano do `requirements.txt`): `streamlit-authenticator`, `bcrypt`.
2. Skopiuj `.streamlit/secrets.example.toml` do `.streamlit/secrets.toml` i uzupełnij:
   - `cookie_key` → mocny, losowy sekret.
   - Listę `emails`, `names` (opcjonalnie) i `hashed_passwords` (bcrypt).
3. Uruchom aplikację. Przy pierwszym uruchomieniu bez `secrets.toml` dostępne jest **konto demo**: `demo@local / demo1234` (tylko lokalnie).

### Generowanie hashy (bcrypt)
W Pythonie:
```python
import streamlit_authenticator as stauth
hashes = stauth.Hasher(["TwojeHaslo1", "InneHaslo2"]).generate()
print(hashes)
```

### Dobre praktyki
- W produkcji **wymagaj HTTPS** (np. reverse proxy z certyfikatem).
- Przechowuj `secrets.toml` poza repozytorium (CI/CD → sekrety).
- Stosuj silne hasła i rotację, `cookie_expiry_days` dostosuj do ryzyka.
- Rozważ SSO (Auth0 / Azure AD / Supabase Auth) — możliwe do podłączenia później.
