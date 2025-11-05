# 🤖 ML / DL Training Platform

Platforma edukacyjna i narzędziowa łącząca **praktyczne szkolenia z uczenia maszynowego i głębokiego**, **mentora AI (RAG)**, **przygotowanie do rozmów rekrutacyjnych** oraz **fiszki wiedzy ML** — wszystko w jednym miejscu.  
Zbudowana w oparciu o **Streamlit**, **Python 3.11+** i architekturę modułową gotową do rozwoju.

---

## 🚀 Szybki start (tryb DEV)

```bash
git clone https://github.com/<TwojeRepo>/ML-DL-Training-Platform.git
cd ML-DL-Training-Platform
pip install -r requirements.txt
streamlit run app.py
```

📦 **Wymagany Python:** 3.11+  
🔑 **Konfiguracja e-maili:** `.streamlit/secrets.toml` (patrz niżej)

---

## 🔒 Logowanie, rejestracja i weryfikacja e-mail

System logowania i rejestracji z weryfikacją adresu e-mail został zaimplementowany w module:

> **`auth_email_verify.py`**

### Jak to działa

1. Użytkownik rejestruje się w aplikacji.  
2. Dane trafiają do pliku:
   ```
   .streamlit/users_pending.json
   ```
3. Na adres e-mail wysyłany jest **kod weryfikacyjny** (ważny 15 minut).  
4. Po potwierdzeniu użytkownik zostaje przeniesiony do:
   ```
   .streamlit/users.json
   ```
5. Można się już zalogować 🎉

---

### 📬 Konfiguracja SMTP (`.streamlit/secrets.toml`)

Utwórz plik z danymi logowania do serwera SMTP:

```toml
[email]
host = "smtp.gmail.com"
port = 587
username = "twoj_email@gmail.com"
password = "twoje_haslo_aplikacji"
from = "ML Training Platform <twoj_email@gmail.com>"
```

📘 **Wskazówka:** Dla Gmaila utwórz [hasło aplikacji](https://myaccount.google.com/apppasswords), nie używaj głównego hasła.

---

## 🧠 Główne funkcje

| Kategoria | Opis |
|------------|------|
| 🧩 **Biblioteka ML/DL** | Interaktywne moduły uczenia maszynowego i głębokiego |
| 🧭 **Mentor AI (RAG)** | Odpowiada na pytania i tłumaczy pojęcia ML |
| 🎯 **Interview Prep** | Pytania i symulacje rozmów rekrutacyjnych |
| 🃏 **Fiszki Wiedzy** | Dynamiczne fiszki do nauki pojęć ML/DL |
| 🔐 **Weryfikacja e-mail** | Rejestracja z potwierdzeniem e-mail (TTL 15 min) |
| 📝 **Dokumentacja i Polityki** | [Regulamin](docs/Terms.md) i [Polityka prywatności](docs/Privacy.md) |

---

## 🧩 Struktura projektu

```bash
ML-DL-Training-Platform/
├── app.py                  # Główna aplikacja Streamlit
├── auth.py                 # Logowanie i zarządzanie sesją
├── auth_email_verify.py    # Rejestracja i weryfikacja e-mail
├── requirements.txt        # Lista zależności
├── Procfile                # Dla wdrożenia np. na Heroku
├── .streamlit/
│   ├── config.toml         # Konfiguracja UI Streamlit
│   ├── users.json          # Baza użytkowników
│   └── users_pending.json  # Użytkownicy oczekujący na potwierdzenie
├── docs/
│   ├── Terms.md            # Regulamin
│   └── Privacy.md          # Polityka prywatności
└── README.md
```

---

## ⚙️ Konfiguracja środowiska

1. Skopiuj plik przykładowy:
   ```bash
   cp .streamlit/secrets.example.toml .streamlit/secrets.toml
   ```
2. Uzupełnij dane SMTP.  
3. (Opcjonalnie) ustaw port lub motyw UI w `.streamlit/config.toml`.

---

## 🧑‍💻 Dla deweloperów

### Instalacja zależności
```bash
pip install -r requirements.txt
```

### Uruchomienie aplikacji
```bash
streamlit run app.py
```

### Lokalna baza użytkowników
Wszystkie konta są przechowywane lokalnie w:
```
.streamlit/users.json
```
Dzięki temu aplikacja działa w pełni **offline** (bez bazy SQL).

---

## 🧩 Integracje i przyszłe rozszerzenia

- 🔗 Integracja z **OpenAI API** (mentor konwersacyjny / RAG)  
- 📦 Import projektów z **GitHub / Kaggle**  
- 🌐 Tryb **offline learning** – dostęp do materiałów bez sieci  
- 📊 Eksport wyników nauki i progresu użytkownika  

---

## 📘 Dokumentacja

- [📜 Regulamin (Terms.md)](docs/Terms.md)  
- [🔒 Polityka Prywatności (Privacy.md)](docs/Privacy.md)

---

## 🧑‍💼 Autor

**Mateusz Marks**  
Specjalista ds. Kalkulacji i Data Science  

📧 **marks.mateusz@wp.pl**  
🔗 [LinkedIn](https://www.linkedin.com/in/mateusz-marks-794326397)  
💻 [GitHub](https://github.com/Marksio90)

---

## 🪪 Licencja

Ten projekt jest udostępniony na licencji **MIT** — możesz go swobodnie rozwijać i modyfikować z zachowaniem informacji o autorze.  
📄 Zobacz plik `LICENSE` *(jeśli dodasz go w repozytorium)*.

---

## ⭐ Wsparcie projektu

Jeśli ta platforma Ci się podoba:
- 🌟 Dodaj gwiazdkę na GitHubie  
- 🧩 Zgłoś sugestie lub błędy w zakładce **Issues**  
- 🤝 Dołącz do rozwoju wersji PRO  

---

> 🧠 *„Nauka ML to podróż. Ta platforma to Twój przewodnik.”* 🚀
