# demo_ml_dl_app.py
# Uproszczona wersja pokazowa ML / DL Training Platform
# uruchom: streamlit run demo_ml_dl_app.py

import streamlit as st
import pandas as pd
from datetime import datetime
import random

st.set_page_config(
    page_title="ML / DL Training Platform",
    page_icon="🤖",
    layout="wide",
)

# === HERO ===
st.markdown(
    """
    <div style="
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #1e293b 45%, #312e81 100%);
        padding: 1.6rem 1.4rem 1.1rem 1.4rem;
        border-radius: 1.5rem;
        color: #e2e8f0;
        margin-bottom: 1.0rem;
    ">
        <h1 style="margin-bottom: .3rem;">🤖 ML / DL Training Platform</h1>
        <p style="margin-bottom: .25rem; opacity: .9;">
            Platforma do nauki i ćwiczenia tematów z uczenia maszynowego, głębokiego i MLOps.
        </p>
        <p style="opacity: 0.6; font-size: 0.73rem;">
            Moduły: biblioteka, mentor AI, sekcja rekrutacyjna, fiszki, plan nauki, statystyki.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# === GÓRNE KAFELKI ===
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📚 Materiały", "48+", "ML / DL / MLOps")
with col2:
    st.metric("🧠 Tryb AI", "aktywny", "Q&A")
with col3:
    st.metric("💬 Pytania", "120+", "rekrutacyjne")
with col4:
    st.metric("📆 Aktualizacja", datetime.today().strftime("%Y-%m-%d"))

st.markdown("")

# === ZAKŁADKI GŁÓWNE ===
(
    tab_overview,
    tab_library,
    tab_mentor,
    tab_interview,
    tab_flashcards,
    tab_learningplan,
    tab_stats,
) = st.tabs(
    [
        "1️⃣ Przegląd",
        "2️⃣ Biblioteka ML/DL",
        "3️⃣ Mentor AI",
        "4️⃣ Rekrutacja",
        "5️⃣ Fiszki",
        "6️⃣ Plan nauki",
        "7️⃣ Postępy / Statystyki",
    ]
)

# ========== 1. PRZEGLĄD ==========
with tab_overview:
    c1, c2 = st.columns((1.1, 0.9), gap="large")
    with c1:
        st.subheader("Struktura platformy")
        st.markdown(
            """
            - **Biblioteka ML/DL** – gotowe lekcje, artykuły, notebooki
            - **Mentor AI** – pytania i szybkie odpowiedzi na zagadnienia ML
            - **Rekrutacja ML** – pytania techniczne + miejsce na własne odpowiedzi
            - **Fiszki** – szybka powtórka pojęć
            - **Plan nauki** – tygodniowy układ materiałów
            - **Postępy / statystyki** – podsumowanie aktywności (mock)
            """
        )
        st.markdown("**Cele platformy:**")
        st.markdown(
            "- uporządkować wiedzę ML/DL\n"
            "- mieć jedno miejsce z pytaniami na rozmowę\n"
            "- mieć prosty interfejs do zadawania pytań AI\n"
            "- możliwość rozbudowy o RAG / OpenAI"
        )

# ========== 2. BIBLIOTEKA ==========
with tab_library:
    st.subheader("📚 Biblioteka ML / DL")
    st.write("Filtruj i przeglądaj przykładowe materiały.")

    sample_data = [
        {
            "tytuł": "Wprowadzenie do uczenia nadzorowanego",
            "poziom": "beginner",
            "typ": "notebook",
            "tagi": "supervised,regression,classification",
        },
        {
            "tytuł": "Konwolucyjne sieci neuronowe (CNN)",
            "poziom": "intermediate",
            "typ": "article",
            "tagi": "cv,deep learning",
        },
        {
            "tytuł": "Feature Engineering dla danych tabelarycznych",
            "poziom": "intermediate",
            "typ": "notebook",
            "tagi": "feature engineering,ml",
        },
        {
            "tytuł": "MLOps – wprowadzenie do MLflow",
            "poziom": "advanced",
            "typ": "video",
            "tagi": "mlops,mlflow,prod",
        },
        {
            "tytuł": "Nienadzorowane uczenie – clustering",
            "poziom": "intermediate",
            "typ": "article",
            "tagi": "unsupervised,kmeans,segmentation",
        },
    ]
    df = pd.DataFrame(sample_data)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        level = st.selectbox(
            "Poziom",
            options=["(wszystkie)", "beginner", "intermediate", "advanced"],
            index=0,
        )
    with col_b:
        res_type = st.selectbox(
            "Typ materiału",
            options=["(wszystkie)", "notebook", "article", "video"],
            index=0,
        )
    with col_c:
        search = st.text_input("Szukaj w tytule / tagach")

    filtered = df.copy()
    if level != "(wszystkie)":
        filtered = filtered[filtered["poziom"] == level]
    if res_type != "(wszystkie)":
        filtered = filtered[filtered["typ"] == res_type]
    if search:
        s = search.lower()
        filtered = filtered[
            filtered["tytuł"].str.lower().str.contains(s)
            | filtered["tagi"].str.lower().str.contains(s)
        ]

    st.write(f"Znaleziono: **{len(filtered)}** materiał(y).")
    st.dataframe(filtered, use_container_width=True)
    st.caption("W pełnej wersji lista może być ładowana z bazy lub pliku konfiguracyjnego.")

# ========== 3. MENTOR AI ==========
with tab_mentor:
    st.subheader("🧠 Mentor AI")
    st.write("Zadaj pytanie związane z ML / DL. Odpowiedź jest generowana lokalnie (symulacja).")

    user_q = st.text_area(
        "Twoje pytanie:",
        value="Na czym polega walidacja krzyżowa?",
        height=90,
    )
    tone = st.selectbox("Styl odpowiedzi", ["zwięzły", "techniczny", "dla początkujących"])

    knowledge_hint = st.multiselect(
        "Zakres tematyczny (pomaga dobrać odpowiedź):",
        ["uczenie nadzorowane", "uczenie nienadzorowane", "przygotowanie danych", "metryki", "deep learning"],
        default=["uczenie nadzorowane"],
    )

    if st.button("🔍 Odpowiedz"):
        base_answer = (
            "Walidacja krzyżowa (cross-validation) dzieli dane na kilka części (foldów). "
            "Model trenuje się na części z nich, a testuje na pozostałej. "
            "Uśrednienie wyników pozwala lepiej oszacować jakość modelu."
        )
        if tone == "techniczny":
            base_answer += (
                " Typowo używa się k-fold (np. k=5). Daje to 5 modeli i 5 wyników metryki, które można uśrednić. "
                "W zadaniach z małą ilością danych to podejście jest bardziej stabilne niż pojedynczy podział."
            )
        elif tone == "dla początkujących":
            base_answer = (
                "Zamiast sprawdzać model tylko raz, sprawdzasz go kilka razy na różnych kawałkach danych. "
                "Dzięki temu widzisz, czy model jest naprawdę dobry, a nie miał szczęście."
            )

        if knowledge_hint:
            base_answer += f"\n\n(uwzględniono zakres: {', '.join(knowledge_hint)})"

        st.success("Odpowiedź:")
        st.write(base_answer)

# ========== 4. REKRUTACJA ==========
with tab_interview:
    st.subheader("💼 Pytania rekrutacyjne – ML")
    st.write("Losuj pytanie i zapisz swoją odpowiedź.")

    questions = [
        "Czym różni się uczenie nadzorowane od nienadzorowanego?",
        "Co to jest data leakage i jak go uniknąć?",
        "Wyjaśnij różnice między MAE, MSE i RMSE.",
        "Dlaczego accuracy nie nadaje się do niezbalansowanych klas?",
        "Na czym polega One-Hot Encoding?",
        "Jak działa regularizacja L1 i L2?",
        "Co to jest walidacja krzyżowa?",
    ]

    q = random.choice(questions)
    st.markdown(f"**Pytanie:** {q}")

    user_ans = st.text_area("Twoja odpowiedź (notatka):", height=110)
    if st.button("💾 Zapisz notatkę"):
        st.info("Tryb pokazowy – notatka nie jest trwale zapisywana.")
        if user_ans.strip():
            st.write("Twoja odpowiedź:")
            st.write(user_ans)

    st.caption("Można rozbudować o oceny odpowiedzi, poziom trudności, eksport do PDF/CV.")

# ========== 5. FISZKI ==========
with tab_flashcards:
    st.subheader("🃏 Fiszki – szybka powtórka")
    st.write("Wybierz zakres i losuj fiszkę.")

    flashcards = {
        "podstawy": [
            ("Uczenie nadzorowane", "Model uczy się na parach (X, y)."),
            ("Uczenie nienadzorowane", "Model szuka struktur w samych X."),
            ("Overfitting", "Model za bardzo dopasowany do danych treningowych."),
        ],
        "preprocessing": [
            ("Normalizacja", "Sprowadzenie cech do podobnej skali."),
            ("One-Hot Encoding", "Zakodowanie zmiennej kategorycznej na wektor 0/1."),
        ],
        "metryki": [
            ("Accuracy", "Udział poprawnych predykcji."),
            ("Precision", "Ile z pozytywnych predykcji było poprawnych."),
            ("Recall", "Ile z prawdziwych pozytywów wykryto."),
        ],
    }

    scope = st.selectbox("Zakres", list(flashcards.keys()), index=0)
    if st.button("🎲 Losuj fiszkę"):
        term, desc = random.choice(flashcards[scope])
        st.markdown(f"**{term}**")
        st.write(desc)

# ========== 6. PLAN NAUKI ==========
with tab_learningplan:
    st.subheader("📅 Plan nauki (przykładowy tydzień)")
    st.write("Prosty plan do przejścia materiałów ML/DL.")

    plan = {
        "Poniedziałek": "Podstawy ML, supervised vs unsupervised",
        "Wtorek": "Przygotowanie danych, brakujące wartości, kategoryczne",
        "Środa": "Modele klasyfikacji (logreg, tree)",
        "Czwartek": "Walidacja krzyżowa, metryki",
        "Piątek": "Wprowadzenie do DL",
        "Sobota": "Notebook z ćwiczeniami",
        "Niedziela": "Powtórka + fiszki",
    }

    for day, task in plan.items():
        st.markdown(f"- **{day}** – {task}")

    st.caption("W wersji pełnej można zapisywać plan per użytkownik i oznaczać zrobione lekcje.")

# ========== 7. POSTĘPY / STATYSTYKI ==========
with tab_stats:
    st.subheader("📊 Postępy")
    st.write("Podgląd przykładowych statystyk użytkownika (wartości przykładowe).")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Przerobione materiały", "12", "+2 w tym tygodniu")
    with col_b:
        st.metric("Sesje z Mentorem AI", "7", "+3")
    with col_c:
        st.metric("Powtórzone fiszki", "34", "+10")

    st.write("Ostatnie aktywności:")
    activities = pd.DataFrame(
        [
            {"data": "2025-11-05", "akcja": "Fiszki – metryki"},
            {"data": "2025-11-05", "akcja": "Pytanie do Mentora: 'data leakage'"},
            {"data": "2025-11-04", "akcja": "Biblioteka – CNN"},
        ]
    )
    st.table(activities)

# === STOPKA ===
st.markdown(
    """
    <hr style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <p style="font-size: 0.7rem; opacity: 0.6;">
        ML / DL Training Platform • Streamlit • modułowa architektura • gotowe do rozbudowy o RAG / auth.
    </p>
    """,
    unsafe_allow_html=True,
)
