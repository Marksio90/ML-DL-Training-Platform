# app.py — ML/DL Training Platform — Encyclopedia (PL/EN, compact full)
from __future__ import annotations
import os, json, re, sqlite3, textwrap, random
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import time

# ====== SŁOWNIK TŁUMACZEŃ ======
translations = {
    "Widok / View": {"pl": "Widok", "en": "View"},
    "Lista": {"pl": "Lista", "en": "List"},
    "Tabela": {"pl": "Tabela", "en": "Table"},
    "Kafelki": {"pl": "Kafelki", "en": "Tiles"},
    "Rok": {"pl": "Rok", "en": "Year"},
    "Dodaj zasób": {"pl": "Dodaj zasób", "en": "Add Resource"},
    "Darmowe": {"pl": "Darmowe", "en": "Free"},
    "Autor": {"pl": "Autor", "en": "Author"},
    "Szacowany czas": {"pl": "Szacowany czas", "en": "Estimated time"},
    "Minuty": {"pl": "Minuty", "en": "Minutes"},
    "Zapisz": {"pl": "Zapisz", "en": "Save"},
    "Anuluj": {"pl": "Anuluj", "en": "Cancel"},
    "Biblioteka": {"pl": "Biblioteka", "en": "Library"},
    "Mentor AI": {"pl": "Mentor AI", "en": "AI Mentor"},
    "Notatki": {"pl": "Notatki", "en": "Notes"},
    "Zakładki": {"pl": "Zakładki", "en": "Bookmarks"},
    "Wyszukaj": {"pl": "Wyszukaj", "en": "Search"},
    "Reset": {"pl": "Resetuj", "en": "Reset"},
    "Dodaj": {"pl": "Dodaj", "en": "Add"},
    "Edytuj": {"pl": "Edytuj", "en": "Edit"},
    "Usuń": {"pl": "Usuń", "en": "Delete"},
}
# Funkcja pomocnicza
def T(text, lang="pl"):
    return translations.get(text, {}).get(lang, text)

# ---- Optional deps (bez twardych wymagań) ----
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

st.set_page_config(page_title="ML/DL Training Platform", page_icon="🚀", layout="wide")

# ====================== I18N ======================
LANGUAGES = {'pl': 'Polski', 'en': 'English'}

def _texts_en():
    return {
        'title':'ML/DL Training Platform — Encyclopedia',
        'subtitle':'Essential resources for ML/DL/AI + RAG mentor',
        'dashboard':'Dashboard','library':'Library','mentor':'AI Mentor',
        'snippets':'Snippets','glossary':'Glossary','interview':'Interview Prep',
        'cases':'Case Studies & Tutorials','roadmaps':'Roadmaps','add_resource':'Add Resource',
        'settings':'Settings','language':'Language','openai_key':'OpenAI Key',
        'reset_data':'Reset Data','confirm_reset':'Confirm full reset','data_reset':'Data has been reset',
        'filters':'Filters','category':'Category','type':'Type','format':'Format',
        'search':'Search (title/desc/tags)','tags':'Tags (comma-separated)','open_resource':'Open resource',
        'mark_done':'Mark completed','marked_done':'Completed ✓','found_n':'Found {n} resources',
        'mentor_intro':'Ask the ML/DL Mentor (has access to the library)','ask':'Ask',
        'tone':'Tone','tone_opts':['concise','gentle','harsh','academic'],
        'length':'Length','length_opts':['short','medium','detailed'],'model':'Model',
        'context_toggle':'Attach library context (RAG)','topk':'Context size','no_key':'No OpenAI key — OFFLINE mode.',
        'answer':'Mentor Answer','used_sources':'Context sources','resources_by_cat':'Resources by category',
        'top_rated':'Top Rated','recent_activity':'Recent activity','study_hours':'Study hours',
        'completed':'Completed','streak':'Streak','none':'No data','role':'Role','weeks':'Duration (weeks)',
        'hours_week':'Hours per week','generate_plan':'Generate roadmap','milestones':'Milestones',
        'choose_snippet':'Choose a snippet','desc':'Description','level':'Level','author':'Author',
        'year':'Year','free':'Free','views':'Views','rating':'Rating','case_intro':'Hands-on case studies',
        'interview_intro':'Practice: Classic ML / DL / LLMs / MLOps / System Design','run_case':'Open tutorial',
        'export':'Export (CSV)','import':'Import (JSON)','import_btn':'Import','download_btn':'Download CSV',
        'all':'(All)','url':'URL',
        # NEW for view toggle:
        'view':'View','list':'List','table':'Table','tiles':'Tiles'
    }

TEXTS = {
    'en': _texts_en(),
    'pl': {**_texts_en(),
        'title':'ML/DL Training Platform — Encyklopedia',
        'subtitle':'Najważniejsze materiały dla ML/DL/AI + mentor z RAG',
        'library':'Biblioteka','mentor':'Mentor AI','snippets':'Snippety','glossary':'Słownik pojęć',
        'interview':'Przygotowanie do rozmów','cases':'Case Studies i Tutoriale','roadmaps':'Mapy drogowe',
        'add_resource':'Dodaj zasób','settings':'Ustawienia','language':'Język','openai_key':'Klucz OpenAI',
        'reset_data':'Reset danych','confirm_reset':'Potwierdź reset wszystkich danych',
        'data_reset':'Dane zostały zresetowane','filters':'Filtry','category':'Kategoria','type':'Typ',
        'format':'Format','search':'Szukaj (tytuł/opis/tagi)','tags':'Tagi (przecinki)',
        'open_resource':'Otwórz zasób','mark_done':'Oznacz ukończone','marked_done':'Ukończone ✓',
        'found_n':'Znaleziono {n} zasobów','mentor_intro':'Zadaj pytanie Mentorowi ML/DL (ma dostęp do biblioteki)',
        'ask':'Pytanie','tone':'Styl','tone_opts':['zwięzły','łagodny','surowy','akademicki'],
        'length':'Długość','length_opts':['krótka','średnia','szczegółowa'],
        'context_toggle':'Dołącz kontekst z biblioteki (RAG)','topk':'Rozmiar kontekstu',
        'no_key':'Brak klucza OpenAI — tryb OFFLINE.','answer':'Odpowiedź Mentora','used_sources':'Źródła kontekstu',
        'resources_by_cat':'Zasoby według kategorii','top_rated':'Najwyżej oceniane',
        'recent_activity':'Ostatnia aktywność','study_hours':'Godziny nauki','completed':'Ukończone','streak':'Seria dni',
        'none':'Brak danych','role':'Rola','weeks':'Czas trwania (tygodnie)','hours_week':'Godzin tygodniowo',
        'generate_plan':'Generuj mapę drogową','milestones':'Kamienie milowe','choose_snippet':'Wybierz snippet',
        'desc':'Opis','level':'Poziom','author':'Autor','year':'Rok','free':'Darmowy','views':'Wyświetlenia',
        'rating':'Ocena','case_intro':'Praktyczne studia przypadków','interview_intro':'Praktyka: ML klasyczny / DL / LLMs / MLOps / System Design',
        'run_case':'Otwórz tutorial','export':'Eksport (CSV)','import':'Import (JSON)','import_btn':'Importuj',
        'download_btn':'Pobierz CSV','all':'(Wszystkie)','url':'URL',
        # NEW for view toggle:
        'view':'Widok','list':'Lista','table':'Tabela','tiles':'Kafelki'
    }
}
def T(k, lang): return TEXTS.get(lang, TEXTS['en']).get(k, TEXTS['en'].get(k, k))

# ====================== Session ======================
def init_session_state():
    if 'language' not in st.session_state: st.session_state.language = 'pl'
    if 'completed_resources' not in st.session_state: st.session_state.completed_resources = []
    if 'study_hours' not in st.session_state: st.session_state.study_hours = 0
    if 'current_streak' not in st.session_state: st.session_state.current_streak = 0
    if 'openai_key' not in st.session_state or not st.session_state.openai_key:
        st.session_state.openai_key = os.getenv('OPENAI_API_KEY') or None  # trzyma się do końca sesji

# ====================== DB ======================
DB_PATH = 'ml_platform.db'
def connect_db(): return sqlite3.connect(DB_PATH)
def ensure_schema(conn: sqlite3.Connection):
    c = conn.cursor()

    # --- Główne tabele ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS resources (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        url TEXT NOT NULL,
        category TEXT NOT NULL,
        type TEXT NOT NULL,
        format TEXT,
        tags TEXT,
        description TEXT,
        difficulty TEXT,
        rating REAL DEFAULT 0,
        views INTEGER DEFAULT 0,
        author TEXT,
        year INTEGER,
        length_min INTEGER,
        free INTEGER DEFAULT 1,
        added_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS user_activity (
        id INTEGER PRIMARY KEY,
        resource_id INTEGER,
        activity_type TEXT,
        meta TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Zakładki / Notatki
    c.execute("""
    CREATE TABLE IF NOT EXISTS bookmarks (
        id INTEGER PRIMARY KEY,
        resource_id INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY,
        resource_id INTEGER,
        body TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Kolekcje (ścieżki) i pozycje w kolekcjach
    c.execute("""
    CREATE TABLE IF NOT EXISTS collections (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS collection_items (
        id INTEGER PRIMARY KEY,
        collection_id INTEGER,
        resource_id INTEGER,
        ord INTEGER DEFAULT 0
    )
    """)
    conn.commit()

    # --- Uzupełnianie brakujących kolumn, gdy baza ze starej wersji ---
    cols = {row[1] for row in c.execute('PRAGMA table_info(resources)').fetchall()}
    required = [
        ("type",       "TEXT NOT NULL DEFAULT 'article'"),
        ("format",     "TEXT"),
        ("tags",       "TEXT"),
        ("description","TEXT"),
        ("difficulty", "TEXT"),
        ("rating",     "REAL DEFAULT 0"),
        ("views",      "INTEGER DEFAULT 0"),
        ("author",     "TEXT"),
        ("year",       "INTEGER"),
        ("length_min", "INTEGER"),
        ("free",       "INTEGER DEFAULT 1"),
        ("added_at",   "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ]
    for col, decl in required:
        if col not in cols:
            c.execute(f'ALTER TABLE resources ADD COLUMN "{col}" {decl}')
    conn.commit()

    # --- Indeksy ---
    c.execute('CREATE INDEX IF NOT EXISTS idx_res_category ON resources(category)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_res_type     ON resources(type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_res_format   ON resources(format)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_res_rating   ON resources(rating)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_res_views    ON resources(views)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_res_title    ON resources(title)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_res_tags     ON resources(tags)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_bm_res       ON bookmarks(resource_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_notes_res    ON notes(resource_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_col_items    ON collection_items(collection_id, ord)')
    conn.commit()

    # --- FTS5: pełnotekstowa wyszukiwarka (lustrzany indeks) ---
    c.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS resources_fts USING fts5(
        title, description, tags, category,
        content='resources', content_rowid='id'
    )
    """)
    # Triggery sync FTS5
    c.execute("""
    CREATE TRIGGER IF NOT EXISTS resources_ai AFTER INSERT ON resources BEGIN
      INSERT INTO resources_fts(rowid, title, description, tags, category)
      VALUES (new.id, new.title, new.description, new.tags, new.category);
    END;""")
    c.execute("""
    CREATE TRIGGER IF NOT EXISTS resources_ad AFTER DELETE ON resources BEGIN
      INSERT INTO resources_fts(resources_fts, rowid, title, description, tags, category)
      VALUES('delete', old.id, old.title, old.description, old.tags, old.category);
    END;""")
    c.execute("""
    CREATE TRIGGER IF NOT EXISTS resources_au AFTER UPDATE ON resources BEGIN
      INSERT INTO resources_fts(resources_fts, rowid, title, description, tags, category)
      VALUES('delete', old.id, old.title, old.description, old.tags, old.category);
      INSERT INTO resources_fts(rowid, title, description, tags, category)
      VALUES (new.id, new.title, new.description, new.tags, new.category);
    END;""")
    # Jednorazowy dosyp do FTS, jeśli pusty (migration)
    need_fill = c.execute("SELECT (SELECT COUNT(*) FROM resources) > (SELECT COUNT(*) FROM resources_fts)").fetchone()[0]
    if need_fill:
        c.execute("INSERT INTO resources_fts(rowid, title, description, tags, category) SELECT id,title,description,tags,category FROM resources")
    conn.commit()

   
def add_resources_bulk(conn, rows: List[Tuple]):
    conn.executemany('''INSERT INTO resources
    (title,url,category,type,format,tags,description,difficulty,rating,author,year,length_min,free)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''', rows)
    conn.commit()

   
def db_count(conn): return conn.execute('SELECT COUNT(*) FROM resources').fetchone()[0]

def build_default_resources(lang: str = "pl") -> List[Tuple]:
    def R(title,url,category,typ,fmt,tags,desc,diff,rate,author,year,length,free=1):
        return (title,url,category,typ,fmt,tags,desc,diff,rate,author,year,length,free)

    rows: List[Tuple] = []

    # --- Core official docs / guides (rozszerzona lista) ---
    core_docs = [
        # Classic ML / Python / Data
        ("Scikit-learn User Guide","https://scikit-learn.org/stable/user_guide.html","Classic ML","documentation","html","sklearn,cv,models",
         "Oficjalna dokumentacja scikit-learn." if lang=="pl" else "Official scikit-learn docs.","intermediate",4.8,"sklearn",2025,240,1),
        ("Pandas User Guide","https://pandas.pydata.org/docs/user_guide/index.html","Data","documentation","html","pandas,etl",
         "Dokumentacja Pandas.","intermediate",4.8,"pandas-dev",2025,180,1),
        ("NumPy Manual","https://numpy.org/doc/stable/","Data","documentation","html","numpy,ndarray",
         "Dokumentacja NumPy.","intermediate",4.8,"NumPy",2025,160,1),
        ("Polars User Guide","https://docs.pola.rs/user-guide/","Data","documentation","html","polars,lazyframe",
         "Przewodnik Polars.","intermediate",4.6,"Polars",2025,120,1),
        ("DuckDB Docs","https://duckdb.org/docs/","Data","documentation","html","duckdb,olap","Dokumentacja DuckDB.","intermediate",4.6,"DuckDB",2025,100,1),

        # Deep Learning / Frameworks
        ("Deep Learning Book","https://www.deeplearningbook.org/","Deep Learning","book","html","dl,nn,optim",
         "Klasyczna książka DL.","advanced",4.8,"Goodfellow/Bengio/Courville",2016,800,1),
        ("PyTorch Docs","https://pytorch.org/docs/stable/index.html","Deep Learning","documentation","html","pytorch,autograd",
         "Oficjalne PyTorch docs.","intermediate",4.8,"PyTorch",2025,200,1),
        ("Lightning Docs","https://lightning.ai/docs/pytorch/stable/","Deep Learning","documentation","html","pytorch,training",
         "PyTorch Lightning.","intermediate",4.6,"Lightning",2025,120,1),
        ("TensorFlow Guides","https://www.tensorflow.org/guide","Deep Learning","documentation","html","tensorflow,keras",
         "Przewodniki TF.","intermediate",4.6,"Google",2025,180,1),

        # NLP / LLMs
        ("Hugging Face Transformers","https://huggingface.co/docs/transformers/index","LLMs","documentation","html","transformers,bert,gpt",
         "Dokumentacja Transformers.","intermediate",4.8,"HF",2025,220,1),
        ("HF Datasets","https://huggingface.co/docs/datasets","NLP","documentation","html","datasets,hf",
         "Biblioteka Datasets.","intermediate",4.6,"HF",2025,120,1),
        ("HF Tokenizers","https://huggingface.co/docs/tokenizers","NLP","documentation","html","tokenizers,hf",
         "Tokenizers.","intermediate",4.5,"HF",2025,80,1),
        ("PEFT Docs","https://huggingface.co/docs/peft/index","LLMs","documentation","html","peft,lora,qlora",
         "PEFT/LoRA/QLoRA.","intermediate",4.6,"HF",2025,60,1),
        ("TRL (RLHF) Docs","https://huggingface.co/docs/trl/index","LLMs","documentation","html","trl,rlhf",
         "TRL/RLHF.","advanced",4.6,"HF",2025,60,1),
        ("Prompt Engineering Guide","https://www.promptingguide.ai/","LLMs","guide","html","prompting",
         "Przewodnik po prompt-engineering.","intermediate",4.7,"PEG",2025,90,1),

        # RAG / LLM app stacks
        ("LangChain Docs","https://python.langchain.com/docs/get_started/introduction","LLMs","documentation","html","langchain,rag,agents",
         "LangChain.","intermediate",4.5,"LangChain",2025,120,1),
        ("LlamaIndex Docs","https://docs.llamaindex.ai/","LLMs","documentation","html","llamaindex,rag",
         "LlamaIndex.","intermediate",4.6,"LlamaIndex",2025,120,1),
        ("Haystack Docs","https://docs.haystack.deepset.ai/docs/intro","LLMs","documentation","html","haystack,rag",
         "Haystack.","intermediate",4.5,"deepset",2025,120,1),

        # Vector databases
        ("FAISS","https://github.com/facebookresearch/faiss","Vector DBs","repo","md","faiss,ann",
         "FAISS — biblioteka ANN.","intermediate",4.7,"Meta",2025,120,1),
        ("Qdrant Docs","https://qdrant.tech/documentation/","Vector DBs","documentation","html","qdrant,vector-db",
         "Qdrant.","intermediate",4.6,"Qdrant",2025,140,1),
        ("Weaviate Docs","https://weaviate.io/developers/weaviate","Vector DBs","documentation","html","weaviate,vector-db",
         "Weaviate.","intermediate",4.6,"Weaviate",2025,140,1),
        ("Milvus Docs","https://milvus.io/docs","Vector DBs","documentation","html","milvus,vector-db",
         "Milvus.","intermediate",4.6,"Zilliz",2025,160,1),
        ("pgvector","https://github.com/pgvector/pgvector","Vector DBs","repo","md","pgvector,postgres",
         "pgvector dla Postgresa.","intermediate",4.5,"pgvector",2025,40,1),

        # MLOps / Serving / Optymalizacja
        ("MLflow Docs","https://mlflow.org/docs/latest/index.html","MLOps","documentation","html","mlflow,registry",
         "MLflow — tracking/registry.","intermediate",4.7,"Databricks",2025,140,1),
        ("Weights & Biases","https://docs.wandb.ai/","MLOps","documentation","html","wandb,tracking",
         "W&B — eksperymenty.","intermediate",4.6,"W&B",2025,120,1),
        ("Neptune.ai","https://docs.neptune.ai/","MLOps","documentation","html","neptune,tracking",
         "Neptune.ai.","intermediate",4.5,"Neptune",2025,100,1),
        ("BentoML Docs","https://docs.bentoml.com/","MLOps","documentation","html","serving,bento",
         "BentoML — serving.","intermediate",4.6,"BentoML",2025,120,1),
        ("Seldon Core","https://docs.seldon.io/projects/seldon-core/en/latest/","MLOps","documentation","html","seldon,k8s",
         "Seldon na Kubernetes.","advanced",4.5,"Seldon",2025,160,1),
        ("TorchServe","https://pytorch.org/serve/","MLOps","documentation","html","serving,torch",
         "TorchServe.","intermediate",4.5,"PyTorch",2025,60,1),
        ("TensorFlow Serving","https://www.tensorflow.org/tfx/guide/serving","MLOps","documentation","html","serving,tensorflow",
         "TF Serving.","intermediate",4.5,"Google",2025,80,1),
        ("Triton Inference Server","https://github.com/triton-inference-server/server","MLOps","repo","md","nvidia,triton",
         "NVIDIA Triton.","advanced",4.6,"NVIDIA",2025,100,1),
        ("ONNX Runtime","https://onnxruntime.ai/","MLOps","documentation","html","onnx,inference",
         "ONNX Runtime.","intermediate",4.5,"ONNX",2025,80,1),
        ("OpenVINO","https://docs.openvino.ai/","MLOps","documentation","html","openvino,optimization",
         "OpenVINO — optymalizacja.","advanced",4.6,"Intel",2025,120,1),
        ("TensorRT","https://developer.nvidia.com/tensorrt","MLOps","documentation","html","tensorrt,optimization",
         "TensorRT.","advanced",4.6,"NVIDIA",2025,120,1),

        # Orkiestracja / Data platform
        ("Apache Airflow","https://airflow.apache.org/docs/","Engineering","documentation","html","airflow,orchestration",
         "Airflow — orkiestracja.","intermediate",4.6,"ASF",2025,160,1),
        ("Prefect","https://docs.prefect.io/","Engineering","documentation","html","prefect,flows",
         "Prefect — flows.","intermediate",4.5,"Prefect",2025,120,1),
        ("Dagster","https://docs.dagster.io/","Engineering","documentation","html","dagster,orchestration",
         "Dagster — data orchestration.","intermediate",4.5,"Dagster",2025,120,1),
        ("Kedro","https://docs.kedro.org/","Engineering","documentation","html","kedro,pipelines",
         "Kedro — pipeline’y.","intermediate",4.5,"Kedro",2025,100,1),
        ("DVC","https://dvc.org/doc","Engineering","documentation","html","dvc,data-versioning",
         "DVC — wersjonowanie danych.","intermediate",4.5,"Iterative",2025,80,1),
        ("Pachyderm","https://docs.pachyderm.com/","Engineering","documentation","html","lineage,pipelines",
         "Pachyderm — lineage/pipelines.","advanced",4.5,"Pachyderm",2025,120,1),
        ("Argo Workflows","https://argoproj.github.io/argo-workflows/","Engineering","documentation","html","argo,workflows",
         "Argo Workflows.","advanced",4.5,"Argo",2025,100,1),
        ("Kubernetes","https://kubernetes.io/docs/home/","Engineering","documentation","html","k8s,containers",
         "Kubernetes docs.","advanced",4.6,"CNCF",2025,200,1),
        ("Docker","https://docs.docker.com/","Engineering","documentation","html","docker,containers",
         "Docker docs.","intermediate",4.6,"Docker",2025,120,1),
        ("Helm","https://helm.sh/docs/","Engineering","documentation","html","helm,k8s",
         "Helm — pakiety K8s.","intermediate",4.5,"CNCF",2025,80,1),
        ("GitHub Actions","https://docs.github.com/actions","Engineering","documentation","html","ci/cd,actions",
         "CI/CD w GitHub Actions.","intermediate",4.5,"GitHub",2025,80,1),
        ("Jenkins","https://www.jenkins.io/doc/","Engineering","documentation","html","jenkins,ci",
         "Jenkins docs.","intermediate",4.4,"Jenkins",2025,80,1),

        # Cloud ML
        ("AWS SageMaker","https://docs.aws.amazon.com/sagemaker/","MLOps","documentation","html","aws,cloud,mlops",
         "SageMaker docs.","advanced",4.5,"AWS",2025,200,1),
        ("GCP Vertex AI","https://cloud.google.com/vertex-ai/docs","MLOps","documentation","html","gcp,cloud,mlops",
         "Vertex AI docs.","advanced",4.5,"Google",2025,200,1),
        ("Azure ML","https://learn.microsoft.com/azure/machine-learning/","MLOps","documentation","html","azure,cloud,mlops",
         "Azure ML docs.","advanced",4.5,"Microsoft",2025,200,1),

        # Data Viz / BI
        ("Apache Superset","https://superset.apache.org/docs/intro","DataViz","documentation","html","bi,dashboards",
         "Superset docs.","intermediate",4.4,"ASF",2025,60,1),
        ("Metabase","https://www.metabase.com/docs/latest/","DataViz","documentation","html","bi,dashboards",
         "Metabase docs.","intermediate",4.4,"Metabase",2025,60,1),

        # Research hubs
        ("Papers With Code","https://paperswithcode.com/","Research","hub","html","sota,papers,code",
         "Publikacje z kodem.","advanced",4.8,"PWC",2025,200,1),
        ("Google AI Blog","https://ai.googleblog.com/","Research","blog","html","ai,blog",
         "Google AI Blog.","intermediate",4.4,"Google",2025,60,1),
        ("Meta AI Blog","https://ai.meta.com/blog/","Research","blog","html","ai,blog",
         "Meta AI Blog.","intermediate",4.4,"Meta",2025,60,1),
    ]
    rows += [R(*x) for x in core_docs]

    # --- Algorytmy (sklearn) ---
    algos = [
        ("Logistic Regression","https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression"),
        ("SVM","https://scikit-learn.org/stable/modules/svm.html"),
        ("Random Forest","https://scikit-learn.org/stable/modules/ensemble.html#random-forests"),
        ("Gradient Boosting","https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting"),
        ("KNN","https://scikit-learn.org/stable/modules/neighbors.html"),
        ("Naive Bayes","https://scikit-learn.org/stable/modules/naive_bayes.html"),
        ("KMeans","https://scikit-learn.org/stable/modules/clustering.html#k-means"),
        ("PCA","https://scikit-learn.org/stable/modules/decomposition.html#pca"),
        ("Calibration","https://scikit-learn.org/stable/modules/calibration.html"),
        ("TimeSeriesSplit","https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split"),
    ]
    for name, url in algos:
        rows.append(R(f"{name} Guide", url, "Classic ML","guide","html", f"{name.lower()},tutorial",
                      "Samouczek." if lang=="pl" else "Tutorial.","beginner",4.5,"scikit-learn",2025,45,1))

    # --- Case studies / tutoriale (więcej pozycji) ---
    cases = [
        ("Fraud Detection (imbalanced)","https://www.kaggle.com/c/ieee-fraud-detection"),
        ("House Prices (regression)","https://www.kaggle.com/c/house-prices-advanced-regression-techniques"),
        ("Titanic (classification)","https://www.kaggle.com/c/titanic"),
        ("DB Sentiment","https://ai.stanford.edu/~amaas/data/sentiment/"),
        ("ImageNet","https://www.image-net.org/"),
        ("COCO Object Detection","https://cocodataset.org/"),
        ("M5 Forecasting","https://www.kaggle.com/c/m5-forecasting-accuracy"),
        ("RAG LangChain QA","https://python.langchain.com/docs/use_cases/question_answering/"),
        ("RAG LlamaIndex","https://docs.llamaindex.ai/en/stable/understanding/"),
        ("BentoML Deploy","https://docs.bentoml.com/en/latest/quickstart.html"),
        ("Triton Model Repository","https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md"),
        ("Evidently Drift","https://docs.evidentlyai.com/"),
        ("Kaggle Notebooks","https://www.kaggle.com/code"),
        ("HF Spaces","https://huggingface.co/spaces"),
        ("Gradio Guides","https://www.gradio.app/guides"),
        ("Customer Churn (classification)","https://www.kaggle.com/datasets/blastchar/telco-customer-churn"),
        ("Retail Recommenders (implicit MF)","https://implicit.readthedocs.io/en/latest/"),
        ("Time Series Forecasting (Prophet)","https://facebook.github.io/prophet/docs/quick_start.html"),
        ("LGBM on Tabular","https://lightgbm.readthedocs.io/en/latest/"),
        ("XGBoost Best Practices","https://xgboost.readthedocs.io/en/stable/"),
        ("SHAP Explanations","https://shap.readthedocs.io/en/latest/"),
        ("Optuna Hyperparameter Tuning","https://optuna.org/"),
        ("Airflow Tutorial","https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html"),
        ("Prefect Flows","https://docs.prefect.io/"),
        ("Dagster Tutorial","https://docs.dagster.io/getting-started"),
    ]
    for t,u in cases:
        rows.append(R(t,u,"Case Study","guide","html","case,tutorial",
                      "Studium przypadku." if lang=="pl" else "Case study.","intermediate",4.5,"Various",2025,90,1))

    # --- Uzupełnienie do ~150 ---
    while len(rows) < 150:
        n=len(rows)+1
        rows.append(R(f"ML Resource #{n}","https://paperswithcode.com/","Research","hub","html",
                      "sota,hub","Agregator SOTA.","intermediate",4.4,"PWC",2025,30,1))
    return rows

    # ====================== GLOSSARY (PL/EN) ======================

OFFLINE_GLOSS_EN = {
    "bias-variance": "Trade-off between under/overfitting; regularization and ensembles help.",
    "overfitting": "Model fits noise in training data; poor generalization.",
    "cross-validation": "K-fold/Stratified CV; fit preprocessors inside folds to avoid leakage.",
    "regularization": "L1/L2/ElasticNet; in DL: dropout, weight decay, early stopping.",
    "feature-engineering": "Scaling, encoding, interactions, leakage checks, target encoding within CV.",
    "metrics": "Cls: F1/PR-AUC; Reg: MAE/RMSE; Ranking: NDCG/MRR; Calibration: Brier/ECE.",
    "confusion-matrix": "TP/FP/FN/TN table; basis for precision/recall.",
    "roc-pr": "ROC-AUC vs PR-AUC; PR-AUC better under heavy class imbalance.",
    "hyperparameters": "Model settings tuned via search (grid/random/BO/Optuna).",
    "ensemble-learning": "Bagging/Boosting/Stacking for robustness and accuracy.",
    "transformer": "Self-attention, multi-head attention, residuals, positional encodings.",
    "tokenization": "BPE/WordPiece split words into subwords for better generalization.",
    "embedding": "Dense vector representation of text/images for similarity/retrieval.",
    "rag": "Hybrid retrieval (BM25+vectors), chunking, reranking, citations, guardrails.",
    "vector-dbs": "FAISS, Qdrant, Weaviate, Milvus; HNSW for ANN.",
    "prompt-engineering": "Instruction, constraints, examples; control length/cost.",
    "prompt-injection": "Adversarial text that tries to override system rules; mitigate with filters.",
    "data-drift": "Input distribution shift; requires monitoring and retraining.",
    "concept-drift": "Target relationship changes; adjust model or features.",
    "model-registry": "Track versions/stages (Staging/Prod), audit, rollback.",
    "serving": "Deploy via FastAPI/Triton/TFServing; latency budgets, autoscaling.",
    "eval-retrieval": "Recall@k, MRR, nDCG@k; click@k; average clicked rank."
}

OFFLINE_GLOSS_PL = {
    "bias-variance": "Balans niedopasowanie/przeuczenie; pomaga regularyzacja i zespoły modeli.",
    "overfitting": "Model uczy się szumu w treningu; słaba generalizacja.",
    "cross-validation": "K-fold/Stratified; preprocessy wewnątrz foldów (bez przecieków).",
    "regularization": "L1/L2/ElasticNet; w DL: dropout, weight decay, early stopping.",
    "feature-engineering": "Skalowanie, enkodowanie, interakcje, kontrola leakage, target encoding w CV.",
    "metrics": "Klasyfikacja: F1/PR-AUC; Regresja: MAE/RMSE; Ranking: NDCG/MRR; Kalibracja: Brier/ECE.",
    "confusion-matrix": "Tabela TP/FP/FN/TN; baza precision/recall.",
    "roc-pr": "ROC-AUC vs PR-AUC; PR-AUC lepsze przy dużej nierównowadze klas.",
    "hyperparameters": "Parametry strojenia (grid/random/BO/Optuna).",
    "ensemble-learning": "Bagging/Boosting/Stacking dla odporności i jakości.",
    "transformer": "Self-attention, multi-head, rezidua, kodowanie pozycji.",
    "tokenization": "BPE/WordPiece dzieli słowa na subwordy dla lepszej generalizacji.",
    "embedding": "Gęste wektory tekstu/obrazów do podobieństwa/retrievalu.",
    "rag": "Hybryda (BM25+wektory), chunking, reranking, cytaty, guardrails.",
    "vector-dbs": "FAISS, Qdrant, Weaviate, Milvus; HNSW dla ANN.",
    "prompt-engineering": "Instrukcja, ograniczenia, przykłady; kontrola długości/kosztu.",
    "prompt-injection": "Tekst adwersarialny próbujący nadpisać zasady; filtry/whitelist.",
    "data-drift": "Zmiana rozkładu wejść; monitoruj i retrenuj.",
    "concept-drift": "Zmiana relacji z targetem; dostosuj model/cechy.",
    "model-registry": "Wersje/stage (Staging/Prod), audyt, rollback.",
    "serving": "Deploy przez FastAPI/Triton/TFServing; budżety opóźnień, autoskaling.",
    "eval-retrieval": "Recall@k, MRR, nDCG@k; click@k; średnia pozycja kliknięcia."
}


def build_snippets()->Dict[str, Dict[str,Any]]:
    S={}
    def add(name, code, category="Misc", lang="python"):
        S[name]={"code":textwrap.dedent(code).strip(), "category":category, "lang":lang}

    # --- Classic ML ---
    add("sklearn_pipeline", """
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        import pandas as pd
        df=pd.read_csv('data.csv'); num=['age','income']; cat=['city','segment']
        X=df[num+cat]; y=df['target']
        pre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
        pipe=Pipeline([('pre',pre),('clf',LogisticRegression(max_iter=3000))])
        print('CV F1:', cross_val_score(pipe,X,y,cv=5,scoring='f1').mean())
    """, "Classic ML")
    add("xgboost_auc", """
        import xgboost as xgb, pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        df=pd.read_csv('data.csv'); X=df.drop('target',axis=1); y=df['target']
        Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
        m=xgb.XGBClassifier(n_estimators=800,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,tree_method='hist',eval_metric='auc')
        m.fit(Xtr,Ytr,eval_set=[(Xte,Yte)],verbose=50,early_stopping_rounds=50)
        print('AUC:', roc_auc_score(Yte, m.predict_proba(Xte)[:,1]))
    ""","Classic ML")
    add("lightgbm_mae", """
        import lightgbm as lgb, pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error
        df=pd.read_csv('data.csv'); X=df.drop('target',axis=1); y=df['target']
        Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=0.2,random_state=42)
        m=lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=64)
        m.fit(Xtr,Ytr, eval_set=[(Xte,Yte)], callbacks=[lgb.early_stopping(100)])
        print('MAE:', mean_absolute_error(Yte, m.predict(Xte)))
    ""","Classic ML")
    add("catboost_categoricals", """
        from catboost import CatBoostClassifier, Pool
        import pandas as pd
        df=pd.read_csv('data.csv'); cat=['city','segment']
        X=df.drop('target',axis=1); y=df['target']
        pool=Pool(X, y, cat_features=[X.columns.get_loc(c) for c in cat])
        m=CatBoostClassifier(iterations=1200, depth=6, learning_rate=0.05, eval_metric='AUC', verbose=100)
        m.fit(pool)
    ""","Classic ML")
    add("calibration_curve", """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        import pandas as pd
        df=pd.read_csv('data.csv'); X=df.drop('target',axis=1); y=df['target']
        Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)
        base=LogisticRegression(max_iter=2000).fit(Xtr,Ytr)
        cal=CalibratedClassifierCV(base, method='isotonic').fit(Xtr,Ytr)
        print('Ready: cal.predict_proba(Xte)')
    ""","Classic ML")
    add("optuna_xgb", """
        import optuna, xgboost as xgb, pandas as pd
        from sklearn.model_selection import cross_val_score
        df=pd.read_csv('data.csv'); X=df.drop('target',axis=1); y=df['target']
        def obj(trial):
            params=dict(n_estimators=trial.suggest_int('n',300,1200),max_depth=trial.suggest_int('d',3,10),
                        learning_rate=trial.suggest_float('lr',1e-3,0.3,log=True),
                        subsample=trial.suggest_float('sub',0.6,1.0), colsample_bytree=trial.suggest_float('col',0.6,1.0),
                        tree_method='hist', eval_metric='auc')
            return cross_val_score(xgb.XGBClassifier(**params), X, y, cv=5, scoring='roc_auc').mean()
        study=optuna.create_study(direction='maximize'); study.optimize(obj, n_trials=50); print(study.best_params)
    ""","Classic ML")

    # --- Deep Learning / PyTorch ---
    add("torch_mlp","""
        import torch, torch.nn as nn, torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        X=torch.randn(4096,30); y=(X.sum(dim=1)>0).long()
        dl=DataLoader(TensorDataset(X,y),batch_size=64,shuffle=True)
        class MLP(nn.Module):
            def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(30,128),nn.ReLU(),nn.Dropout(0.1),nn.Linear(128,2))
            def forward(self,x): return self.net(x)
        m=MLP(); opt=optim.AdamW(m.parameters(),1e-3); crit=nn.CrossEntropyLoss()
        for e in range(10):
            for xb,yb in dl: opt.zero_grad(); loss=crit(m(xb), yb); loss.backward(); opt.step()
    ""","Deep Learning")
    add("torch_amp_training","""
        scaler=torch.cuda.amp.GradScaler()
        for xb,yb in dl:
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss=crit(m(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    ""","Deep Learning")
    add("torchvision_augs","""
        import torchvision.transforms as T
        train_tfms=T.Compose([T.RandomResizedCrop(224),T.RandomHorizontalFlip(),T.ColorJitter(0.2,0.2,0.2),T.ToTensor()])
    ""","Deep Learning")

    # --- LLMs / RAG ---
    add("hf_sentiment","""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        m='distilbert-base-uncased-finetuned-sst-2-english'
        tok=AutoTokenizer.from_pretrained(m); mdl=AutoModelForSequenceClassification.from_pretrained(m)
        x=tok('This model is awesome!', return_tensors='pt')
        with torch.no_grad(): p=torch.softmax(mdl(**x).logits, dim=-1).squeeze()
        print('neg,pos:', p.tolist())
    ""","LLMs")
    add("hf_generation","""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m='gpt2'; tok=AutoTokenizer.from_pretrained(m); mdl=AutoModelForCausalLM.from_pretrained(m)
        x=tok('Once upon a time', return_tensors='pt'); out=mdl.generate(**x, max_length=60, do_sample=True, top_p=0.95)
        print(tok.decode(out[0], skip_special_tokens=True))
    ""","LLMs")
    add("peft_lora","""
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base='gpt2'; tok=AutoTokenizer.from_pretrained(base); mdl=AutoModelForCausalLM.from_pretrained(base)
        cfg=LoraConfig(r=8, lora_alpha=16, target_modules=['c_attn'], lora_dropout=0.05)
        mdl=get_peft_model(mdl, cfg); mdl.print_trainable_parameters()
    ""","LLMs")
    add("faiss_topk","""
        import faiss, numpy as np
        d=384; idx=faiss.IndexFlatIP(d)
        X=np.random.randn(1000,d).astype('float32'); X/=np.linalg.norm(X,axis=1,keepdims=True)
        idx.add(X); q=np.random.randn(1,d).astype('float32'); q/=np.linalg.norm(q,axis=1,keepdims=True)
        sims, ids = idx.search(q, 5); print(ids[0])
    ""","LLMs")
    add("langchain_basic_rag","""
        from langchain_community.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        texts=['doc about cats','doc about dogs']; splitter=RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
        docs=sum([splitter.split_text(t) for t in texts], [])
        emb=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vs=FAISS.from_texts(docs, emb); print(vs.similarity_search('dogs', k=2))
    ""","LLMs")

    # --- MLOps / Serving / Testing ---
    add("fastapi_stub","""
        from fastapi import FastAPI
        import uvicorn
        app=FastAPI()
        @app.get('/health')
        def health(): return {'ok': True}
        if __name__=='__main__': uvicorn.run(app, host='0.0.0.0', port=8000)
    ""","MLOps")
    add("mlflow_min","""
        import mlflow, random
        mlflow.set_experiment('exp')
        with mlflow.start_run():
            mlflow.log_metric('accuracy', random.random())
            mlflow.log_param('model','LogReg')
    ""","MLOps")
    add("pytest_contract","""
        # test_data_contracts.py
        import pandas as pd
        def test_no_nulls():
            df=pd.read_csv('data.csv')
            assert df.isnull().mean().max()<0.05
    ""","MLOps")

    # --- Data / Spark / Polars ---
    add("polars_groupby","""
        import polars as pl
        df=pl.read_csv('data.csv')
        out=(df.groupby('category').agg([pl.col('value').mean().alias('mean'), pl.col('value').std().alias('std')]))
        print(out)
    ""","Data")
    add("spark_quick","""
        from pyspark.sql import SparkSession
        spark=SparkSession.builder.appName('quick').getOrCreate()
        df=spark.read.csv('data.csv', header=True, inferSchema=True)
        df.groupBy('category').count().show()
    ""","Data")

    # --- SQL ---
    add("sql_window_rank","""
        SELECT region, seller, revenue,
               RANK() OVER (PARTITION BY region ORDER BY revenue DESC) AS rnk
        FROM sales;
    ""","SQL","sql")
    add("sql_upsert_postgres","""
        INSERT INTO users(id, name, email) VALUES ($1,$2,$3)
        ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name, email=EXCLUDED.email;
    ""","SQL","sql")
    return S

SNIPPETS = build_snippets()

def build_interview(lang: str) -> dict:
    # helper do tłumaczeń
    L = (lambda pl, en: pl) if lang == "pl" else (lambda pl, en: en)

    # Każda kategoria ma listę rekordów: {"level": "...", "q": "...", "a": "...", "tags":[...]}
    data = {
        "Classic ML": [
            {"level":"Junior",
             "q": L("Wyjaśnij różnicę Precision/Recall i kiedy użyć F1.",
                    "Explain Precision vs Recall and when to use F1."),
             "a": L("Precision=TP/(TP+FP), Recall=TP/(TP+FN). F1 przy niezrównoważeniu klas i gdy ważny balans błędów.",
                    "Precision=TP/(TP+FP), Recall=TP/(TP+FN). Use F1 under class imbalance or when both errors matter."),
             "tags":["metrics","imbalance"]},
            {"level":"Junior",
             "q": L("Czym jest overfitting i jak mu przeciwdziałać?",
                    "What is overfitting and how to mitigate it?"),
             "a": L("Model uczy się szumu. Rozwiązania: regularyzacja, prostszy model, więcej danych, data augmentation, CV, early stopping.",
                    "Model fits noise. Fix: regularization, simpler model, more data, augmentation, CV, early stopping."),
             "tags":["generalization"]},
            {"level":"Mid",
             "q": L("Jak zaprojektujesz walidację dla szeregów czasowych?",
                    "How do you design validation for time series?"),
             "a": L("Forward chaining/TimeSeriesSplit; żadnego mieszania przyszłości do przeszłości; okna; metryki na horyzonty.",
                    "Forward chaining/TimeSeriesSplit; no leakage from future; sliding windows; horizon-specific metrics."),
             "tags":["validation","time-series"]},
            {"level":"Mid",
             "q": L("Target encoding: ryzyka i best practices?",
                    "Target encoding: risks and best practices?"),
             "a": L("Ryzyko leakage. Rób encoding w CV (na foldach), z regularizacją (smoothing, noise), bez użycia całego targetu.",
                    "Leakage risk. Do it inside CV folds with regularization (smoothing, noise), never on full target."),
             "tags":["encoding","leakage"]},
            {"level":"Senior",
             "q": L("Jak zbudujesz stabilny pipeline dla dużego tabularu (produkcyjnie)?",
                    "Design a robust tabular ML pipeline for production."),
             "a": L("Kontrakty danych, testy (schematy/zakresy), feature store, versioning, MLflow registry, monitoring driftu/latencji, retrain schedule.",
                    "Data contracts, tests (schema/ranges), feature store, versioning, MLflow registry, drift/latency monitoring, retrain schedule."),
             "tags":["mlops","production"]},
        ],

        "Deep Learning": [
            {"level":"Junior",
             "q": L("Po co BatchNorm i dropout?",
                    "Why BatchNorm and dropout?"),
             "a": L("BN stabilizuje i przyspiesza trening; dropout – regularizacja przez losowe wyzerowanie neuronów.",
                    "BN stabilizes/speeds training; dropout is regularization via random neuron deactivation."),
             "tags":["regularization","training"]},
            {"level":"Mid",
             "q": L("Jak dobrać harmonogram uczenia (LR schedule)?",
                    "How to choose a LR schedule?"),
             "a": L("One-cycle/cosine, warmup. Monitoruj val loss i stabilność. W DL częsty wybór to cosine annealing z warmup.",
                    "One-cycle/cosine with warmup; monitor val loss and stability; cosine annealing with warmup is common."),
             "tags":["optimization"]},
            {"level":"Senior",
             "q": L("Jak wyskalować trening na 10M próbek obrazów?",
                    "How to scale training to 10M image samples?"),
             "a": L("Sharding danych, distributed data parallel, gradient accumulation, mixed precision, checkpointing, profilowanie I/O, solidny dataloader.",
                    "Data sharding, DDP, grad accumulation, mixed precision, checkpointing, I/O profiling, strong dataloaders."),
             "tags":["scaling","distributed"]},
        ],

        "LLM & NLP": [
            {"level":"Junior",
             "q": L("Na czym polega tokenizacja BPE?",
                    "What is BPE tokenization?"),
             "a": L("Częste pary znaków łączone w subwordy – kompromis między znakami a słowami, lepsza generalizacja.",
                    "Frequent character pair merges into subwords; compromise between chars and words; better generalization."),
             "tags":["tokenization"]},
            {"level":"Mid",
             "q": L("Jak poprawić trafność RAG bez zmiany modelu LLM?",
                    "How to improve RAG retrieval quality without changing the LLM?"),
             "a": L("Lepszy chunking (nagłówki, semantyczny), hybryda BM25+wektory, normalizacja, reranking, deduplikacja, cytaty/grounding.",
                    "Better chunking (semantic/headings), BM25+vectors hybrid, normalization, reranking, dedup, citations/grounding."),
             "tags":["rag","retrieval"]},
            {"level":"Senior",
             "q": L("Kiedy PEFT/LoRA vs pełny finetuning?",
                    "When choose PEFT/LoRA vs full finetuning?"),
             "a": L("PEFT/LoRA przy ograniczonych zasobach/domenie/stylu; full FT gdy duża zmiana zdolności modelu i masz budżet.",
                    "PEFT/LoRA for low-cost domain/style adaptation; full FT when capacities must change and budget allows."),
             "tags":["peft","finetune"]},
        ],

        "MLOps": [
            {"level":"Junior",
             "q": L("Po co Model Registry?",
                    "Why a Model Registry?"),
             "a": L("Wersjonowanie, promowanie przez stage (Staging/Prod), audyt, rollback.",
                    "Versioning, stage promotion (Staging/Prod), audit, rollback."),
             "tags":["registry"]},
            {"level":"Mid",
             "q": L("A/B vs canary – kiedy co?",
                    "A/B vs canary – when to use which?"),
             "a": L("Canary: stopniowe wypuszczanie i rollback szybciej; A/B: równoległe grupy do testów hipotez.",
                    "Canary: gradual rollout with quick rollback; A/B: parallel groups for hypothesis testing."),
             "tags":["deployment","experimentation"]},
            {"level":"Senior",
             "q": L("Architektura serving@scale na K8s (LLM/ML).",
                    "Serving@scale on K8s (LLM/ML)."),
             "a": L("Autoscaling (HPA), batching, Triton/TFServing, trace (OTEL), metrics (Prom/Grafana), rate limits, cache, kosztomierz.",
                    "HPA autoscaling, batching, Triton/TFServing, tracing (OTEL), metrics (Prom/Grafana), rate limits, cache, cost meter."),
             "tags":["k8s","serving","observability"]},
        ],

        "System Design (ML/LLM)": [
            {"level":"Mid",
             "q": L("Zaprojektuj rekomendacje produktów (sklep online).",
                    "Design a product recommender for e-commerce."),
             "a": L("Funkcje: historię, kontekst, popularność; kaskada: candidate gen → ranker; feedback loop; AB testy; cold-start.",
                    "Features: history, context, popularity; cascade: candidate gen→ranker; feedback loop; AB tests; cold-start handling."),
             "tags":["recsys","ranking","abtesting"]},
            {"level":"Senior",
             "q": L("System Q&A z RAG + bezpieczeństwo.",
                    "Q&A with RAG + safety."),
             "a": L("Ingest (OCR/clean/chunk), indeks (BM25+ANN), reranking, cytaty; guardrails (toxicity/PII), rate-limit, audit log, eval.",
                    "Ingest (OCR/clean/chunk), index (BM25+ANN), reranking, citations; guardrails (toxicity/PII), rate-limit, audit log, eval."),
             "tags":["rag","safety","governance"]},
        ],

        "Statistics & Math": [
            {"level":"Junior",
             "q": L("Central Limit Theorem – intuicja?",
                    "Central Limit Theorem – intuition?"),
             "a": L("Średnia z próbek dąży do rozkładu normalnego (przy spełnieniu warunków), co pozwala na wnioski z prób.",
                    "Sample means approach normality (under conditions), enabling inference from samples."),
             "tags":["clt"]},
            {"level":"Mid",
             "q": L("Test A/B: jak dobrać wielkość próby?",
                    "A/B testing: how to choose sample size?"),
             "a": L("Moc testu, istotność, spodziewany efekt, wariancja; użyj wzorów/ narzędzi; planuj horyzont.",
                    "Power, significance, expected effect, variance; use formulas/tools; plan horizon."),
             "tags":["abtest","power"]},
        ],

        "LLM Safety & Ethics": [
            {"level":"Mid",
             "q": L("Jak bronić się przed prompt injection?",
                    "How to mitigate prompt injection?"),
             "a": L("Separacja instrukcji, filtrowanie tool calls, dopasowanie do schematów, whitelisting, ograniczenia przeglądarki/IO.",
                    "Separate system prompts, filter tool calls, schema validation, whitelists, restricted browsing/IO."),
             "tags":["security","prompt-injection"]},
            {"level":"Senior",
             "q": L("Red teaming LLM – jak podejść?",
                    "LLM red teaming – approach?"),
             "a": L("Katalog ataków, generatory adwersarialne, scenariusze domenowe, scoreboard metryk, pętle hardeningu.",
                    "Attack catalog, adversarial generators, domain scenarios, metric scoreboard, hardening loops."),
             "tags":["redteam","eval"]},
        ],

        "Behavioral": [
            {"level":"All",
             "q": L("Opowiedz o projekcie z porażką i czego się nauczyłeś/aś.",
                    "Tell me about a failed project and what you learned."),
             "a": L("Użyj STAR: Sytuacja→Zadanie→Działania→Rezultat; pokaż odpowiedzialność, lekcje, co zmieniłeś/aś.",
                    "Use STAR: Situation→Task→Actions→Result; show ownership, lessons, what you changed."),
             "tags":["star","ownership"]},
            {"level":"All",
             "q": L("Jak priorytetyzujesz zadania pod presją czasu?",
                    "How do you prioritize under time pressure?"),
             "a": L("Matryca pilności/ważności, rozbicie ryzyk, wczesna walidacja hipotez, komunikacja z interesariuszami.",
                    "Urgent/important matrix, risk breakdown, early hypothesis validation, stakeholder comms."),
             "tags":["prioritization"]},
        ],
    }

    # Dodatkowe kategorie: Experimentation, Data Eng for ML
    data["Experimentation"] = [
        {"level":"Mid",
         "q": L("Jak unikać p-hacking w eksperymentach?",
                "How to avoid p-hacking in experiments?"),
         "a": L("Pre-registration, czyste KPI, korekcja na wielokrotność, monitorowanie mocy testu.",
                "Pre-registration, clean KPIs, multiple testing correction, monitoring test power."),
         "tags":["abtest","ethics"]},
        {"level":"Senior",
         "q": L("Jak zaprojektujesz eksperyment online dla nowej funkcji modelu?",
                "Design an online experiment for a new model feature."),
         "a": L("Definiuj hipotezę, KPI, jednostkę randomizacji, segmentację; ramię kontrolne vs treatment; sanity checks.",
                "Define hypothesis, KPIs, unit of randomization, segmentation; control vs treatment arms; sanity checks."),
         "tags":["online-exp"]},
    ]

    data["Data Engineering for ML"] = [
        {"level":"Mid",
         "q": L("Jak egzekwować kontrakty danych?",
                "How to enforce data contracts?"),
         "a": L("Schematy (Great Expectations), zakresy i reguły; testy w CI; alarmy; wersjonowanie.",
                "Schemas (Great Expectations), ranges/rules; CI tests; alerts; versioning."),
         "tags":["data-quality","contracts"]},
        {"level":"Senior",
         "q": L("Skalowalny ingestion logów zdarzeń – jak?",
                "Scalable event log ingestion – how?"),
         "a": L("Kafka→obj storage (parquet/iceberg), partycjonowanie po czasie/kluczach, kompaktowanie, CDC.",
                "Kafka→object storage (parquet/iceberg), partition by time/keys, compaction, CDC."),
         "tags":["kafka","lake"]},
    ]

    return data


# ====================== HELPERS ======================

def fts_where_clause(q: str) -> tuple[str, list]:
    """
    Buduje WHERE oparte o FTS5 (MATCH) dla szybkiego pełnotekstowego searcha.
    Zwraca (where_sql, params).
    Uwaga: działa gdy masz utworzoną tabelę resources_fts (ensure_schema).
    """
    q = (q or "").strip()
    if len(q) < 3:
        # Za krótkie – lepiej nie używać MATCH (brak wyników/mała trafność).
        return "1=1", []
    # Prosty wariant; można rozbudować o NEAR, OR, pola, itp.
    return "id IN (SELECT rowid FROM resources_fts WHERE resources_fts MATCH ?)", [q]

def log_activity(conn, resource_id: int, activity_type: str, meta: dict | None = None) -> None:
    """
    Lekka telemetria kliknięć/akcji. Nie zrywa działania przy błędzie.
    """
    try:
        conn.execute(
            "INSERT INTO user_activity(resource_id, activity_type, meta) VALUES (?, ?, ?)",
            (int(resource_id), str(activity_type), json.dumps(meta or {}))
        )
        conn.commit()
    except Exception:
        pass

def estimate_tokens(text: str) -> int:
    """
    Bardzo przybliżony licznik tokenów (~4 znaki = 1 token).
    Przydaje się do szacowania kosztu promptu.
    """
    t = max(1, int(len(text or "") / 4))
    return t

def token_split(s: str) -> list[str]:
    """
    Proste tokenizowanie do „offline” dopasowań (fallback, bez FTS).
    """
    import re
    return [t for t in re.split(r'[^a-z0-9+#\-]+', (s or "").lower()) if t]

def simple_match_score(query_tokens: list[str], row) -> int:
    """
    Fallback scoring dla offline retrieval: zlicza wystąpienia tokenów w polach.
    """
    hay = " ".join([
        str(row.get('title','')),
        str(row.get('description','')),
        str(row.get('category','')),
        str(row.get('tags',''))
    ]).lower()
    return sum(1 for t in query_tokens if t in hay)

# ======= Interview helpers =======
FILLERS_PL = ["yyy", "eee", "hmm", "no więc", "tak naprawdę", "w sensie"]
FILLERS_EN = ["um", "uh", "like", "you know", "so", "actually", "basically"]

KEYWORDS_DEFAULT = {
    "Classic ML": ["cross-validation","regularization","leakage","metrics","feature engineering","class imbalance","AUC","F1"],
    "Deep Learning": ["batch norm","dropout","learning rate","scheduler","augmentation","overfitting","precision","recall"],
    "LLM & NLP": ["tokenization","BPE","transformer","attention","RAG","embedding","LoRA","prompt"],
    "MLOps": ["mlflow","registry","monitoring","drift","canary","A/B","CI/CD","serving","Kubernetes"],
    "System Design (ML/LLM)": ["latency","throughput","caching","ranker","candidate generation","BM25","ANN","guardrails"],
}

def _pick_lang_list(lang:str)->list[str]:
    return FILLERS_PL if lang=="pl" else FILLERS_EN

def _count_fillers(text:str, lang:str)->dict:
    text_l = (text or "").lower()
    return {w: text_l.count(w) for w in _pick_lang_list(lang)}

def _keyword_hits(text:str, keywords:list[str])->tuple[int,list[str]]:
    text_l = (text or "").lower()
    hits = [k for k in keywords if k.lower() in text_l]
    return (len(hits), hits)

def _star_coverage(text:str, lang:str)->dict:
    t = (text or "").lower()
    if lang=="pl":
        keys = {"S":"sytuac", "T":"zadani", "A":"działa", "R":"rezult"}
    else:
        keys = {"S":"situat", "T":"task", "A":"action", "R":"result"}
    return {k: (keys[k] in t) for k in keys}

def _speaking_rate(text:str, seconds:int|None)->float|None:
    if not text or not seconds or seconds<=0: return None
    words = len([x for x in text.split() if x.strip()])
    return round(60.0*words/max(1,seconds), 1)

def _stt_whisper(audio_bytes:bytes, client)->str:
    # Proste STT (opcjonalne) — działa jeśli masz klucz OpenAI ustawiony
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes); tmp=f.name
        t = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=open(tmp, "rb")
        )
        return getattr(t, "text", "") or ""
    except Exception as e:
        return f"(Transkrypcja nieudana: {e})"

# ====================== Mentor (RAG) ======================
def token_split(s:str)->List[str]:
    return [t for t in re.split(r'[^a-z0-9+#\-]+', s.lower()) if t]
def simple_match_score(tokens:List[str], row:pd.Series)->int:
    hay = " ".join([str(row.get('title','')), str(row.get('description','')), str(row.get('category','')), str(row.get('tags',''))]).lower()
    return sum(1 for t in tokens if t in hay)
def fts_where_clause(q: str) -> Tuple[str, list]:
    """Buduje WHERE oparte o FTS5 (match)."""
    q = q.strip()
    if len(q) < 3:
        return "1=1", []
    # użyj prostego matcha; można rozszerzyć o operatory NEAR, OR itd.
    return "id IN (SELECT rowid FROM resources_fts WHERE resources_fts MATCH ?)", [q]
def log_activity(conn, resource_id: int, activity_type: str, meta: Optional[dict] = None):
    try:
        conn.execute(
            "INSERT INTO user_activity(resource_id, activity_type, meta) VALUES (?, ?, ?)",
            (resource_id, activity_type, json.dumps(meta or {}))
        )
        conn.commit()
    except Exception:
        pass

def get_openai_client()->Optional['OpenAI']:
    key = st.session_state.get('openai_key') or os.getenv('OPENAI_API_KEY')
    if not key or not OPENAI_AVAILABLE: return None
    try: return OpenAI(api_key=key)
    except Exception: return None

MENTOR_SYSTEM_PROMPT_PL = (
"Jesteś „Mistrz ML/DL” — doświadczony, konkretny mentor.\n"
"- Odpowiadaj jasno, krok po kroku, krótkie akapity i listy.\n"
"- Cytuj 2–6 źródeł z kontekstu, gdy dostępne.\n"
"- Pokaż minimalny, działający kod, jeśli to pomaga.\n"
"- Zakończ checklistą wdrożenia, gdy adekwatne.\n"
)

def retrieve_context(query:str, topk:int=8)->pd.DataFrame:
    conn=connect_db(); df=pd.read_sql("SELECT * FROM resources", conn); conn.close()
    if df.empty: return df
    toks=token_split(query)
    df['score']=df.apply(lambda r: simple_match_score(toks,r),axis=1)
    return df.sort_values(['score','rating','views'],ascending=[False,False,False]).head(topk).copy()

def build_context_text(df:pd.DataFrame)->str:
    lines=[]
    for _,r in df.iterrows():
        lines.append(f"- {r['title']} ({r['category']} | {r['type']} | ⭐{float(r.get('rating',0)):.1f}) — {r['url']}\n  {r.get('description') or ''}")
    return "\n".join(lines)

def offline_answer(query:str, ctx_df:pd.DataFrame, lang:str)->str:
    gloss = OFFLINE_GLOSS_PL if lang=="pl" else OFFLINE_GLOSS_EN
    toks=token_split(query); notes=[]
    for k,v in gloss.items():
        if any(t in k or t in v.lower() for t in toks): notes.append(f"**{k}**: {v}")
    if not notes: notes.append("Brak dopasowania — patrz kontekst." if lang=="pl" else "No direct note—see context.")
    ctx = build_context_text(ctx_df) if not ctx_df.empty else "—"
    parts = [
        f"### {T('answer',lang)}",
        *[f"- {n}" for n in notes],
        "",
        "**Checklist:**",
        ("- Zdefiniuj cel/metryki\n- EDA → split (bez leakage)\n- Pipeline + walidacja\n- Tuning (Optuna)\n- Registry (MLflow), testy, kontrakty\n- Deploy (Bento/Seldon/FastAPI) + monitoring (Evidently)\n- Retrain & drift")
        if lang=="pl" else
        "- Define objective/metrics\n- EDA → split (no leakage)\n- Pipeline + validation\n- Tuning (Optuna)\n- Registry (MLflow), tests, contracts\n- Deploy (Bento/Seldon/FastAPI) + monitoring (Evidently)\n- Retrain & drift",
        "",
        f"**{T('used_sources',lang)}:**",
        ctx
    ]
    return "\n".join(parts)

# ====================== Views ======================
def view_dashboard(lang:str):
    st.header(T('dashboard',lang))
    c1,c2,c3=st.columns(3)
    c1.metric(T('completed',lang), len(st.session_state.completed_resources))
    c2.metric(T('study_hours',lang), st.session_state.study_hours)
    c3.metric(T('streak',lang), st.session_state.current_streak)
    conn=connect_db()
    df_cat=pd.read_sql('SELECT category, COUNT(*) count FROM resources GROUP BY category',conn)
    if not df_cat.empty:
        st.subheader(T('resources_by_cat',lang))
        if PLOTLY_AVAILABLE:
            fig=px.bar(df_cat,x='category',y='count'); st.plotly_chart(fig,use_container_width=True)
        else:
            st.bar_chart(df_cat.set_index('category'))
    top=pd.read_sql('SELECT title, category, rating FROM resources ORDER BY rating DESC LIMIT 20',conn)
    if not top.empty:
        st.subheader(T('top_rated',lang)); st.dataframe(top, use_container_width=True)
    act=pd.read_sql('''SELECT r.title, ua.activity_type, ua.timestamp
                       FROM user_activity ua JOIN resources r ON ua.resource_id=r.id
                       ORDER BY ua.timestamp DESC LIMIT 20''', conn)
    if not act.empty:
        st.subheader(T('recent_activity',lang)); st.dataframe(act, use_container_width=True)
    conn.close()

def view_library(lang: str):
    import math, json
    import streamlit.components.v1 as components

    st.header(T('library', lang))
    conn = connect_db()

    # --- Filtry z bazy ---
    categories = ['(All)'] + [row[0] for row in conn.execute(
        'SELECT DISTINCT category FROM resources ORDER BY category').fetchall()]
    types = ['(All)'] + [row[0] for row in conn.execute(
        'SELECT DISTINCT "type" FROM resources ORDER BY "type"').fetchall()]
    formats = ['(All)'] + [row[0] for row in conn.execute(
        'SELECT DISTINCT "format" FROM resources ORDER BY "format"').fetchall()]

    with st.expander(T('filters', lang), expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1: cat = st.selectbox(T('category', lang), categories)
        with c2: typ = st.selectbox(T('type', lang), types)
        with c3: fmt = st.selectbox(T('format', lang), formats)
        q = st.text_input(T('search', lang), "")
        tag_query = st.text_input(T('tags', lang), "")  
        
    layout = st.segmented_control(
    T('view', lang),
    [T('list', lang), T('table', lang), T('tiles', lang)],
    selection_mode="single",
    default=T('list', lang)
    )

    # --- WHERE i paramy (preferuj FTS jeśli q >= 3) ---
    where_parts = []
    params = [] 

    if q and len(q.strip()) >= 3:
        fts_sql, fts_params = fts_where_clause(q)
        where_parts.append(fts_sql)
        params.extend(fts_params)
    else:
        where_parts.append("1=1")
        if q:  # krótki tekst -> LIKE
            where_parts.append("(title LIKE ? OR description LIKE ?)")
            params.extend([f"%{q}%", f"%{q}%"])

    if cat != '(All)': where_parts.append("category = ?"); params.append(cat)
    if typ != '(All)': where_parts.append('"type" = ?'); params.append(typ)
    if fmt != '(All)': where_parts.append('"format" = ?'); params.append(fmt)
    if tag_query:
        for t in [t.strip().lower() for t in tag_query.split(",") if t.strip()]:
            where_parts.append("lower(tags) LIKE ?"); params.append(f"%{t}%")
    where_sql = " AND ".join(where_parts)

    total = conn.execute(f"SELECT COUNT(*) FROM resources WHERE {where_sql}", params).fetchone()[0]

    # --- Paginacja ---
    per_page = st.selectbox("Na stronę / Per page", [10, 20, 30, 50, 100], index=1)
    num_pages = max(1, math.ceil(total / per_page))
    pg_key = f"lib_page_{hash((cat, typ, fmt, q, tag_query, per_page))}"
    page = st.session_state.get(pg_key, 1)
    if page > num_pages: page = num_pages

    colp1, colp2, colp3 = st.columns([1,2,1])
    with colp1:
        if st.button("◀ Prev", disabled=(page<=1)):
            page = max(1, page-1)
    with colp2:
        st.markdown(f"**{T('found_n', lang).format(n=total)}** — strona {page}/{num_pages}")
    with colp3:
        if st.button("Next ▶", disabled=(page>=num_pages)):
            page = min(num_pages, page+1)
    st.session_state[pg_key] = page
    offset = (page-1)*per_page

    # --- Dane strony ---
    df = pd.read_sql(f"""
        SELECT * FROM resources
        WHERE {where_sql}
        ORDER BY rating DESC, views DESC, id ASC
        LIMIT ? OFFSET ?""", conn, params=tuple(params + [per_page, offset]))

    

    # --- Eksport / Import ---
    colA, colB = st.columns([1,1])
    with colA:
        full_df = pd.read_sql(
            f"""SELECT id,title,url,category,type,format,tags,difficulty,rating,author,year,length_min,free
                FROM resources WHERE {where_sql}
                ORDER BY rating DESC, views DESC""",
            conn, params=tuple(params)
        )
        st.download_button(T('download_btn', lang),
            data=full_df.to_csv(index=False),
            file_name="resources_export.csv",
            mime="text/csv"
        )
    with colB:
        st.write(T('import', lang))
        up = st.file_uploader("JSON", type=["json"])
        if up and st.button(T('import_btn', lang)):
            try:
                payload = json.load(up)
                rows=[]
                for r in payload:
                    rows.append((
                        r['title'], r['url'], r['category'],
                        r.get('type','article'), r.get('format','html'),
                        r.get('tags',''), r.get('description',''),
                        r.get('difficulty','intermediate'),
                        float(r.get('rating',0)), r.get('author',''),
                        r.get('year',None), r.get('length_min',None),
                        int(r.get('free',1))
                    ))
                add_resources_bulk(conn, rows)
                st.success("Zaimportowano zasoby." if lang=="pl" else "Resources imported.")
                st.rerun()
            except Exception as e:
                st.error(f"{'Błąd importu' if lang=='pl' else 'Import error'}: {e}")

    # --- Widoki ---
    if layout == T('table', lang):
        view_cols = ['title','category','type','format','rating','tags','year','author','views','url']
        st.dataframe(df[view_cols], use_container_width=True, hide_index=True)
    elif layout == T('tiles', lang):
        ncols = 3
        cols = st.columns(ncols)
        for i, (_, r) in enumerate(df.iterrows()):
            with cols[i % ncols]:
                st.markdown(f"**{r['title']}**  \n_{r['category']} · {r['type']} · ⭐ {float(r.get('rating',0)):.1f}_")
                st.caption(r.get('description') or '-')
                if st.button(T('open_resource', lang), key=f"open_card_{r['id']}"):
                    conn.execute("UPDATE resources SET views = views + 1 WHERE id=?", (int(r['id']),))
                    log_activity(conn, int(r['id']), "opened", {"via":"card"})
                    components.html(f"<script>window.open({json.dumps(str(r['url']))}, '_blank');</script>", height=0)
                if st.button("⭐ Zakładka" if lang=="pl" else "⭐ Bookmark", key=f"bm_card_{r['id']}"):
                    conn.execute("INSERT INTO bookmarks(resource_id) VALUES(?)", (int(r['id']),)); conn.commit()
                    st.toast("Dodano")
    else:
        # Lista (expandery) + zakładki/notatki
        for _, r in df.iterrows():
            title_line = f"{r['title']} — {r['category']} | {str(r['type']).upper()} | ⭐ {float(r.get('rating',0)):.1f}"
            with st.expander(title_line):
                c1, c2, c3 = st.columns([2,1,1])
                with c1:
                    st.write(f"**{T('desc',lang)}:** {r.get('description') or '-'}")
                    st.write(f"**Tagi/Tags:** {r.get('tags') or '-'}")
                    st.write(f"**{T('level',lang)}:** {r.get('difficulty') or '-'}")
                    st.write(f"**{T('author',lang)}:** {r.get('author') or '-'} | **{T('year',lang)}:** {r.get('year') or '-'} | **Format:** {r.get('format') or '-'}")
                    st.write(f"**URL:** {r['url']}")
                with c2:
                    st.metric(T('rating',lang), f"{float(r.get('rating') or 0.0):.1f}/5")
                    st.metric(T('views',lang), int(r.get('views') or 0))
                    st.metric(T('free',lang), "✅" if int(r.get('free',1))==1 else "❌")
                with c3:
                    rid = int(r['id'])
                    if st.link_button(T('open_resource', lang), r['url'], type="primary", use_container_width=True):
                        conn.execute("UPDATE resources SET views = views + 1 WHERE id = ?", (rid,))
                        log_activity(conn, rid, "opened", {"via": "list"})

                    done = rid in st.session_state.completed_resources
                    if st.button(T('mark_done', lang) if not done else T('marked_done', lang), key=f"done_{rid}"):
                        if not done:
                            st.session_state.completed_resources.append(rid)
                            st.session_state.study_hours += 1
                            log_activity(conn, rid, "completed")
                            st.success("✓"); st.rerun()

                    if st.button("⭐ Zakładka" if lang=="pl" else "⭐ Bookmark", key=f"bm_{rid}"):
                        conn.execute("INSERT INTO bookmarks(resource_id) VALUES (?)", (rid,))
                        conn.commit(); st.toast("Dodano")

                # Notatki
                with st.form(f"note_{rid}", clear_on_submit=True):
                    body = st.text_area("Notatka" if lang=="pl" else "Note", height=80)
                    if st.form_submit_button("Zapisz"):
                        if body.strip():
                            conn.execute("INSERT INTO notes(resource_id, body) VALUES (?,?)", (rid, body.strip()))
                            conn.commit(); st.success("Zapisano"); st.rerun()

                notes_df = pd.read_sql("SELECT body, created_at FROM notes WHERE resource_id=? ORDER BY created_at DESC", conn, params=(rid,))
                if not notes_df.empty:
                    st.caption("Notatki:"); st.dataframe(notes_df, use_container_width=True, hide_index=True)

    conn.close()

def view_collections(lang: str):
    st.header("🗂️ Kolekcje / Collections")
    conn = connect_db()

    # Tworzenie nowej kolekcji
    with st.form("new_collection", clear_on_submit=True):
        name = st.text_input("Nazwa kolekcji / Name")
        desc = st.text_area("Opis / Description", height=80)
        if st.form_submit_button("Utwórz / Create") and name.strip():
            conn.execute("INSERT INTO collections(name, description) VALUES(?,?)", (name.strip(), desc.strip()))
            conn.commit(); st.success("Utworzono"); st.rerun()

    # Lista kolekcji
    cols_df = pd.read_sql("SELECT id, name, description, created_at FROM collections ORDER BY created_at DESC", conn)
    if cols_df.empty:
        st.info("Brak kolekcji.")
        conn.close(); return

    for _, row in cols_df.iterrows():
        with st.expander(f"{row['name']} — {row['created_at']}"):
            st.write(row.get('description') or '')
            # Pozycje
            items = pd.read_sql("""
                SELECT ci.id as cid, ci.ord, r.id as rid, r.title, r.category, r.type, r.rating
                FROM collection_items ci JOIN resources r ON ci.resource_id=r.id
                WHERE ci.collection_id=? ORDER BY ci.ord ASC, r.rating DESC
            """, conn, params=(int(row['id']),))
            if not items.empty:
                st.dataframe(items[['ord','title','category','type','rating']], use_container_width=True, hide_index=True)
            # Dodaj zasób po ID
            with st.form(f"add_item_{row['id']}", clear_on_submit=True):
                rid = st.number_input("ID zasobu / Resource ID", min_value=1, step=1)
                ordv = st.number_input("Kolejność / Order", value=0, step=1)
                if st.form_submit_button("Dodaj / Add"):
                    conn.execute("INSERT INTO collection_items(collection_id, resource_id, ord) VALUES(?,?,?)",
                                 (int(row['id']), int(rid), int(ordv)))
                    conn.commit(); st.success("Dodano"); st.rerun()
    conn.close()

def view_cases(lang: str):
    import streamlit as st

    st.header("🧪 " + T('cases', lang))
    st.caption(
        T('case_intro', lang)
        if 'case_intro' in TEXTS.get(lang, {})
        else ("Praktyczne studia przypadków i tutoriale krok po kroku."
              if lang == "pl" else
              "Practical case studies and step-by-step tutorials.")
    )

    # ✅ 27 pozycji, linki zweryfikowane na stabilne ścieżki / oficjalne strony
    cases = [
        # ─── Tabular / klasyki (Kaggle / UCI) ─────────────────────────────────────
        {"title":"Fraud Detection (Imbalanced Dataset)",
         "desc_pl":"Wykrywanie oszustw z danymi niezrównoważonymi.",
         "desc_en":"Fraud detection with imbalanced data.",
         "url":"https://www.kaggle.com/competitions/ieee-fraud-detection"},
        {"title":"House Prices (Regression)",
         "desc_pl":"Regresja cen domów – klasyczny problem ML.",
         "desc_en":"House prices regression – a classic ML problem.",
         "url":"https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques"},
        {"title":"Titanic (Classification)",
         "desc_pl":"Klasyfikacja przeżycia pasażerów.",
         "desc_en":"Passenger survival classification.",
         "url":"https://www.kaggle.com/competitions/titanic"},
        {"title":"Telco Customer Churn (IBM)",
         "desc_pl":"Prognozowanie odejść klientów (IBM Telco).",
         "desc_en":"Customer churn prediction (IBM Telco).",
         "url":"https://www.kaggle.com/datasets/blastchar/telco-customer-churn"},
        {"title":"Credit Default Risk",
         "desc_pl":"Ryzyko niespłacenia kredytu.",
         "desc_en":"Credit default risk prediction.",
         "url":"https://www.kaggle.com/competitions/home-credit-default-risk"},
        {"title":"Adult Income (UCI)",
         "desc_pl":"Klasyfikacja dochodu >50K (UCI Adult).",
         "desc_en":"Income >50K classification (UCI Adult).",
         "url":"https://archive.ics.uci.edu/dataset/2/adult"},
        {"title":"NYC Taxi Trip Duration",
         "desc_pl":"Regresja czasu przejazdu taxi (NYC).",
         "desc_en":"Taxi trip duration regression (NYC).",
         "url":"https://www.kaggle.com/competitions/nyc-taxi-trip-duration"},

        # ─── NLP / RAG ────────────────────────────────────────────────────────────
        {"title":"Scikit-learn: Working With Text Data",
         "desc_pl":"Klasyczny tutorial przetwarzania tekstu (20 newsgroups).",
         "desc_en":"Classic text analytics tutorial (20 newsgroups).",
         "url":"https://scikit-learn.org/1.4/tutorial/text_analytics/working_with_text_data.html"},
        {"title":"Sentiment Analysis (Transformers)",
         "desc_pl":"Sekcja zadań Transformers: klasyfikacja sekwencji.",
         "desc_en":"Transformers tasks: sequence classification.",
         "url":"https://huggingface.co/docs/transformers/tasks/sequence_classification"},
        {"title":"LangChain: Question Answering (RAG)",
         "desc_pl":"Oficjalny use-case QA/RAG w LangChain.",
         "desc_en":"Official QA/RAG use case in LangChain.",
         "url":"https://python.langchain.com/docs/use_cases/question_answering/"},
        {"title":"LlamaIndex: Get Started (RAG)",
         "desc_pl":"Wprowadzenie do LlamaIndex (indeksowanie i zapytania).",
         "desc_en":"LlamaIndex getting started (indexing and queries).",
         "url":"https://docs.llamaindex.ai/en/stable/get_started/"},
        {"title":"Text Embeddings + FAISS",
         "desc_pl":"Repo FAISS – szybkie wyszukiwanie podobieństwa.",
         "desc_en":"FAISS repo – fast similarity search.",
         "url":"https://github.com/facebookresearch/faiss"},

        # ─── Computer Vision ──────────────────────────────────────────────────────
        {"title":"PyTorch Transfer Learning",
         "desc_pl":"Transfer learning w klasyfikacji obrazów (oficjalny tutorial).",
         "desc_en":"Transfer learning for image classification (official tutorial).",
         "url":"https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"},
        {"title":"COCO – Object Detection",
         "desc_pl":"Zbiór COCO i zadania detekcji.",
         "desc_en":"COCO dataset and detection tasks.",
         "url":"https://cocodataset.org/#home"},
        {"title":"U-Net Image Segmentation (PyTorch impl.)",
         "desc_pl":"Praktyczna implementacja U-Net.",
         "desc_en":"Practical U-Net implementation.",
         "url":"https://github.com/milesial/Pytorch-UNet"},
        {"title":"Learn PyTorch: Transfer Learning (guide)",
         "desc_pl":"Przewodnik krok po kroku (nieoficjalny, solidny).",
         "desc_en":"Step-by-step guide (community, solid).",
         "url":"https://www.learnpytorch.io/06_pytorch_transfer_learning/"},

        # ─── Time Series ──────────────────────────────────────────────────────────
        {"title":"Prophet Quick Start",
         "desc_pl":"Oficjalny quick start dla Meta Prophet.",
         "desc_en":"Official quick start for Meta Prophet.",
         "url":"https://facebook.github.io/prophet/docs/quick_start.html"},
        {"title":"M5 Forecasting (Kaggle)",
         "desc_pl":"Prognozowanie sprzedaży w czasie (M5).",
         "desc_en":"Sales forecasting over time (M5).",
         "url":"https://www.kaggle.com/competitions/m5-forecasting-accuracy"},

        # ─── Wyjaśnialność / Tuning ──────────────────────────────────────────────
        {"title":"SHAP – Documentation",
         "desc_pl":"Wyjaśnialność modeli metodą SHAP.",
         "desc_en":"Model explainability with SHAP.",
         "url":"https://shap.readthedocs.io/en/latest/"},
        {"title":"Optuna – Getting Started",
         "desc_pl":"Strojenie hiperparametrów (Optuna).",
         "desc_en":"Hyperparameter optimization (Optuna).",
         "url":"https://optuna.org/"},

        # ─── MLOps / Serving / Orchestration ─────────────────────────────────────
        {"title":"MLflow Tracking – Quickstart",
         "desc_pl":"Szybki start z MLflow Tracking i rejestrem modeli.",
         "desc_en":"MLflow Tracking quickstart and model registry.",
         "url":"https://mlflow.org/docs/latest/ml/tracking/quickstart/"},
        {"title":"BentoML – Quickstart",
         "desc_pl":"Szybkie wdrażanie modeli z BentoML.",
         "desc_en":"Quick model deployment with BentoML.",
         "url":"https://docs.bentoml.com/en/latest/quickstart.html"},
        {"title":"NVIDIA Triton Inference Server",
         "desc_pl":"Serwowanie modeli na Triton.",
         "desc_en":"Serving models with Triton.",
         "url":"https://github.com/triton-inference-server/server"},
        {"title":"Airflow – Tutorials (stable)",
         "desc_pl":"Oficjalne tutoriale Apache Airflow.",
         "desc_en":"Official Apache Airflow tutorials.",
         "url":"https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html"},
        {"title":"Prefect – Quickstart (v3)",
         "desc_pl":"Szybki start z Prefect 3.",
         "desc_en":"Prefect 3 quickstart.",
         "url":"https://docs.prefect.io/v3/get-started/quickstart"},
        {"title":"Dagster – Getting Started",
         "desc_pl":"Szybki start z Dagster.",
         "desc_en":"Dagster getting started.",
         "url":"https://docs.dagster.io/getting-started/quickstart"},
        {"title":"Evidently – Docs",
         "desc_pl":"Monitorowanie jakości i dryfu danych.",
         "desc_en":"Monitoring data quality & drift.",
         "url":"https://docs.evidentlyai.com/"},
        {"title":"Gradio – Guides",
         "desc_pl":"Szybkie interfejsy do modeli.",
         "desc_en":"Quick UIs for models.",
         "url":"https://www.gradio.app/guides"},
        {"title":"Hugging Face Spaces – Docs",
         "desc_pl":"Hostowanie dem na HF Spaces.",
         "desc_en":"Host demos on HF Spaces.",
         "url":"https://huggingface.co/docs/hub/spaces"},
    ]

    # Render listy – expander + link_button (otwiera pewnie w nowej karcie)
    for case in cases:
        with st.expander(case["title"]):
            st.write(case["desc_pl"] if lang == "pl" else case["desc_en"])
            st.link_button(
                T('run_case', lang) if 'run_case' in TEXTS.get(lang, {}) else
                ("Otwórz w nowej karcie" if lang == "pl" else "Open in new tab"),
                case["url"]
            )


def _role_phases(role: str) -> list:
    """
    Definicja etapów nauki dla różnych ról w ML/AI.
    """
    phases = {
        "Data Scientist": [
            {"name": "Podstawy", "topics": ["Algebra", "Statystyka", "Python", "EDA", "Wizualizacje"], "weeks": 8},
            {"name": "Klasyczne ML", "topics": ["Modele", "CV", "Regularizacja", "Metryki"], "weeks": 10},
            {"name": "Wyjaśnialność", "topics": ["SHAP", "Permutation Importance"], "weeks": 3},
            {"name": "Wdrożenie", "topics": ["FastAPI", "Deploy", "Monitoring"], "weeks": 3},
        ],
        "ML Engineer": [
            {"name": "Podstawy kodowania", "topics": ["Python", "NumPy", "Pandas"], "weeks": 6},
            {"name": "Deep Learning", "topics": ["PyTorch", "CNN", "Optymalizacja"], "weeks": 8},
            {"name": "MLOps", "topics": ["MLflow", "CI/CD", "Monitoring"], "weeks": 6},
            {"name": "Skalowanie", "topics": ["Kubernetes", "Ray", "Serving"], "weeks": 4},
        ],
        "LLM Engineer": [
            {"name": "Transformery", "topics": ["Tokenizacja", "Self-Attention", "Embeddingi"], "weeks": 6},
            {"name": "Praca z LLM", "topics": ["Prompting", "Context Engineering"], "weeks": 8},
            {"name": "RAG", "topics": ["FAISS", "Qdrant", "LangChain"], "weeks": 6},
            {"name": "Fine-tuning", "topics": ["LoRA", "PEFT", "Bezpieczeństwo"], "weeks": 6},
        ],
    }
    return phases.get(role, [])


def _generate_plan(role: str, weeks: int, hours: int, lang: str) -> str:
    """
    Generuje szczegółowy plan nauki z kamieniami milowymi.
    """
    phases = _role_phases(role)
    if not phases:
        return "Brak danych dla wybranej roli." if lang == "pl" else "No data for this role."

    total_weeks = sum(p['weeks'] for p in phases)
    scale = weeks / total_weeks

    plan = [f"### {'Plan nauki' if lang == 'pl' else 'Learning Plan'} — {role}"]
    plan.append(f"**{'Łączny czas' if lang == 'pl' else 'Total time'}:** {weeks} tygodni (~{weeks*hours}h)")

    for phase in phases:
        phase_weeks = max(1, round(phase['weeks'] * scale))
        plan.append(f"\n**{phase['name']} — {phase_weeks} tyg.**")
        plan.append(f"Tematy: {', '.join(phase['topics'])}")
        plan.append("Outputs: notatki, mini-projekt, checklisty.")

    plan.append("\n**Kamienie milowe:**")
    plan.extend([
        "- Tydzień 1: środowisko + EDA",
        "- Tydzień 2–3: model + CV/metryki",
        "- Połowa: mini-projekt (GitHub + README)",
        "- Koniec: wdrożenie API + monitoring"
    ])

    return "\n".join(plan)

def view_add_resource(lang: str):
    """
    Formularz do dodawania nowych zasobów do biblioteki.
    """
    st.header("➕ " + ( "Dodaj nowy zasób" if lang == "pl" else "Add New Resource"))
    st.caption(
        "Wprowadź dane zasobu, aby dodać go do biblioteki."
        if lang == "pl"
        else "Fill in the details to add a new resource to the library."
    )

    with st.form("add_resource_form"):
        title = st.text_input("Tytuł" if lang == "pl" else "Title")
        url = st.text_input("Adres URL")
        category = st.text_input("Kategoria" if lang == "pl" else "Category")
        type_ = st.selectbox(
            "Typ" if lang == "pl" else "Type",
            ["article", "documentation", "book", "guide", "video", "paper", "repo"]
        )
        format_ = st.selectbox("Format", ["html", "pdf", "md", "video"])
        tags = st.text_input("Tagi (przecinki)" if lang == "pl" else "Tags (comma-separated)")
        description = st.text_area("Opis" if lang == "pl" else "Description")
        difficulty = st.selectbox(
            "Poziom trudności" if lang == "pl" else "Difficulty",
            ["beginner", "intermediate", "advanced"]
        )
        rating = st.slider("Ocena" if lang == "pl" else "Rating", 0.0, 5.0, 4.5, 0.1)
        author = st.text_input("Autor" if lang == "pl" else "Author")
        year = st.number_input(T('year', lang), min_value=2000, max_value=2025, value=2025)
        free = st.checkbox("Darmowy" if lang == "pl" else "Free", value=True, key="add_resource_free")
        length_min = st.number_input(
            "Szacowany czas (min)" if lang == "pl" else "Estimated time (min)",
            min_value=10, max_value=1000, value=60
        )
        free = st.checkbox("Darmowy" if lang == "pl" else "Free", value=True)

        submitted = st.form_submit_button("Dodaj zasób" if lang == "pl" else "Add Resource")

        if submitted:
            if not title.strip() or not url.strip() or not category.strip():
                st.error(
                    "Proszę uzupełnić wszystkie wymagane pola."
                    if lang == "pl"
                    else "Please fill in all required fields."
                )
            else:
                conn = connect_db()
                try:
                    conn.execute('''
                        INSERT INTO resources
                        (title, url, category, type, format, tags, description, difficulty, rating, author, year, length_min, free)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        title, url, category, type_, format_, tags, description, difficulty, rating, author, year, length_min, int(free)
                    ))
                    conn.commit()
                    st.success(
                        "Zasób został pomyślnie dodany!"
                        if lang == "pl"
                        else "Resource added successfully!"
                    )
                except Exception as e:
                    st.error(f"Błąd dodawania: {e}" if lang == "pl" else f"Error adding resource: {e}")
                finally:
                    conn.close()


def view_roadmaps(lang: str):
    """
    Widok mapy drogowej — generuje plan nauki dla wybranej roli.
    """
    st.header("🗺️ " + ("Plan Nauki / Roadmapa" if lang == "pl" else "Learning Roadmap"))
    st.caption(
        "Wygeneruj spersonalizowany plan nauki krok po kroku."
        if lang == "pl"
        else "Generate a personalized step-by-step learning plan."
    )

    roles = ["Data Scientist", "ML Engineer", "LLM Engineer"]
    role = st.selectbox("Wybierz rolę" if lang == "pl" else "Choose role", roles)
    weeks = st.slider("Liczba tygodni" if lang == "pl" else "Number of weeks", 4, 52, 24)
    hours = st.slider("Godziny tygodniowo" if lang == "pl" else "Hours per week", 2, 40, 10)

    if st.button("Generuj plan" if lang == "pl" else "Generate plan", type="primary"):
        st.markdown(_generate_plan(role, weeks, hours, lang))


def view_my(lang: str):
    st.header("⭐ Moje / My")
    conn = connect_db()
    tab1, tab2 = st.tabs(["Zakładki / Bookmarks", "Notatki / Notes"])

    with tab1:
        bm = pd.read_sql("""
            SELECT b.created_at, r.id, r.title, r.category, r.type, r.rating, r.url
            FROM bookmarks b JOIN resources r ON b.resource_id=r.id
            ORDER BY b.created_at DESC
        """, conn)
        if bm.empty:
            st.info("Brak zakładek.")
        else:
            st.dataframe(bm, use_container_width=True, hide_index=True)

    with tab2:
        nt = pd.read_sql("""
            SELECT n.created_at, r.id, r.title, n.body
            FROM notes n JOIN resources r ON n.resource_id=r.id
            ORDER BY n.created_at DESC
        """, conn)
        if nt.empty:
            st.info("Brak notatek.")
        else:
            st.dataframe(nt, use_container_width=True, hide_index=True)
    conn.close()


def view_mentor(lang:str):
    st.header("🤖 "+T('mentor',lang)); st.caption(T('mentor_intro',lang))
    q=st.text_area(T('ask',lang), height=160, placeholder="np. Jak zbudować RAG? / How to build a robust RAG?")
    c1,c2,c3,c4=st.columns(4)
    tone=c1.selectbox(T('tone',lang), TEXTS[lang]['tone_opts'])
    length=c2.selectbox(T('length',lang), TEXTS[lang]['length_opts'])
    with c3:
        with st.expander(T('model',lang), expanded=False):
            model=st.selectbox("OpenAI model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)
    rag_on=c4.toggle(T('context_toggle',lang), value=True)
    topk=st.slider(T('topk',lang), 3, 16, 8)
    client=get_openai_client()
    if client is None: st.info(T('no_key',lang))
    if st.button("🚀 "+T('mentor',lang), type="primary", use_container_width=True, disabled=not q.strip()):
        ctx_df=retrieve_context(q, topk=topk) if rag_on else pd.DataFrame()
        ctx_text=build_context_text(ctx_df) if rag_on and not ctx_df.empty else ""
        if client:
            user_msg = "Pytanie/Question: "+q+"\nStyl/Tone: "+tone+", Długość/Length: "+length
            if ctx_text: user_msg += "\nKontekst/Context:\n"+ctx_text
            try:
                resp=client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":MENTOR_SYSTEM_PROMPT_PL},{"role":"user","content":user_msg}],
                    temperature=0.2,
                )
                answer=resp.choices[0].message.content
            except Exception as e:
                st.warning(f"OpenAI error: {e}. OFFLINE.")
                answer=offline_answer(q, ctx_df, lang)
        else:
            answer=offline_answer(q, ctx_df, lang)
        st.markdown("### "+T('answer',lang)); st.write(answer)
        if not ctx_df.empty:
            with st.expander(T('used_sources',lang), expanded=False):
                st.write(ctx_df[['title','url','category','type','rating','tags']])

def view_snippets(lang:str):
    st.header("🧩 "+T('snippets',lang))
    cats=[T('all',lang)]+sorted(set(v['category'] for v in SNIPPETS.values()))
    col1,col2=st.columns([1,2])
    cat=col1.selectbox(T('category',lang), cats)
    q=col1.text_input(T('search',lang), "")
    keys=[k for k,v in SNIPPETS.items() if (cat==T('all',lang) or v['category']==cat) and (q.lower() in k.lower() or q.lower() in v['code'].lower())]
    if not keys: st.info("Brak snippetów / No snippets."); return
    name=col2.selectbox(T('choose_snippet',lang), sorted(keys))
    st.code(SNIPPETS[name]['code'], language=SNIPPETS[name]['lang'])

def view_glossary(lang: str):
    st.header("📚 " + T('glossary', lang))
    topics = OFFLINE_GLOSS_PL if lang == "pl" else OFFLINE_GLOSS_EN
    for term, desc in sorted(topics.items()):
        with st.expander(term):
            st.write(desc)

def view_interview(lang: str):

    # Prosty helper do PL/EN
    L = lambda pl, en: pl if lang == "pl" else en

    st.header("🧠 " + L("Przygotowanie do rozmów", "Interview Prep"))
    st.caption(L("Konkret: pytania z odpowiedziami + fiszki. Zero gadania, sama treść.",
                 "Concrete: Q&A + flashcards. No fluff, just substance."))

    # ====================== DANE ======================
    # Q&A – zwięzłe pytania z treściwymi odpowiedziami
    QA_BANK: Dict[str, List[Dict[str, str]]] = {
        "ML / Data Science": [
            {"q": L("Wyjaśnij bias vs variance.", "Explain bias vs variance."),
             "a": L("Bias=błąd systematyczny; Variance=wrażliwość na dane. Balans: regularyzacja, CV, ensembling.",
                    "Bias=systematic error; Variance=sensitivity to data. Balance via regularization, CV, ensembling.")},
            {"q": L("Czym jest data leakage i jak go unikać?", "What is data leakage and how to avoid it?"),
             "a": L("Przyszła info w treningu. Split po czasie; transformacje/encoding tylko wewnątrz foldów; sanity-checki.",
                    "Future info in training. Time-based split; transforms/encoding inside folds; sanity checks.")},
            {"q": L("Jakie metryki przy niezrównoważeniu klas?", "Which metrics for class imbalance?"),
             "a": L("F1, PR-AUC, Recall@k; Accuracy bywa mylące. Dobieraj do kosztu FP/FN.",
                    "F1, PR-AUC, Recall@k; Accuracy may mislead. Choose by FP/FN cost.")},
            {"q": L("Kiedy model liniowy, a kiedy tree-based?", "When linear vs tree-based?"),
             "a": L("Linear: interpretowalność, dużo cech. GBM/Trees: nieliniowości, interakcje, mniej FE.",
                    "Linear: interpretability, many features. GBM/Trees: nonlinearity, interactions, less FE.")},
        ],
        "Deep Learning": [
            {"q": L("BatchNorm vs Dropout — po co i kiedy?", "BatchNorm vs Dropout — why and when?"),
             "a": L("BN stabilizuje/szybszy trening; Dropout=regularizacja. CNN: częściej BN; Transformers: dropout w kilku miejscach.",
                    "BN stabilizes/speeds; Dropout regularizes. CNN: more BN; Transformers: dropout in several spots.")},
            {"q": L("Jaki harmonogram LR wybrać?", "Which LR schedule?"),
             "a": L("Cosine/One-cycle z warmupem; monitoruj val loss/plateau; unikaj zbyt agresywnego decay.",
                    "Cosine/One-cycle with warmup; monitor val loss/plateau; avoid overly aggressive decay.")},
            {"q": L("Gradient clipping — dlaczego?", "Gradient clipping — why?"),
             "a": L("Ogranicza normę gradientu (np. 1.0) i stabilizuje trening przy eksplozji gradientów.",
                    "Bounds gradient norm (e.g., 1.0) to stabilize training under exploding gradients.")},
            {"q": L("Transfer learning — kiedy pomaga?", "Transfer learning — when?"),
             "a": L("Mało danych; domena zbliżona do pretrainingu. Zamrażaj niższe warstwy, fine-tune wyższe.",
                    "Low data; domain close to pretraining. Freeze lower layers, fine-tune higher.")},
        ],
        "MLOps / Cloud": [
            {"q": L("Po co Model Registry (MLflow)?", "Why a Model Registry?"),
             "a": L("Wersjonowanie, stage (Staging→Prod), audyt, rollback, porównania.",
                    "Versioning, stages (Staging→Prod), audit, rollback, comparisons.")},
            {"q": L("Canary vs A/B — kiedy co?", "Canary vs A/B — when?"),
             "a": L("Canary: wdrożenia prod z szybkim rollbackiem. A/B: eksperyment hipotez.",
                    "Canary: prod rollout with quick rollback. A/B: hypothesis testing.")},
            {"q": L("Shadow traffic — co to?", "Shadow traffic — what is it?"),
             "a": L("Lustrzane żądania do nowego modelu bez wpływu na userów — zbierasz metryki.",
                    "Mirrored requests to the new model without user impact — you collect metrics.")},
            {"q": L("Jak monitorować drift?", "How to monitor drift?"),
             "a": L("Statystyki cech/wyjść, EMD/PSI, alarmy, porównania rozkładów, okresowy retrain.",
                    "Feature/output stats, EMD/PSI, alerts, dist. comparisons, periodic retrain.")},
        ],
        "Behavioral": [
            {"q": L("Opowiedz o porażce (STAR).", "Describe a failure (STAR)."),
             "a": L("Sytuacja→Zadanie→Działania→Rezultat; pokaż ownership i lekcje.",
                    "Situation→Task→Action→Result; show ownership and lessons.")},
            {"q": L("Jak priorytetyzujesz pod presją?", "How do you prioritize under pressure?"),
             "a": L("Matryca ważne/pilne, szybka walidacja ryzyk, komunikacja, timeboxing.",
                    "Urgent/important matrix, quick risk validation, communication, timeboxing.")},
            {"q": L("Konflikt w zespole — podejście?", "Team conflict — approach?"),
             "a": L("Empatia, fakty bez oskarżeń, wspólne rozwiązania, domknięcie.",
                    "Empathy, facts without blame, collaborative solutions, closure.")},
            {"q": L("Czego szukasz w nowej roli?", "What do you look for in a new role?"),
             "a": L("Wpływ na produkt, wzrost, kultura feedbacku, sensowne procesy techniczne.",
                    "Product impact, growth, feedback culture, solid tech processes.")},
        ],
    }

    # FLASHCARDS — po 10 na kategorię (ML, DL, MLOps, Behavioral)
    FLASHCARDS: Dict[str, List[Dict[str, str]]] = {
        "ML": [
            {"q": L("Overfitting — definicja?", "Define overfitting."),
             "a": L("Zbyt dobre dopasowanie do treningu, słaba generalizacja.", "Too good fit to train, poor generalization.")},
            {"q": L("Cross-validation — po co?", "Why cross-validation?"),
             "a": L("Stabilna estymacja jakości, mniejsza wariancja.", "Stable quality estimate, lower variance.")},
            {"q": L("Precision vs Recall?", "Precision vs Recall?"),
             "a": L("Precision=czystość; Recall=pokrycie pozytywów.", "Precision=purity; Recall=coverage.")},
            {"q": L("F1 — kiedy używać?", "F1 — when to use?"),
             "a": L("Gdy ważny balans P/R i niezrównoważone klasy.", "When balancing P/R under imbalance.")},
            {"q": L("PR-AUC vs ROC-AUC?", "PR-AUC vs ROC-AUC?"),
             "a": L("PR-AUC lepsze przy rzadkich pozytywach.", "PR-AUC better for rare positives.")},
            {"q": L("Target leakage — przykład?", "Target leakage — example?"),
             "a": L("Cecha obliczona z targetu.", "Feature derived from target.")},
            {"q": L("Regularization — rodzaje?", "Regularization — types?"),
             "a": L("L1/L2/ElasticNet; w DL: dropout, weight decay.", "L1/L2/ElasticNet; in DL: dropout, weight decay.")},
            {"q": L("Feature scaling — kiedy?", "Feature scaling — when?"),
             "a": L("Linear, kNN, SVM; drzewa mniej wrażliwe.", "Linear, kNN, SVM; trees less sensitive.")},
            {"q": L("Kalibracja — po co?", "Calibration — why?"),
             "a": L("Lepsze prawdopodobieństwa (Platt/Isotonic).", "Better probabilities (Platt/Isotonic).")},
            {"q": L("KFold vs StratifiedKFold?", "KFold vs StratifiedKFold?"),
             "a": L("Stratified utrzymuje proporcje klas.", "Stratified preserves class ratios.")},
        ],
        "DL": [
            {"q": L("Dropout — rola?", "Role of Dropout?"),
             "a": L("Regularizacja przez losową dezaktywację.", "Regularization via random deactivation.")},
            {"q": L("BatchNorm — cel?", "BatchNorm — purpose?"),
             "a": L("Stabilizuje aktywacje, szybszy trening.", "Stabilizes activations, faster training.")},
            {"q": L("Warmup LR — po co?", "Why LR warmup?"),
             "a": L("Stabilizuje start treningu.", "Stabilizes early training.")},
            {"q": L("Residual connections — plusy?", "Residual connections — benefits?"),
             "a": L("Lepszy przepływ gradientu, głębsze sieci.", "Better gradient flow, deeper nets.")},
            {"q": L("Attention — intuicja?", "Attention — intuition?"),
             "a": L("Wagę na ważne tokeny, adaptacyjny kontekst.", "Weights important tokens, adaptive context.")},
            {"q": L("Label smoothing — po co?", "Label smoothing — why?"),
             "a": L("Lepsza kalibracja, mniejsza pewność.", "Better calibration, less overconfidence.")},
            {"q": L("Data augmentation — przykłady?", "Data augmentation — examples?"),
             "a": "Flip, crop, jitter, mixup, cutmix."},
            {"q": L("Gradient clipping — kiedy?", "Gradient clipping — when?"),
             "a": L("Przy eksplodujących gradientach.", "When gradients explode.")},
            {"q": L("One-cycle policy — idea?", "One-cycle policy — idea?"),
             "a": L("LR rośnie, potem maleje; szybka konwergencja.", "LR rises then falls; faster convergence.")},
            {"q": L("Early stopping — rola?", "Early stopping — role?"),
             "a": L("Stop przy braku poprawy na walidacji.", "Stop when validation plateaus.")},
        ],
        "MLOps": [
            {"q": L("Model Registry — funkcja?", "Model Registry — function?"),
             "a": L("Wersjonowanie modeli, stage, audyt.", "Model versioning, stages, audit.")},
            {"q": L("Canary rollout — co to?", "Canary rollout — what?"),
             "a": L("Stopniowe wdrożenie, szybki rollback.", "Gradual rollout, quick rollback.")},
            {"q": L("Shadow traffic — sens?", "Shadow traffic — why?"),
             "a": L("Metryki bez wpływu na userów.", "Collect metrics without user impact.")},
            {"q": L("Data/Concept drift — różnica?", "Data/Concept drift — difference?"),
             "a": L("Data=wejścia się zmieniają; Concept=zależność X→y.", "Data=input shift; Concept=relationship shift.")},
            {"q": L("Monitoring — co mierzyć?", "Monitoring — what to measure?"),
             "a": L("Latency, error rate, koszt, drift, kalibracja.", "Latency, errors, cost, drift, calibration.")},
            {"q": L("Feature store — korzyści?", "Feature store — benefits?"),
             "a": L("Spójność online/offline, reuse, lineage.", "Online/offline parity, reuse, lineage.")},
            {"q": L("CI/CD dla ML — składowe?", "CI/CD for ML — components?"),
             "a": L("Testy, lint, build, registry, deploy, rollback.", "Tests, lint, build, registry, deploy, rollback.")},
            {"q": L("Observability — narzędzia?", "Observability — tools?"),
             "a": L("OpenTelemetry, Prom/Grafana, Sentry.", "OpenTelemetry, Prom/Grafana, Sentry.")},
            {"q": L("A/B vs Feature Flag?", "A/B vs Feature Flag?"),
             "a": L("A/B=eksperyment; FF=kontrola widoczności.", "A/B=experiment; FF=visibility control.")},
            {"q": L("Reproducibility — jak?", "Reproducibility — how?"),
             "a": L("Seed, wersje danych/kodu, kontenery.", "Seed, data/code versions, containers.")},
        ],
        "Behavioral": [
            {"q": L("STAR — rozwinięcie?", "STAR — expand?"),
             "a": "Situation, Task, Action, Result"},
            {"q": L("Ownership — co pokażesz?", "Ownership — what to show?"),
             "a": L("Inicjatywa, odpowiedzialność, domykanie.", "Initiative, responsibility, closing the loop.")},
            {"q": L("Komunikacja — klucze?", "Communication — keys?"),
             "a": L("Regularne update’y, ryzyka, decyzje.", "Regular updates, risks, decisions.")},
            {"q": L("Priorytetyzacja — narzędzie?", "Prioritization — tool?"),
             "a": L("Matryca ważne/pilne, ICE/RICE.", "Urgent/important matrix, ICE/RICE.")},
            {"q": L("Feedback — jak przyjmować?", "Feedback — how to receive?"),
             "a": L("Słuchaj, parafrazuj, plan działania.", "Listen, paraphrase, action plan.")},
            {"q": L("Konflikt — pierwsze kroki?", "Conflict — first steps?"),
             "a": L("Fakty, empatia, cel wspólny.", "Facts, empathy, common goal.")},
            {"q": L("Pytania do rekrutera — przykład?", "Questions to interviewer — example?"),
             "a": L("Wskaźniki sukcesu roli, roadmap, kultura code review.", "Role success metrics, roadmap, code review culture.")},
            {"q": L("Gdy czegoś nie wiesz?", "When you don't know something?"),
             "a": L("Powiedz szczerze, opisz jak sprawdzisz.", "Say honestly, outline how you'll verify.")},
            {"q": L("Największy wpływ — przykład?", "Biggest impact — example?"),
             "a": L("Użyj liczb: % poprawy, koszty, czas.", "Use numbers: %, cost, time.")},
            {"q": L("Deadline jutro — co robisz?", "Deadline tomorrow — what do you do?"),
             "a": L("MVP, ryzyka, komunikacja.", "MVP, risks, communication.")},
        ],
    }

    # ====================== ZAKŁADKI ======================
    tab_qa, tab_cards = st.tabs([L("Pytania i odpowiedzi", "Q&A"),
                                 L("Fiszki", "Flashcards")])

    # --------- Q&A ----------
    with tab_qa:
        st.subheader("📌 " + L("Pytania i odpowiedzi (konkret)", "Q&A (concise & exact)"))
        cat = st.selectbox(L("Kategoria", "Category"), list(QA_BANK.keys()), index=0)
        qlist = QA_BANK.get(cat, [])
        if not qlist:
            st.info(L("Brak pytań w tej kategorii.", "No questions in this category."))
        else:
            search = st.text_input(L("Filtr (fraza/słowa kluczowe)", "Filter (phrase/keywords)"), "")
            if search.strip():
                s = search.strip().lower()
                filtered = [qa for qa in qlist if s in qa["q"].lower() or s in qa["a"].lower()]
            else:
                filtered = qlist

            st.caption(L(f"Wyświetlam {len(filtered)} z {len(qlist)} pytań.",
                         f"Showing {len(filtered)} of {len(qlist)} questions."))

            for i, qa in enumerate(filtered, 1):
                with st.expander(f"{i}. {qa['q']}"):
                    st.markdown(qa["a"])

    # --------- FLASHCARDS ----------
    with tab_cards:
        st.subheader("🃏 " + L("Fiszki — szybka powtórka", "Flashcards — quick drill"))

        col = st.columns(3)
        card_cat = col[0].selectbox(
            L("Kategoria fiszek", "Flashcard category"),
            list(FLASHCARDS.keys()), index=0
        )
        mode = col[1].selectbox(
            L("Tryb", "Mode"),
            [L("Losowa fiszka", "Random card"), L("Quiz: 10 pytań", "Quiz: 10 cards")],
            index=0
        )
        show_ans_default = col[2].checkbox(
            L("Pokaż odpowiedzi od razu", "Show answers immediately"),
            value=False
        )

        # ------ Losowa fiszka ------
        if mode == L("Losowa fiszka", "Random card"):
            if st.button(L("Wylosuj fiszkę", "Draw a card"), key="fc_draw"):
                card = random.choice(FLASHCARDS[card_cat])
                st.session_state._rcard = {"q": card["q"], "a": card["a"],
                                           "revealed": bool(show_ans_default), "cat": card_cat}

            r = st.session_state.get("_rcard")
            if r and r.get("cat") != card_cat:
                st.session_state.pop("_rcard", None)
                r = None
            r = st.session_state.get("_rcard")

            if r:
                st.info("**Q:** " + r["q"])
                if not r.get("revealed", False):
                    if st.button(L("Pokaż odpowiedź", "Show answer"), key="fc_reveal"):
                        r["revealed"] = True
                        st.session_state._rcard = r
                        st.rerun()
                if r.get("revealed", False):
                    st.success("**A:** " + r["a"])

                c1, c2 = st.columns(2)
                if c1.button(L("Następna", "Next"), key="fc_next"):
                    card = random.choice(FLASHCARDS[card_cat])
                    st.session_state._rcard = {"q": card["q"], "a": card["a"],
                                               "revealed": bool(show_ans_default), "cat": card_cat}
                    st.rerun()
                if c2.button("Reset", key="fc_reset_one"):
                    st.session_state.pop("_rcard", None)
                    st.rerun()
            else:
                st.caption(L("Kliknij „Wylosuj fiszkę”, aby zacząć.",
                             "Click “Draw a card” to begin."))

        # ------ Quiz: 10 pytań ------
        else:
            pool = FLASHCARDS[card_cat]
            QUIZ_N = 10
            if len(pool) < QUIZ_N:
                st.warning(L(f"Za mało fiszek w tej kategorii (min. {QUIZ_N}).",
                             f"Not enough flashcards in this category (min. {QUIZ_N})."))
            else:
                if st.button(L(f"Start quizu ({QUIZ_N})", f"Start quiz ({QUIZ_N})"), key="fc_start"):
                    st.session_state._fc = {"items": random.sample(pool, QUIZ_N),
                                            "idx": 0, "score": 0, "revealed": False}
                    st.rerun()

                sess = st.session_state.get("_fc")
                if sess:
                    i = sess["idx"]
                    if i < len(sess["items"]):
                        card = sess["items"][i]
                        st.markdown(f"**{i+1}/{len(sess['items'])}** — {card['q']}")

                        if show_ans_default or sess["revealed"]:
                            st.success("**A:** " + card["a"])
                        else:
                            if st.button(L("Pokaż odpowiedź", "Reveal"), key=f"fc_rev_{i}"):
                                sess["revealed"] = True
                                st.session_state._fc = sess
                                st.rerun()

                        c1, c2 = st.columns(2)
                        if c1.button(L("Zaliczone", "Correct"), key=f"fc_ok_{i}"):
                            sess["score"] += 1
                            sess["idx"] += 1
                            sess["revealed"] = False
                            st.session_state._fc = sess
                            st.rerun()

                        if c2.button(L("Pomin / Następne", "Skip / Next"), key=f"fc_skip_{i}"):
                            sess["idx"] += 1
                            sess["revealed"] = False
                            st.session_state._fc = sess
                            st.rerun()
                    else:
                        st.success(L(f"Wynik: {sess['score']}/{len(sess['items'])}",
                                     f"Score: {sess['score']}/{len(sess['items'])}"))
                        if st.button("Reset", key="fc_reset"):
                            st.session_state.pop("_fc", None)
                            st.rerun()

    # ===================== Coding Katas =====================
        katas = [
            {
                "name": "Confusion matrix + metrics",
                "desc": L("Policz precision/recall/F1 i narysuj macierz pomyłek.",
                          "Compute precision/recall/F1 and plot a confusion matrix."),
                "snippet": """\
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
y_true = pd.read_csv("y_true.csv")['y']
y_pred = pd.read_csv("y_pred.csv")['yhat']
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))"""
            },
            {
                "name": "TimeSeriesSplit + model",
                "desc": L("Walidacja szeregów czasowych z modelem regresyjnym.",
                          "TimeSeriesSplit validation with a regressor."),
                "snippet": """\
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import numpy as np
X, y = np.random.randn(1000,20), np.random.randn(1000)
tscv = TimeSeriesSplit(n_splits=5)
for i,(tr,te) in enumerate(tscv.split(X)):
    # fit your model here; example:
    pred = y[tr].mean() * np.ones_like(y[te])
    print(i, "MAE", mean_absolute_error(y[te], pred))"""
            },
            {
                "name": "RAG: BM25 + FAISS szkic",
                "desc": L("Szkic hybrydowego retrievera: BM25 + FAISS.",
                          "Sketch of a hybrid retriever: BM25 + FAISS."),
                "snippet": """\
# Use real libraries in practice (rank-bm25, faiss). This is a high-level sketch:
# 1) build BM25 over tokenized docs; 2) build vector index (FAISS); 3) fuse ranks (e.g., reciprocal rank fusion)."""
            },
            {
                "name": "Feature importance (SHAP) szkic",
                "desc": L("Szkic obliczania SHAP dla modelu tree-based.",
                          "Sketch of computing SHAP for a tree-based model."),
                "snippet": """\
# pip install shap
import shap, xgboost as xgb
# model = xgb.XGBClassifier(...).fit(X_train, y_train)
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_valid)
# shap.summary_plot(shap_values, X_valid)"""
            },
        ]
        which = st.selectbox("Zadanie / Kata", [k["name"] for k in katas])
        sel = next(k for k in katas if k["name"] == which)
        st.write(sel["desc"])
        st.code(sel["snippet"], language="python")

    # ===================== System Design (ML/LLM) =====================
        templates = {
            "Recommender (ranking)": [
                L("Warstwy: candidate gen → ranker; cechy: historia, kontekst, popularność; AB-testy.",
                  "Layers: candidate gen → ranker; features: history, context, popularity; A/B tests."),
                L("Online vs offline features; budżet latencji; cache i fallback.",
                  "Online vs offline features; latency budget; cache and fallback."),
            ],
            "RAG Q&A (enterprise)": [
                L("Ingest: OCR→clean→chunk; indeks: BM25+ANN; reranking; cytaty/grounding.",
                  "Ingest: OCR→clean→chunk; index: BM25+ANN; reranking; citations/grounding."),
                L("Guardrails: filtr PII/tox, whitelisting tool calls; audit log; kosztometry.",
                  "Guardrails: PII/tox filters, tool call whitelists; audit logs; cost metering."),
            ],
            "Streaming scoring (real-time ML)": [
                L("Kafka→przetwarzanie strumieniowe→feature store online→serving; monitoring SLO, drift.",
                  "Kafka→stream processing→online feature store→serving; SLO monitoring, drift."),
                L("Canary rollout, rollback, shadow traffic; tracing i metryki.",
                  "Canary rollout, rollback, shadow traffic; tracing and metrics."),
            ],
        }
        choice = st.selectbox("Wzorzec / Template", list(templates.keys()))
        for bullet in templates[choice]:
            st.markdown(f"- {bullet}")

    # ===================== Math & Stats =====================
        st.subheader(L("Ściąga / Cheats", "Cheats"))
        st.markdown("- CLT, prawo wielkich liczb, rozkłady: Normal, Bernoulli, Binomial, Poisson\n"
                    "- Estymatory: nieobciążony/zgodny/efektywny; przedziały ufności\n"
                    "- Testy: z, t-test, chi-kwadrat; korekcja wielokrotna (Bonferroni/Benjamini-Hochberg)\n"
                    "- A/B: moc, istotność, MDE, efekt; CUPED; sequential tests")
        st.subheader(L("Szybkie pytania", "Quick-fire"))
        qz = [
            (L("Czym różni się MAE od RMSE?", "Difference between MAE and RMSE?"),
             L("RMSE silniej karze outliery; MAE odporniejszy.", "RMSE penalizes outliers more; MAE is more robust.")),
            (L("Co to jest MDE w A/B?", "What is MDE in A/B?"),
             L("Minimal Detectable Effect – najmniejsza zmiana, którą chcesz wykryć z zadaną mocą.",
               "Minimal Detectable Effect – smallest change detectable with given power.")),
        ]
        for i,(q,a) in enumerate(qz,1):
            with st.expander(f"{i}. {q}"):
                st.write(a)

    
# ====================== Init & Main ======================
def init_database(lang:str):
    conn=connect_db(); ensure_schema(conn)
    if db_count(conn)==0: add_resources_bulk(conn, build_default_resources(lang))
    conn.close()
def reset_all_data():
    st.session_state.completed_resources=[]; st.session_state.study_hours=0; st.session_state.current_streak=0
    conn=connect_db(); conn.execute('DELETE FROM user_activity'); conn.execute('UPDATE resources SET views=0'); conn.commit(); conn.close()

def main():
    init_session_state()
    init_database(st.session_state.language)
    lang=st.session_state.language

    st.title(T('title',lang)); st.caption(T('subtitle',lang))
    with st.sidebar:
        st.header(T('settings',lang))
        new_lang=st.selectbox(T('language',lang), list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x],
                              index=list(LANGUAGES.keys()).index(lang))
        if new_lang!=lang:
            st.session_state.language=new_lang; st.rerun()
        key_in=st.text_input(T('openai_key',lang), type="password", value=st.session_state.get('openai_key') or "", placeholder="sk-...")
        if key_in and key_in!=st.session_state.get('openai_key'):
            st.session_state.openai_key=key_in; st.success("Saved / Zapisano (do końca sesji).")
        st.divider()
        if st.button(T('reset_data',lang)):
            if st.button(T('confirm_reset',lang)):
                reset_all_data(); st.success(T('data_reset',lang)); st.rerun()
        st.divider()
        page=st.radio("Menu", [T('dashboard',lang),T('library',lang),T('mentor',lang),T('snippets',lang),
                               T('glossary',lang),T('interview',lang),T('cases',lang),T('roadmaps',lang),T('add_resource',lang)],
                      label_visibility="collapsed")

    if page==T('dashboard',lang): view_dashboard(lang)
    elif page==T('library',lang): view_library(lang)
    elif page==T('mentor',lang): view_mentor(lang)
    elif page==T('snippets',lang): view_snippets(lang)
    elif page==T('glossary',lang): view_glossary(lang)
    elif page==T('interview',lang): view_interview(lang)
    elif page==T('cases',lang): view_cases(lang)
    elif page==T('roadmaps',lang): view_roadmaps(lang)
    elif page==T('add_resource',lang): view_add_resource(lang)

if __name__=="__main__":
    main()