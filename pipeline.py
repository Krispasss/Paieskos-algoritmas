import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
DATA_PARQUET = Path("data/dataset.parquet")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_dataset(parquet_path: str | Path = DATA_PARQUET, max_samples: int = 10000) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    if "text" not in df.columns:
        title = df.get("title", "").fillna("").astype(str)
        desc = df.get("description", "").fillna("").astype(str)
        cont = df.get("content", "").fillna("").astype(str)
        df["text"] = (title.str.strip() + ". " + desc.str.strip() + " " + cont.str.strip()).str.strip()

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() >= 50].drop_duplicates(subset=["text"]).reset_index(drop=True)

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"Naudojama straipsnių: {len(df)}")

    return df


# -----------------------------
# 1) KEYWORDS išgavimas
# -----------------------------
def extract_keywords_per_article(df: pd.DataFrame, top_n: int = 6, max_chars: int = 1200) -> pd.DataFrame:
    kw_model = KeyBERT(model=MODEL_NAME)
    texts = df["text"].str.slice(0, max_chars).tolist()

    keywords_list = []
    for t in tqdm(texts, desc="KeyBERT keywords per article"):
        kws = kw_model.extract_keywords(
            t,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n
        )
        keywords_list.append([k for k, _ in kws])

    out = df.copy()
    out["keywords"] = keywords_list
    return out


def build_keyword_vocabulary(df: pd.DataFrame, min_df: int = 20, max_vocab: int = 5000) -> list[str]:
    counter = Counter()
    for kws in df["keywords"]:
        counter.update(set(kws))

    vocab = [k for k, c in counter.items() if c >= min_df]
    vocab = sorted(vocab, key=lambda k: counter[k], reverse=True)[:max_vocab]
    return vocab


# -----------------------------
# 2) KEYWORDS klasterizacija -> temos
# -----------------------------
def cluster_keywords(vocab: list[str], n_topics: int = 5) -> tuple[KMeans, dict]:
    model = SentenceTransformer(MODEL_NAME)
    kw_emb = model.encode(vocab, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    km = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
    labels = km.fit_predict(kw_emb)

    kw_to_topic = {kw: int(lbl) for kw, lbl in zip(vocab, labels)}
    return km, kw_to_topic


def assign_articles_to_topics(df: pd.DataFrame, kw_to_topic: dict, n_topics: int = 5) -> pd.DataFrame:
    topic_ids = []
    for kws in df["keywords"]:
        votes = [0] * n_topics
        for k in kws:
            if k in kw_to_topic:
                votes[kw_to_topic[k]] += 1

        topic_ids.append(int(np.argmax(votes)) if sum(votes) > 0 else 0)

    out = df.copy()
    out["topic_id"] = topic_ids
    return out


def build_topic_keywords(df: pd.DataFrame, kw_to_topic: dict, n_topics: int = 5, top_k: int = 10) -> dict:
    topic_counts = {t: Counter() for t in range(n_topics)}
    for kws, t in zip(df["keywords"], df["topic_id"]):
        for k in kws:
            if k in kw_to_topic and kw_to_topic[k] == t:
                topic_counts[t][k] += 1

    return {t: [k for k, _ in topic_counts[t].most_common(top_k)] for t in range(n_topics)}


# -----------------------------
# 3) Straipsnių embeddings SEMANTINEI paieškai
# -----------------------------
def build_article_embeddings(df: pd.DataFrame, batch_size: int = 128, max_chars: int = 2500) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    texts = df["text"].str.slice(0, max_chars).tolist()
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


# -----------------------------
# Save/Load
# -----------------------------
def save_artifacts(df: pd.DataFrame, embeddings: np.ndarray, topic_keywords: dict, kw_to_topic: dict, km_keywords: KMeans, n_topics: int):
    df.to_parquet(ARTIFACTS_DIR / "articles.parquet", index=False)
    np.save(ARTIFACTS_DIR / "embeddings.npy", embeddings)

    (ARTIFACTS_DIR / "topic_keywords.json").write_text(json.dumps(topic_keywords, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "kw_to_topic.json").write_text(json.dumps(kw_to_topic, ensure_ascii=False, indent=2), encoding="utf-8")
    dump(km_keywords, ARTIFACTS_DIR / "kmeans_keywords.pkl")

    (ARTIFACTS_DIR / "meta.json").write_text(json.dumps({"n_topics": n_topics}, indent=2), encoding="utf-8")


def load_artifacts():
    df = pd.read_parquet(ARTIFACTS_DIR / "articles.parquet")
    embeddings = np.load(ARTIFACTS_DIR / "embeddings.npy")

    topic_keywords = json.loads((ARTIFACTS_DIR / "topic_keywords.json").read_text(encoding="utf-8"))
    topic_keywords = {int(k): v for k, v in topic_keywords.items()}

    kw_to_topic = json.loads((ARTIFACTS_DIR / "kw_to_topic.json").read_text(encoding="utf-8"))
    km_keywords = load(ARTIFACTS_DIR / "kmeans_keywords.pkl")

    meta_path = ARTIFACTS_DIR / "meta.json"
    meta = {"n_topics": int(df["topic_id"].nunique())}
    if meta_path.exists() and meta_path.stat().st_size > 0:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return df, embeddings, topic_keywords, kw_to_topic, km_keywords, meta


# -----------------------------
# Full pipeline
# -----------------------------
def full_pipeline(n_topics: int = 5, top_n_keywords_per_article: int = 6, min_df: int = 20):
    df = load_dataset(DATA_PARQUET)

    # 1) keywords
    df = extract_keywords_per_article(df, top_n=top_n_keywords_per_article)

    # 2) keyword vocab + clustering into topics
    vocab = build_keyword_vocabulary(df, min_df=min_df, max_vocab=5000)
    km_keywords, kw_to_topic = cluster_keywords(vocab, n_topics=n_topics)

    # 3) assign articles to topics (based on keyword groups)
    df = assign_articles_to_topics(df, kw_to_topic, n_topics=n_topics)

    # 4) topic keywords for UI
    topic_keywords = build_topic_keywords(df, kw_to_topic, n_topics=n_topics, top_k=10)

    # 5) article embeddings for semantic search (word or sentence)
    embeddings = build_article_embeddings(df)

    save_artifacts(df, embeddings, topic_keywords, kw_to_topic, km_keywords, n_topics)
    return df, embeddings, topic_keywords, kw_to_topic, km_keywords, {"n_topics": n_topics}


# -----------------------------
# Search
# -----------------------------
def search(query: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 5) -> pd.DataFrame:
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, embeddings)[0]

    idx = np.argsort(-sims)[:top_k]
    res = df.iloc[idx].copy()
    res["score"] = sims[idx]
    return res[["title", "topic_id", "score", "text"]]