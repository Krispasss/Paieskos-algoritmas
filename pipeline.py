import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from joblib import dump, load
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

ARTIFACTS_DIR = Path("artifacts")   
ARTIFACTS_DIR.mkdir(exist_ok=True)

DATA_PARQUET = Path("data/dataset.parquet")


def load_dataset(parquet_path: str | Path = DATA_PARQUET) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    # garantuojam, kad yra "text"
    if "text" not in df.columns:
        title = df.get("title", "").fillna("").astype(str)
        desc = df.get("description", "").fillna("").astype(str)
        cont = df.get("content", "").fillna("").astype(str)
        df["text"] = (title.str.strip() + ". " + desc.str.strip() + " " + cont.str.strip()).str.strip()

    # minimalus valymas
    df["text"] = df["text"].fillna("").astype(str)
    if "title" in df.columns:
        df["title"] = df["title"].fillna("").astype(str)
    else:
        df["title"] = df["text"].str.slice(0, 80)

    df = df[df["text"].str.len() >= 50].drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def extract_keywords(df: pd.DataFrame, top_n: int = 6, limit: int = 15000) -> pd.DataFrame:
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")

    out = df.copy()
    out["keywords"] = [[] for _ in range(len(out))]

    texts = out["text"].str.slice(0, 1200).tolist()

    n = min(limit, len(out))
    for i in tqdm(range(n), desc="KeyBERT keywords (limited)"):
        t = texts[i]
        kws = kw_model.extract_keywords(
            t,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
        )
        out.at[i, "keywords"] = [k for k, _ in kws]

    return out


def build_embeddings(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 128) -> np.ndarray:
    model = SentenceTransformer(model_name)

    texts = df["text"].str.slice(0, 2500).tolist()  # užtenka temoms/paieškai
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def cluster_topics(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(embeddings)
    return km


def full_pipeline(n_topics: int = 5):
    df = load_dataset(DATA_PARQUET)

    df = extract_keywords(df, top_n=6)
    embeddings = build_embeddings(df, batch_size=128)

    km = cluster_topics(embeddings, n_clusters=n_topics)

    # VIDUJE laikom 0..k-1, bet GUI rodysim +1
    df["topic_id"] = km.labels_

    topic_keywords = build_topic_keywords(df, top_k=10)

    # išsisaugom ir n_topics, kad app žinotų
    save_artifacts(df, embeddings, km, topic_keywords)
    (ARTIFACTS_DIR / "meta.json").write_text(
        json.dumps({"n_topics": n_topics}, indent=2),
        encoding="utf-8"
    )
    return df, embeddings, km, topic_keywords


def load_artifacts():
    df = pd.read_parquet(ARTIFACTS_DIR / "articles.parquet")
    embeddings = np.load(ARTIFACTS_DIR / "embeddings.npy")
    km = load(ARTIFACTS_DIR / "kmeans.pkl")
    topic_keywords = json.loads((ARTIFACTS_DIR / "topic_keywords.json").read_text(encoding="utf-8"))
    topic_keywords = {int(k): v for k, v in topic_keywords.items()}

    meta_path = ARTIFACTS_DIR / "meta.json"
    meta = {"n_topics": int(df["topic_id"].nunique())} if "topic_id" in df.columns else {"n_topics": 5}
    if meta_path.exists() and meta_path.stat().st_size > 0:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return df, embeddings, km, topic_keywords, meta


def build_topic_keywords(df: pd.DataFrame, top_k: int = 10) -> dict:
    """
    Imame dažniausias KeyBERT frazes kiekvienoje temoje.
    """
    topic_keywords = {}
    for topic_id, group in df.groupby("topic_id"):
        all_kws = []
        for kws in group["keywords"]:
            all_kws.extend(kws)
        counts = Counter(all_kws)
        topic_keywords[int(topic_id)] = [k for k, _ in counts.most_common(top_k)]
    return topic_keywords


def save_artifacts(df: pd.DataFrame, embeddings: np.ndarray, km: KMeans, topic_keywords: dict):
    df.to_parquet(ARTIFACTS_DIR / "articles.parquet", index=False)
    np.save(ARTIFACTS_DIR / "embeddings.npy", embeddings)
    dump(km, ARTIFACTS_DIR / "kmeans.pkl")
    (ARTIFACTS_DIR / "topic_keywords.json").write_text(
        json.dumps(topic_keywords, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def search(query: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 5,
           model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, embeddings)[0]

    idx = np.argsort(-sims)[:top_k]
    res = df.iloc[idx].copy()
    res["score"] = sims[idx]
    return res[["title", "topic_id", "score", "text"]]