import streamlit as st
from pathlib import Path

from pipeline import full_pipeline, load_artifacts, search

st.set_page_config(page_title="AI Search Lab", layout="wide")
st.title("Paieškos algoritmų tyrimas: temos + paieška")

ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

ART_ARTICLES = ART_DIR / "articles.parquet"
ART_EMB = ART_DIR / "embeddings.npy"
ART_KM = ART_DIR / "kmeans.pkl"
ART_TOPICS = ART_DIR / "topic_keywords.json"
ART_META = ART_DIR / "meta.json"

def artifacts_ok() -> bool:
    required = [ART_ARTICLES, ART_EMB, ART_KM, ART_TOPICS]  # meta.json nebereikalingas
    if not all(p.exists() for p in required):
        return False
    if ART_ARTICLES.stat().st_size == 0:
        return False
    return True

with st.sidebar:
    st.header("Nustatymai")
    n_topics = st.slider("Temų skaičius (KMeans klasteriai)", 2, 10, 5)

    if st.button("Perskaičiuoti temas (paleisti pipeline)"):
        with st.spinner("Skaičiuojama..."):
            full_pipeline(n_topics=n_topics)
        st.success(f"Paruošta! Temos: {n_topics}")

if not artifacts_ok():
    st.warning("Artifacts dar nėra (arba sugadinti). Paspausk: 'Perskaičiuoti temas (paleisti pipeline)'.")
    st.stop()

df, embeddings, km, topic_keywords, meta = load_artifacts()
current_topics = int(meta.get("n_topics", df["topic_id"].nunique()))

if current_topics != n_topics:
    st.info(f"Dabar užkrauti artifacts turi {current_topics} temas. Jei nori {n_topics}, paspausk kairėje 'Perskaičiuoti temas'.")

# --- Paieška ---
st.subheader("Paieška")
query = st.text_input("Įvesk paieškos frazę", value="")
top_k = st.slider("Rezultatų kiekis", 3, 10, 5)

if query.strip():
    results = search(query, df, embeddings, top_k=top_k)
    for _, row in results.iterrows():
        st.markdown(f"### {row['title']}")
        st.write(f"Tema: {int(row['topic_id']) + 1} | Similarity: {row['score']:.3f}")  # +1 numeracijai
        st.caption(row["text"][:350] + "...")
        st.divider()

# --- Temos ---
st.subheader(f"Temos ({current_topics})")

# dinaminis stulpelių skaičius pagal temų kiekį
cols = st.columns(min(current_topics, 5))  # kad nesuspaustų per daug; jei >5, rodysim eilėmis

topics = list(range(current_topics))
for i, t in enumerate(topics):
    col = cols[i % len(cols)]
    with col:
        st.markdown(f"## Tema {t + 1}")  # numeracija nuo 1
        st.write("**Raktažodžiai:** " + ", ".join(topic_keywords.get(t, [])[:10]))

        sample = df[df["topic_id"] == t].head(10)
        st.write("**Pavyzdiniai straipsniai:**")
        for title in sample["title"].tolist():
            st.write("• " + title)

    # jei turim daugiau nei 5 temas, kas 5 perkeliam į naują eilę
    if (i + 1) % len(cols) == 0 and (i + 1) < len(topics) and current_topics > 5:
        cols = st.columns(min(current_topics - (i + 1), 5))