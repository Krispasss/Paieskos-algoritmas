from pathlib import Path

p = Path("artifacts")
files = ["articles.parquet", "embeddings.npy", "kmeans.pkl", "topic_keywords.json", "meta.json"]

print("Artifacts folder exists:", p.exists())
for f in files:
    fp = p / f
    if fp.exists():
        print(f"{f}: OK, size={fp.stat().st_size} bytes")
    else:
        print(f"{f}: MISSING")