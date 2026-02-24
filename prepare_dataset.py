import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/Data.json")   
OUTPUT_PATH = Path("data/dataset.parquet")


def to_text(x) -> str:
    """
    Saugiai paverčia bet kokį objektą į vieną tekstinę reikšmę.
    Sutvarko: None, NaN, list, numpy array, dict, string.
    """
    # None
    if x is None:
        return ""

    # numpy array
    if isinstance(x, np.ndarray):
        try:
            return ", ".join(map(str, x.tolist()))
        except Exception:
            return str(x)

    # list / tuple / set
    if isinstance(x, (list, tuple, set)):
        try:
            return ", ".join(map(str, x))
        except Exception:
            return str(x)

    # dict
    if isinstance(x, dict):
        return str(x)

    # pandas NaN (tik paprastiems skaliarams)
    try:
        if pd.isna(x):
            return ""
    except Exception:
        # kai pd.isna nepalaiko tipo (pvz. objektas), ignoruojam
        pass

    return str(x)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Neradau failo: {DATA_PATH.resolve()}")

    print("Loading JSON:", DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Converting to DataFrame...")
    # sample buvo kaip {"1": {...}, "2": {...}} -> orient="index"
    df = pd.DataFrame.from_dict(data, orient="index")

    print("Original columns:", df.columns.tolist())

    # Pervadinimai
    df = df.rename(columns={
        "Titile": "title",        # faile klaida "Titile"
        "Title": "title",         # jei yra "Title"
        "Description": "description",
        "Content": "content",
    })

    # Pasiliekam tik reikalingus + optional
    keep_cols = [c for c in ["title", "description", "content", "Topic", "Time"] if c in df.columns]
    df = df[keep_cols].copy()

    # Tekstiniai laukai
    for c in ["title", "description", "content"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # Probleminiai laukai -> viską paverčiam į string
    for c in ["Topic", "Time"]:
        if c in df.columns:
            df[c] = df[c].apply(to_text)

    # Sukuriam bendrą tekstą, kurį naudosim paieškai/keywords/klasteriams
    df["text"] = (
        df.get("title", "").astype(str).str.strip()
        + ". "
        + df.get("description", "").astype(str).str.strip()
        + " "
        + df.get("content", "").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Minimalus filtravimas, kad neliktų šiukšlių
    df = df[df["text"].str.len() >= 50].drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("Final shape:", df.shape)
    print("Final columns:", df.columns.tolist())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Saugom į parquet
    print("Saving to parquet:", OUTPUT_PATH)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("DONE ✔ Saved:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()