import pandas as pd

df = pd.read_parquet("data/dataset.parquet")
print(df.shape)
print(df.head(3))
print(df.columns)