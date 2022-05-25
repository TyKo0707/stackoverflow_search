import pandas as pd
from environs import Env

env = Env()
env.read_env()
DATA_PATH = env.str("DATA_PATH")

df1 = pd.read_parquet(f'{DATA_PATH}/raw_data1.gzip', engine="pyarrow")
df2 = pd.read_parquet(f'{DATA_PATH}/raw_data2.gzip', engine="pyarrow")

# Check if dataframes contains NaN values
print(df1.isna().sum(), df2.isna().sum())

# Drop NaN values
df1 = df1.dropna(axis=0)
df2 = df2.dropna(axis=0)

# Check if dataframes contains duplicated values
print(df1.duplicated().any(), df2.duplicated().any())

# Combining duplicate titles with one answer to one title with many answers (also combining their score)
de_duplicated_data1 = df1.groupby(['id', 'title', 'body', 'tags'], as_index=False) \
    .agg(combined_answers=('answers', lambda x: "\n".join(x)), combined_score=('score', 'sum'))
de_duplicated_data2 = df2.groupby(['id', 'title', 'body', 'tags'], as_index=False) \
    .agg(combined_answers=('answers', lambda x: "\n".join(x)), combined_score=('score', 'sum'))

de_duplicated_data1.to_parquet(f"{DATA_PATH}de_duplicated_data1.gzip", compression='gzip', index=False)
de_duplicated_data2.to_parquet(f"{DATA_PATH}de_duplicated_data2.gzip", compression='gzip', index=False)
