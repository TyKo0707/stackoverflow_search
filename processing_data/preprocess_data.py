import pandas as pd

chunk = pd.read_csv('../collect_data/out.csv', chunksize=1000)
df = pd.concat(chunk)

# Check if dataframe contains NaN values
print(df.isna().sum())

# Drop NaN values
df = df.dropna(axis=0)

# Check if dataframe contains duplicated values
print(df.duplicated().any())

# Combining duplicate titles with one answer to one title with many answers (also combining their score)
grouped_data = df.groupby(['id', 'title', 'body', 'tags'], as_index=False)\
    .agg(combined_answers=('answers', lambda x: "\n".join(x)), combined_score=('score', 'sum'))
de_duplicated_data = pd.DataFrame(grouped_data)

print(de_duplicated_data.head(), de_duplicated_data.columns)
