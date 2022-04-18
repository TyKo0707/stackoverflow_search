import heapq
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

# Convert raw text data of tags into lists
de_duplicated_data.tags = de_duplicated_data.tags.apply(lambda x: x.split('|'))

# Make a dictionary to count the frequencies for all tags
tag_freq_dict = {}
for tags in de_duplicated_data.tags:
    for tag in tags:
        if tag not in tag_freq_dict:
            tag_freq_dict[tag] = 0
        else:
            tag_freq_dict[tag] += 1

most_common_tags = heapq.nlargest(100, tag_freq_dict, key=tag_freq_dict.get)

final_indices = []
for i, tags in enumerate(de_duplicated_data.tags.values.tolist()):
    # The minimum length for common tags should be 2
    if len(set(tags).intersection(set(most_common_tags))) > 1:
        final_indices.append(i)

final_data = de_duplicated_data.iloc[final_indices]
