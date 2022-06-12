import spacy
import heapq
import pandas as pd
from search_engine.processing_data.normalize_functions import preprocess_text
from environs import Env
from search_engine.processing_data.html_to_text import content_to_tokens

env = Env()
env.read_env()
DATA_PATH = env.str("DATA_PATH")

EN = spacy.load('en_core_web_sm')


def save_parts(dataframe, nd):
    for i in range(5, int(dataframe.shape[0] / 100000) + 1):
        if i == int(dataframe.shape[0] / 100000):
            df = dataframe[int(i * 10e4):int(dataframe.shape[0])]
        else:
            df = dataframe[int(i * 10e4):int((i + 1) * 10e4)]

        content_token_df = normalize_content(df)
        content_token_df.to_parquet(f"{DATA_PATH}/df{nd}/main_data_{i}.gzip", compression='gzip', index=False)


def normalize_content(dataframe):
    content_token_df = content_to_tokens(dataframe)

    # Convert raw text data of tags into lists
    content_token_df.tags = content_token_df.tags.apply(lambda x: x.split('|'))

    # Make a dictionary to count the frequencies for all tags
    tag_freq_dict = {}
    for tags in content_token_df.tags:
        for tag in tags:
            if tag not in tag_freq_dict:
                tag_freq_dict[tag] = 0
            else:
                tag_freq_dict[tag] += 1

    most_common_tags = heapq.nlargest(1250, tag_freq_dict, key=tag_freq_dict.get)

    final_indices = []
    for i, tags in enumerate(content_token_df.tags.values.tolist()):
        # The minimum length for common tags should be 2
        if len(set(tags).intersection(set(most_common_tags))) > 1:
            final_indices.append(i)

    final_data = content_token_df.iloc[final_indices]

    # Preprocess text for 'question_body', 'post_corpus' and a new column 'processed_title'
    final_data.question_content = final_data.question_content.apply(lambda x: preprocess_text(x))
    final_data.post_corpus = final_data.post_corpus.apply(lambda x: preprocess_text(x))
    final_data['processed_title'] = final_data.original_title.apply(lambda x: preprocess_text(x))

    # Normalize numeric data for the scores
    final_data['overall_scores'] = (final_data.overall_scores - final_data.overall_scores.mean()) / (
            final_data.overall_scores.max() - final_data.overall_scores.min())

    final_data.tags = final_data.tags.apply(lambda x: '|'.join(x))  # Combine the lists back into text data
    final_data = final_data.drop(['answers_content'], axis=1)

    return final_data


df1 = pd.read_parquet(f'{DATA_PATH}/de_duplicated_data1.gzip', engine="pyarrow")
save_parts(df1, 1)
