from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from environs import Env
from sklearn.model_selection import train_test_split

env = Env()
env.read_env()

DATA_PATH = env.str("DATA_PATH")

df_title = pd.read_csv(DATA_PATH + 'dec_dataset.csv', engine='pyarrow')
df_tags = pd.read_csv(DATA_PATH + 'enc_dataset.csv', engine='pyarrow')

titles = df_title.title


def split_tags(string):
    if string:
        return [int(i) for i in string.split('|')]


df_tags.tags = df_tags.tags.apply(split_tags)
df_tags.dropna(inplace=True, axis=0)
tags = df_tags.tags

y_title = pd.get_dummies(df_title['category'])
y_tags = pd.get_dummies(df_tags['category'])

vectorizer_title = TfidfVectorizer(analyzer='word',
                                   min_df=0.0,
                                   max_df=1.0,
                                   strip_accents=None,
                                   encoding='utf-8',
                                   preprocessor=None,
                                   token_pattern=r"[a-zA-Z0-9_+#]+",
                                   max_features=1000)

title_tfidf = vectorizer_title.fit_transform(titles)

X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(title_tfidf, y_title, test_size=0.2,
                                                                            random_state=0)

X_train_tags, X_test_tags, y_train_tags, y_test_tags = train_test_split(tags, y_tags, test_size=0.2,
                                                                        random_state=0)
