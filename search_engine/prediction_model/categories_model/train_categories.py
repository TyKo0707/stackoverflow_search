from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from environs import Env

env = Env()
env.read_env()

DATA_PATH = env.str("DATA_PATH")

df = pd.read_csv(DATA_PATH + 'dec_dataset.csv', engine='pyarrow')[:20]

titles = df.title

categories_bin = pd.get_dummies(df['category'])

vectorizer_title = TfidfVectorizer(analyzer='word',
                                   min_df=0.0,
                                   max_df=1.0,
                                   strip_accents=None,
                                   encoding='utf-8',
                                   preprocessor=None,
                                   token_pattern=r"[a-zA-Z0-9_+#]+",
                                   max_features=1000)

title_tfidf = vectorizer_title.fit_transform(titles)
tfidf_tokens = vectorizer_title.get_feature_names()
df_tfidf = pd.DataFrame(data=title_tfidf.toarray(), index=[f'Doc{i}' for i in range(df.shape[0])],
                        columns=tfidf_tokens)

print(df_tfidf['c#'])
