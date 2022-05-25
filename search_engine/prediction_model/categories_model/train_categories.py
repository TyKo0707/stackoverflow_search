from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from environs import Env
from sklearn.linear_model import SGDClassifier, LogisticRegression, PassiveAggressiveClassifier, Perceptron
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

env = Env()
env.read_env()

DATA_PATH = env.str("DATA_PATH")

df_title = pd.read_csv(DATA_PATH + 'dec_dataset.csv', engine='pyarrow').sample(frac=1)
df_tags = pd.read_csv(DATA_PATH + 'enc_dataset.csv', engine='pyarrow').sample(frac=1)[:2000]

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


def avg_jacard(y_true, y_pred):
    jacard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(axis=1)

    return jacard.mean() * 100


def print_score(y_pred, y_test, clf):
    print("Clf: ", clf.__class__.__name__)
    print(f"Jacard score: {avg_jacard(y_test, y_pred)}")
    print(f"Hamming loss: {hamming_loss(y_pred, y_test) * 100}")
    print("---")


dummy = DummyClassifier()
sgd = SGDClassifier()
lr = LogisticRegression()
mn = MultinomialNB()
svc = LinearSVC()
perceptron = Perceptron()
pac = PassiveAggressiveClassifier()

for classifier in [dummy, sgd, lr, mn, svc, perceptron, pac]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train_title, y_train_title)
    y_pred = clf.predict(X_test_title)
    print_score(y_pred, y_test_title, classifier)

# Clf:  DummyClassifier
# Jacard score: 0.0
# Hamming loss: 3.0303030303030303
# ---
# Clf:  SGDClassifier
# Jacard score: 46.9462324474752
# Hamming loss: 1.8584287846922478
# ---
# Clf:  LogisticRegression
# Jacard score: 46.73060990376649
# Hamming loss: 1.858042537412877
# ---
# Clf:  MultinomialNB
# Jacard score: 24.982474029698555
# Hamming loss: 2.3796694882030422
# ---
# Clf:  LinearSVC
# Jacard score: 50.09240966158944
# Hamming loss: 1.7840761834133831
# ---
# Clf:  Perceptron
# Jacard score: 46.54812737662779
# Hamming loss: 2.841814357970116
# ---
# Clf:  PassiveAggressiveClassifier
# Jacard score: 50.76689397319058
# Hamming loss: 1.9534456154174464
