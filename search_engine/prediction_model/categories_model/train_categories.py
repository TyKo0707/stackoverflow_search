import pickle
import warnings
import gensim
from keras import Sequential
from keras.layers import Embedding, BatchNormalization, Dropout, Dense, GRU
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from logger import get_logger

env = Env()
env.read_env()
logger = get_logger()

warnings.filterwarnings('ignore')

DATA_PATH = env.str("DATA_PATH")
MODELS = env.str("MODELS")
VOCAB_SIZE = env.int("VOCAB_SIZE")
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")

with open(TRAIN_TEST_PATH + "tokenizer.txt", 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

w2v_model = gensim.models.word2vec.Word2Vec.load(MODELS + 'SO_word2vec_embeddings.bin')


def question_to_vec(question, embeddings, dim=300):
    question_embedding = np.zeros(dim)
    valid_words = 0
    for word in question.split(' '):
        if word in embeddings.wv.index_to_key:
            valid_words += 1
            question_embedding += embeddings.syn1neg[embeddings.wv.key_to_index[word]]
    if valid_words > 0:
        return question_embedding / valid_words
    else:
        return question_embedding


def split_tags(string):
    if string:
        return [int(i) for i in string.split('|')]


df_title = pd.read_csv(DATA_PATH + 'dec_dataset.csv', engine='pyarrow').sample(frac=1)
df_tags = pd.read_csv(DATA_PATH + 'enc_dataset.csv', engine='pyarrow').sample(frac=1)
df_tags.tags = df_tags.tags.apply(split_tags)
df_tags.dropna(inplace=True, axis=0)
tags = df_tags.tags

titles = df_title.title


def convert_titles_to_vec(df: pd.DataFrame, column_name: str):
    t_vectors = []
    for title in df[column_name].values:
        t_vector = question_to_vec(title, w2v_model)
        t_vectors.append(t_vector)
    return np.array(t_vectors)


y_title = pd.get_dummies(df_title['category'])
y_tags = pd.get_dummies(df_tags['category'])

titles_vectors = convert_titles_to_vec(df_title, 'title')

multilabel_binarizer = MultiLabelBinarizer()
y_bin = multilabel_binarizer.fit_transform(df_tags.tags)

X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(titles_vectors,
                                                                            df_title['category'],
                                                                            test_size=0.2,
                                                                            random_state=0)
X_train_tags, X_test_tags, y_train_tags, y_test_tags = train_test_split(y_bin, df_tags['category'], test_size=0.2,
                                                                        random_state=0)

# region title model
logreg_title = LogisticRegression(n_jobs=1, C=1e5)
logreg_title = logreg_title.fit(X_train_title, y_train_title)
y_pred = logreg_title.predict(X_test_title)
pickle.dump(logreg_title, open('model_title.pkl', 'wb'))
print('accuracy %s' % accuracy_score(y_pred, y_test_title))

print(classification_report(y_test_title, y_pred, target_names=logreg_title.classes_))
# endregion

# region tags model
logreg_tags = LogisticRegression(n_jobs=1, C=1e5)
logreg_tags = logreg_tags.fit(X_train_tags, y_train_tags)
y_pred = logreg_tags.predict(X_test_tags)
pickle.dump(logreg_tags, open('model_tags.pkl', 'wb'))
print('accuracy %s' % accuracy_score(y_pred, y_test_tags))

print(classification_report(y_test_tags, y_pred, target_names=logreg_tags.classes_))
# endregion
