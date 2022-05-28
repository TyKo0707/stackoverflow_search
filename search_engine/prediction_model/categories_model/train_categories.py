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

W2V_SIZE = 300
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 300

with open(TRAIN_TEST_PATH + "tokenizer.txt", 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

w2v_model = gensim.models.word2vec.Word2Vec.load(MODELS + 'SO_word2vec_embeddings.bin')


# embedding_matrix = np.load(TRAIN_TEST_PATH + 'embedding_matrix.npz', allow_pickle=True)
# embedding_matrix = embedding_matrix.f.arr_0


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


df_title = pd.read_csv(DATA_PATH + 'dec_dataset.csv', engine='pyarrow').sample(frac=1)
df_tags = pd.read_csv(DATA_PATH + 'enc_dataset.csv', engine='pyarrow').sample(frac=1)[:2000]

titles = df_title.title


def convert_titles_to_vec(df: pd.DataFrame, column_name: str):
    t_vectors = []
    for title in df[column_name].values:
        t_vector = question_to_vec(title, w2v_model)
        t_vectors.append(t_vector)
    return np.array(t_vectors)


def split_tags(string):
    if string:
        return [int(i) for i in string.split('|')]


df_tags.tags = df_tags.tags.apply(split_tags)
df_tags.dropna(inplace=True, axis=0)
tags = df_tags.tags

y_title = pd.get_dummies(df_title['category'])
y_tags = pd.get_dummies(df_tags['category'])

c_t_d = ['c#', 'c++', '.net', 'c']


def in_ctd(s):
    if s not in c_t_d:
        return True
    else:
        return False


mask = df_title['category'].apply(in_ctd)

df_title = df_title[mask]

titles_vectors = convert_titles_to_vec(df_title, 'title')

X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(titles_vectors,
                                                                            df_title['category'],
                                                                            test_size=0.2,
                                                                            random_state=0)
# X_train_tags, X_test_tags, y_train_tags, y_test_tags = train_test_split(tags, y_tags, test_size=0.2,
#                                                                         random_state=0)

# region model
# X_train_title_padded = tokenizer.texts_to_sequences(X_train_title)
# X_train_title_padded = pad_sequences(X_train_title_padded, maxlen=MAX_SEQUENCE_LENGTH)
# X_test_title_padded = tokenizer.texts_to_sequences(X_test_title)
# X_test_title_padded = pad_sequences(X_test_title_padded, maxlen=MAX_SEQUENCE_LENGTH)
#
# model = Sequential()
# model.add(
#     Embedding(VOCAB_SIZE + 1, W2V_SIZE, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
# model.add(GRU(300, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(400, activation='relu', kernel_initializer="he_normal"))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# model.add(Dense(150, activation='relu'))
# model.add(Dense(34, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               metrics='accuracy',
#               optimizer="adam")
# model.summary()
#
# # Train Model
# BATCH_SIZE = 256
# logger.info("Start fitting model")
# history = model.fit(x=X_train_title_padded, y=y_train_title,
#                     batch_size=BATCH_SIZE,
#                     epochs=8,
#                     validation_split=0.1,
#                     verbose=2)
#
# logger.info("End of fitting model")
# print("Evaluate model on test data")
# pickle.dump(model, open('model.pkl', 'wb'))
# model.save_weights("titles.h5")
# model_json = model.to_json()
# json_file = open("titles.json", "w")
# json_file.write(model_json)
# json_file.close()
# results = model.evaluate(X_test_title_padded, y_test_title, batch_size=128)
# print("test loss, test acc:", results)
# endregion

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(X_train_title, y_train_title)
y_pred = logreg.predict(X_test_title)
print('accuracy %s' % accuracy_score(y_pred, y_test_title))

print(classification_report(y_test_title, y_pred, target_names=logreg.classes_))
# region base
# sgd = SGDClassifier()
# lr = LogisticRegression()
# mn = MultinomialNB()
# svc = LinearSVC()
# perceptron = Perceptron()
# pac = PassiveAggressiveClassifier()
#
# for classifier in [sgd, lr, mn, svc, perceptron, pac]:
#     clf = OneVsRestClassifier(classifier)
#     clf.fit(X_train_title, y_train_title)
#     y_pred = clf.predict(X_test_title)
#     print_score(y_pred, y_test_title, classifier)
# endregion

# region load_model
# with open('titles.json', 'r') as json_file:
#     loaded_model_json = json_file.read()
#     model = model_from_json(loaded_model_json)
#
# # load weights into new model
# model.load_weights("titles.h5")
#
# model.compile(loss='binary_crossentropy',
#               metrics='accuracy',
#               optimizer="adam")
# score = model.evaluate(X_test_title_padded, y_test_title, batch_size=128, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
# endregion
