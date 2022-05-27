import pickle

import gensim
from keras import Sequential
from keras.layers import Embedding, BatchNormalization, Dropout, Dense, GRU
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
embedding_matrix = np.load(TRAIN_TEST_PATH + 'embedding_matrix.npz', allow_pickle=True)
embedding_matrix = embedding_matrix.f.arr_0


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


# def predict_category(text):
#     # Tokenize text
#     x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_SEQUENCE_LENGTH)
#     # Predict
#     prediction = model.predict([x_test])[0]
#     for i, value in enumerate(prediction):
#         if value > 0.5:
#             prediction[i] = 1
#         else:
#             prediction[i] = 0
#     tags = tag_encoder.inverse_transform(np.array([prediction]))
#     return tags


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

# titles_vectors = convert_titles_to_vec(df_title, 'title')

y_title = pd.get_dummies(df_title['category'])
y_tags = pd.get_dummies(df_tags['category'])

X_train_title, X_test_title, y_train_title, y_test_title = train_test_split(np.array(df_title['title']), y_title, test_size=0.2,
                                                                            random_state=0)
X_train_tags, X_test_tags, y_train_tags, y_test_tags = train_test_split(tags, y_tags, test_size=0.2,
                                                                        random_state=0)

X_train_title_padded = tokenizer.texts_to_sequences(X_train_title)
X_train_title_padded = pad_sequences(X_train_title_padded, maxlen=MAX_SEQUENCE_LENGTH)


def avg_jacard(y_true, y_pred):
    jacard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(axis=1)

    return jacard.mean() * 100


def print_score(y_pred, y_test, clf):
    print("Clf: ", clf.__class__.__name__)
    print(f"Jacard score: {avg_jacard(y_test, y_pred)}")
    print(f"Hamming loss: {hamming_loss(y_pred, y_test) * 100}")
    print("---")


model = Sequential()
model.add(
    Embedding(VOCAB_SIZE + 1, W2V_SIZE, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
model.add(GRU(300, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(400, activation='relu', kernel_initializer="he_normal"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(150, activation='relu'))
model.add(Dense(34, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              metrics='accuracy',
              optimizer="adam")
model.summary()


# Train Model

BATCH_SIZE = 1024
logger.info("Start fitting model")
history = model.fit(x=X_train_title_padded, y=y_train_title,
                    batch_size=BATCH_SIZE,
                    epochs=50,
                    validation_split=0.1,
                    verbose=2)


logger.info("End of fitting model")
print("Evaluate model on test data")
results = model.evaluate(X_test_title, y_test_title, batch_size=128)
print("test loss, test acc:", results)

# model.save_weights("stack.h5")
# model_json = model.to_json()
# json_file = open("stack.json", "w")
# json_file.write(model_json)
# json_file.close()


# region huy
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
