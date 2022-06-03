import spacy
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, GRU
from keras.layers import BatchNormalization
from environs import Env
import numpy as np
from logger import get_logger
import keras.backend as K


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


logger = get_logger()

EN = spacy.load('en_core_web_sm')

env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
VOCAB_SIZE = env.int("VOCAB_SIZE")

W2V_SIZE = 300
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 300

X_train_padded = np.load(TRAIN_TEST_PATH + 'x_train_padded.npy', allow_pickle=True)
y_train = np.load(TRAIN_TEST_PATH + 'y_train.npy', allow_pickle=True)
X_test_padded = np.load(TRAIN_TEST_PATH + 'x_test_padded.npy', allow_pickle=True)
y_test = np.load(TRAIN_TEST_PATH + 'y_test.npy', allow_pickle=True)
embedding_matrix = np.load(TRAIN_TEST_PATH + 'embedding_matrix.npy', allow_pickle=True)

result_metrics = [
    BinaryAccuracy(name='accuracy'),
    Precision(name='precision'),
    Recall(name='recall'),
    f1_metric
]

model = Sequential()
model.add(
    Embedding(VOCAB_SIZE + 1, W2V_SIZE, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
model.add(GRU(300, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(400, activation='relu', kernel_initializer="he_normal"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(150, activation='relu'))
model.add(Dense(1000, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              metrics=result_metrics,
              optimizer="adam")
model.summary()


# Train Model

BATCH_SIZE = 1024
logger.info("Start fitting model")
history = model.fit(x=X_train_padded, y=y_train,
                    batch_size=BATCH_SIZE,
                    epochs=20,
                    validation_split=0.1,
                    verbose=2)

logger.info("End of fitting model")

model.save_weights("stack.h5")
model_json = model.to_json()
json_file = open("stack.json", "w")
json_file.write(model_json)
json_file.close()
