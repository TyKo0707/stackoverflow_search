from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from prediction_model.tag_predictor.tag_predictor import preprocessed_data, final_tag_data
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from environs import Env
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from logger import get_logger

logger = get_logger(handle_errors=False)

env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
W2V_MODEL_PATH = env.str("W2V_MODEL_PATH")

W2V_SIZE = 300
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 300

tag_encoder = MultiLabelBinarizer()
tags_encoded = tag_encoder.fit_transform(final_tag_data)

data = pd.DataFrame(columns=['corpus_code_combined'])
data["corpus_code_combined"] = preprocessed_data.post_corpus

w2v_model = Word2Vec.load(W2V_MODEL_PATH)

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(np.array(data.corpus_code_combined),
                                                    tags_encoded, test_size=0.2, random_state=42)
print("TRAIN size:", len(X_train))
print("TEST size:", len(X_test))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_data.post_corpus)

word_index = tokenizer.word_index
vocab_size = len(word_index)
print(f'Found {len(word_index)} unique tokens.')

# Convert the data to padded sequences
X_train_padded = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_padded, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_train_padded.shape)

X_test_padded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_padded, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_test_padded.shape)

# Embedding matrix for the embedding layer
embedding_matrix = np.zeros((vocab_size + 1, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)

np.save(TRAIN_TEST_PATH + 'x_train.npy', X_train)
np.save(TRAIN_TEST_PATH + 'x_test.npy', X_test)
np.save(TRAIN_TEST_PATH + 'y_train.npy', y_train)
np.save(TRAIN_TEST_PATH + 'y_test.npy', y_test)
np.save(TRAIN_TEST_PATH + 'x_train_padded.npy', X_train_padded)
np.save(TRAIN_TEST_PATH + 'x_test_padded.npy', X_test_padded)

logger.info("All datasets were saved")
