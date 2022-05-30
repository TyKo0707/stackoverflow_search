from environs import Env
from logger import get_logger
import pandas as pd
from keras.preprocessing.text import Tokenizer
import pickle

logger = get_logger(handle_errors=False)

env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
W2V_MODEL_PATH = env.str("W2V_MODEL_PATH")
FINAL_DATA = env.str("FINAL_DATA")

W2V_SIZE = 300
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 300

preprocessed_data = pd.read_csv(FINAL_DATA)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_data.post_corpus)

with open(TRAIN_TEST_PATH + "tokenizer.txt", 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

word_index = tokenizer.word_index
vocab_size = len(word_index)
print(f'Found {len(word_index)} unique tokens.')


if __name__ == '__main__':

    from sklearn.preprocessing import MultiLabelBinarizer
    import numpy as np
    from search_engine.prediction_model.tag_predictor.tag_predictor import final_tag_data
    from sklearn.model_selection import train_test_split
    from gensim.models import Word2Vec
    from keras.preprocessing.sequence import pad_sequences

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

    np.savez_compressed(TRAIN_TEST_PATH + 'x_train.npz', X_train)
    np.savez_compressed(TRAIN_TEST_PATH + 'x_test.npz', X_test)
    np.savez_compressed(TRAIN_TEST_PATH + 'y_train.npz', y_train)
    np.savez_compressed(TRAIN_TEST_PATH + 'y_test.npz', y_test)
    np.savez_compressed(TRAIN_TEST_PATH + 'x_train_padded.npz', X_train_padded)
    np.savez_compressed(TRAIN_TEST_PATH + 'x_test_padded.npz', X_test_padded)
    np.savez_compressed(TRAIN_TEST_PATH + 'embedding_matrix.npz', embedding_matrix)

    logger.info("All datasets were saved")
