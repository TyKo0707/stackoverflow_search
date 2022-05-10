import gensim
from search_engine.prediction_model.tag_predictor.tag_predictor import preprocessed_data
from environs import Env
import numpy as np
from pathlib import Path
from logger import get_logger

env = Env()
env.read_env()
MODELS_PATH = env.str("MODELS")
W2V_MODEL_PATH = env.str("W2V_MODEL_PATH")

my_file = Path(W2V_MODEL_PATH)

if not my_file.is_file():

    logger = get_logger(handle_errors=False)

    W2V_SIZE = 300
    W2V_WINDOW = 7
    W2V_EPOCH = 32
    W2V_MIN_COUNT = 10

    documents = [_text.split() for _text in np.array(preprocessed_data.post_corpus)]
    w2v_model = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE,
                                                window=W2V_WINDOW,
                                                min_count=W2V_MIN_COUNT,
                                                workers=8)

    w2v_model.build_vocab(documents)
    words = list(w2v_model.wv.key_to_index.keys())
    vocab_size = len(words)
    print("Vocab size", vocab_size)

    # Train Word Embeddings
    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
    w2v_model.save(MODELS_PATH + '/SO_word2vec_embeddings.bin')
    logger.info("END embedding created")
