import gensim
from tag_predictor import preprocessed_data
from environs import Env

env = Env()
env.read_env()
MODELS_PATH = env.str("MODELS")

# WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# Train Word Embeddings and save
# Collect corpus for training word embeddings

documents = [_text.split() for _text in np.array(preprocessed_data.post_corpus)]
w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 
                                            window=W2V_WINDOW, 
                                            min_count=W2V_MIN_COUNT, 
                                            workers=8)
w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)

# Train Word Embeddings
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
w2v_model.save(MODELS_PATH + '/SO_word2vec_embeddings.bin')
print("END embedding created")
