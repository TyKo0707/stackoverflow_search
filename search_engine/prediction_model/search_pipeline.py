import pandas as pd
from keras.metrics import Precision, Recall
import keras.losses
import numpy as np
from search_engine.processing_data.normalize_functions import preprocess_text
import gensim
import tensorflow as tf
from keras.models import model_from_json
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os
import keras.backend as K
import nltk
from logger import get_logger
from environs import Env

logger = get_logger()
env = Env()
env.read_env()
FINAL_DATA = env.str("FINAL_DATA")
MODELS = env.str("MODELS")
DATA_PATH = env.str("DATA_PATH")
CATEGORIES_DIRECTORY = DATA_PATH + 'dbc/'
MAX_SEQUENCE_LENGTH = 300
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
nltk.download('stopwords')

df_keys = pd.read_csv(DATA_PATH + 'tags_keys.csv', engine='pyarrow')
title_embeddings = np.load(TRAIN_TEST_PATH + 'embedding_matrix.npz', allow_pickle=True)
title_embeddings = title_embeddings.f.arr_0

# Import saved Word2vec Embeddings
w2v_model = gensim.models.word2vec.Word2Vec.load(MODELS + 'SO_word2vec_embeddings.bin')

model_tags = pickle.load(open(MODELS + 'model_tags.pkl', 'rb'))
mlb = pickle.load(open(MODELS + 'mlb.pkl', 'rb'))

dict_of_dfs = {}

for file in os.scandir(CATEGORIES_DIRECTORY):
    if file.is_file():
        df = pd.read_csv(file.path, engine='pyarrow')
        dict_of_dfs[file.name[:-4]] = df


def encode_tags(list_of_tags: list, keys: pd.DataFrame):
    new_l = []
    for k in list_of_tags:
        if k in keys.tag.values:
            new_l.append(keys[keys.tag == k].code.values[0])

    return new_l


def get_category_df(list_of_tags, num_of_tags):
    t = encode_tags(list_of_tags, df_keys)
    tags = mlb.transform([t, []])
    full_res = list(model_tags.predict_proba(tags)[0])
    if max(full_res) < 0.95:
        res = [model_tags.classes_[full_res.index(c)] for c in sorted(full_res)[-num_of_tags:]]
        return pd.concat([dict_of_dfs[i] for i in res])
    else:
        res = model_tags.classes_[full_res.index(max(full_res))]
        return dict_of_dfs[res]


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# Custom loss function to handle multilabel classification task (modified cross entropy)
def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # K.epsilon() = 1e-7
    # Multi-task loss
    return K.mean(K.sum(-y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))


def load_tag_encoder():
    with open(TRAIN_TEST_PATH + "final_tags.txt", "rb") as final_tag:  # Unpickling
        final_tag_data = pickle.load(final_tag)
    tag_encode = MultiLabelBinarizer()
    tags_encoded = tag_encode.fit_transform(final_tag_data)
    return tag_encode


def predict_tags(text, num_of_tags):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_SEQUENCE_LENGTH)
    # Predict
    prediction = model.predict([x_test])[0]
    pr_sort = np.sort(prediction)
    for i, value in enumerate(prediction):
        if value in pr_sort[-num_of_tags:]:
            prediction[i] = 1
        else:
            prediction[i] = 0
    tags = tag_encoder.inverse_transform(np.array([prediction]))
    return tags


# Load model and other relevant stuff
tag_encoder = load_tag_encoder()

with open(TRAIN_TEST_PATH + "tokenizer.txt", 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

keras.losses.multitask_loss = multitask_loss
graph = tf.compat.v1.get_default_graph()
with open(MODELS + 'stack.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json,
                            custom_objects={'f1': f1_metric, 'recall': Recall, 'precision': Precision})
# load weights into new model
model.load_weights(MODELS + "stack.h5")


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


def most_common(string, tags):
    tag_list = string.split('|')
    count = 0
    for i in tag_list:
        if i in tags:
            count += 1
    return count


def search_results(search_string, num_results):
    # preprocessing the input search string
    search_string = preprocess_text(search_string)
    search_vect = np.array([question_to_vec(search_string, w2v_model)])

    # Getting the predicted tags
    tags = list(predict_tags(search_string, 5))
    tags = [item for t in tags for item in t]
    preprocessed_data = get_category_df(tags, 3)
    tags = set(tags)

    search_res = []
    all_title_embeddings = []

    preprocessed_data['common_tags_num'] = preprocessed_data['tags'].apply(lambda x: most_common(x, tags))
    preprocessed_data.sort_values(by=['common_tags_num'], inplace=True)
    preprocessed_data = preprocessed_data[-500:]
    preprocessed_data.reset_index(inplace=True, drop=True)

    # calculating the tfidf
    masked_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=preprocessed_data.shape[0])
    masked_vectorizer.fit_transform(preprocessed_data['post_corpus'].values)

    # calculating the tfidf of the input string
    input_query = [search_string]
    search_string_tfidf = masked_vectorizer.transform(input_query)

    # getting the title embedding from word to vec model
    for title in preprocessed_data.post_corpus:
        title = preprocess_text(title)
        all_title_embeddings.append(question_to_vec(title, w2v_model))
    all_title_embeddings = np.array(all_title_embeddings)

    # calculating the cosine similarity
    cosine_similarities = pd.Series(cosine_similarity(search_vect, all_title_embeddings)[0])

    for i, j in cosine_similarities.nlargest(int(num_results)).iteritems():
        output = preprocessed_data.iloc[i].post_corpus
        temp = {
            'title': str(preprocessed_data.original_title[i]),
            'url': str(preprocessed_data.question_url[i]),
            'similarity_score': str(j)[:5],
            'votes': str(preprocessed_data.overall_scores[i]),
            'body': str(output),
            'tags': str(preprocessed_data.tags[i])
        }
        search_res.append(temp)
    return search_res


if __name__ == '__main__':
    print(search_results('error_log per virtual host one linux server running apache php 5 multiple virtual hosts '
                         'separate log files seem separate php virtual hosts set apache php log easiest way would', 2))
