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
import keras.backend as K
import nltk
from logger import get_logger
from environs import Env

logger = get_logger()
env = Env()
env.read_env()
FINAL_DATA = env.str("FINAL_DATA")
MODELS = env.str("MODELS")
MAX_SEQUENCE_LENGTH = 300
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
nltk.download('stopwords')

preprocessed_data = pd.read_csv(FINAL_DATA)
preprocessed_data.tags = preprocessed_data.tags.apply(lambda x: x.split('|'))
title_embeddings = np.load(TRAIN_TEST_PATH + 'embedding_matrix.npy')

# Import saved Word2vec Embeddings
w2v_model = gensim.models.word2vec.Word2Vec.load(MODELS + 'SO_word2vec_embeddings.bin')


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
    return tags_encoded


def predict_tags(text):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_SEQUENCE_LENGTH)
    # Predict
    prediction = model.predict([x_test])[0]
    for i, value in enumerate(prediction):
        if value > 0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0
    tags = tag_encoder.inverse_transform(np.array([prediction]))
    return tags


# Load model and other relevant stuff
tag_encoder = load_tag_encoder()

with open(TRAIN_TEST_PATH + "final_tags.txt", 'rb') as tokenizer_file:
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


# calculating the tfidf of the all the title
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=preprocessed_data.shape[0])
vectorizer.fit_transform(preprocessed_data['processed_title'].values)


def search_results(search_string, num_results):
    # preprocessing the input search string
    search_string = preprocess_text(search_string)
    search_vect = np.array([question_to_vec(search_string, w2v_model)])

    # Getting the predicted tags
    tags = list(predict_tags(search_string))
    tags = [item for t in tags for item in t]
    tags = set(tags)

    if len(tags) != 0:
        search_res = []
        mask = [True if len(tags.intersection(set(preprocessed_data.iloc[i].tags))) >= 2 else False
                for i in range(preprocessed_data.shape[0])]
        data_new = preprocessed_data[mask]
        data_new.reset_index(inplace=True, drop=True)
        all_title_embeddings = []

        # calculating the tfidf
        masked_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=data_new.shape[0])
        masked_vectorizer.fit_transform(data_new['processed_title'].values)

        # calculating the tfidf of the input string
        input_query = [search_string]
        search_string_tfidf = masked_vectorizer.transform(input_query)

        # getting the title embedding from word to vec model
        for title in data_new.processed_title:
            all_title_embeddings.append(question_to_vec(title, w2v_model))
        all_title_embeddings = np.array(all_title_embeddings)

        # calculating the cosine similarity
        cosine_similarities = pd.Series(cosine_similarity(search_vect, all_title_embeddings)[0])

        # adding additional scores like overall score, sentiment, search string tfidf to the cosine similarity
        cosine_similarities = cosine_similarities.add(
            (0.4 * data_new.overall_scores) + (0.1 * data_new.sentiment_polarity) + (0.1 * masked_vectorizer.idf_),
            fill_value=0)

        for i, j in cosine_similarities.nlargest(int(num_results)).iteritems():
            output = ''
            for t in data_new.iloc[i].question_content.split():
                if t.lower() in search_string:
                    output += " <b style='color: #464646'>" + str(t) + "</b>"
                else:
                    output += " " + str(t)
            temp = {
                'title': str(data_new.original_title[i]),
                'url': str(data_new.question_url[i]),
                'similarity_score': str(j)[:5],
                'votes': str(data_new.overall_scores[i]),
                'body': str(output),
                'tags': tags
            }
            search_res.append(temp)
        return search_res

    else:
        input_query = [search_string]
        all_title_embeddings = title_embeddings

        # calculating the tfidf of the input search query
        search_string_tfidf = vectorizer.transform(input_query)
        cosine_similarities = pd.Series(cosine_similarity(search_vect, all_title_embeddings)[0])
        cosine_similarities = cosine_similarities.add(
            (0.4 * preprocessed_data.overall_scores) + (0.1 * preprocessed_data.sentiment_polarity) +
            (0.2 * vectorizer.idf_), fill_value=0)

        search_res = []
        for i, j in cosine_similarities.nlargest(int(num_results)).iteritems():
            output = ''
            for t in preprocessed_data.iloc[i].question_content.split():
                if t.lower() in search_string:
                    output += " <b style='color: #464646'>" + str(t) + "</b>"
                else:
                    output += " " + str(t)
            temp = {
                'title': str(preprocessed_data.original_title[i]),
                'url': str(preprocessed_data.question_url[i]),
                'similarity_score': str(j)[:5],
                'votes': str(preprocessed_data.overall_scores[i]),
                'body': str(output),
                'tags': tags
            }
            search_res.append(temp)
        return search_res


if __name__ == '__main__':
    print(search_results('c++ public class', 2))
