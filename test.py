import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
from gensim.models.phrases import Phrases, Phraser
import spacy  # For preprocessing
import logging  # Setting up the loggings to monitor gensim
import multiprocessing
from gensim.models import Word2Vec

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)

df = pd.read_csv('simpsons_dataset.csv')
print(df.head())
print(df.isna().sum())

df = df.dropna().reset_index(drop=True)
print(df.head())
print(df.isna().sum())

nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])  # disabling Named Entity Recognition for speed


# def cleaning(doc):
#     # Lemmatizes and removes stopwords
#     # doc needs to be a spacy Doc object
#     txt = [token.lemma_ for token in doc if not token.is_stop]
#     # Word2Vec uses context words to learn the vector representation of a target word,
#     # if a sentence is only one or two words long,
#     # the benefit for the training is very small
#     if len(txt) > 2:
#         return ' '.join(txt)
#
#
# brief_cleaning = [re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words']]
#
# t = time()
#
# txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
#
# print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
#
# df_clean = pd.DataFrame({'clean': txt})
# df_clean = df_clean.dropna().drop_duplicates()
# print(df_clean.shape)
#
# sent = [row.split() for row in df_clean['clean']]
#
# phrases = Phrases(sent, min_count=30, progress_per=10000)
# bigram = Phraser(phrases)
#
# sentences = bigram[sent]
#
# word_freq = defaultdict(int)
# for sent in sentences:
#     for i in sent:
#         word_freq[i] += 1
#
# print(len(word_freq))
# print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])
#
# cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
# print(f"Number of cores in computer: {cores}")

# w2v_model = Word2Vec(min_count=20,
#                      window=2,
#                      vector_size=300,
#                      sample=6e-5,
#                      alpha=0.03,
#                      min_alpha=0.0007,
#                      negative=20,
#                      workers=cores-1)
#
# t = time()
#
# w2v_model.build_vocab(sentences, progress_per=10000)
#
# print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
#
# t = time()
#
# w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
# w2v_model.save("w2v.model")
#
# print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# Most similar words
w2v_model = Word2Vec.load("w2v.model")
print(w2v_model.wv.most_similar(positive=["homer"]))
print(w2v_model.wv.most_similar(positive=["marge"]))
print(w2v_model.wv.most_similar(positive=["bart"]))
print(w2v_model.wv.most_similar(positive=["lisa"]))

# Similarities
print(w2v_model.wv.similarity("moe", 'tavern'))
print(w2v_model.wv.similarity('maggie', 'baby'))

# Odd one out
print(w2v_model.wv.doesnt_match(['jimbo', 'milhouse', 'kearney']))
print(w2v_model.wv.doesnt_match(["nelson", "bart", "milhouse"]))
print(w2v_model.wv.doesnt_match(['homer', 'patty', 'selma']))
