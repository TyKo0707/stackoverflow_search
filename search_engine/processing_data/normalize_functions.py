import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')
EN = spacy.load('en_core_web_sm')


def tokenize_text(text):
    # Apply tokenization using spacy to docstrings.
    tokens = EN.tokenizer(text)
    return [token.text.lower() for token in tokens if not token.is_space]


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def normalize(words):
    words = to_lowercase(words)
    words = RegexpTokenizer(r"[a-zA-Z0-9._+#]+").tokenize(' '.join(words))
    words = remove_stopwords(words)
    return words


def tokenize_code(text):
    # A very basic procedure for tokenizing code strings.
    return RegexpTokenizer(r"[a-zA-Z0-9._+#]+").tokenize(text)


def preprocess_text(text):
    return ' '.join(normalize(tokenize_text(text)))
