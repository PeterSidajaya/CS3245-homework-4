from nltk.corpus import wordnet as wn
from collections import defaultdict
from constants import *
from nltk.corpus import stopwords
from flashtext import KeywordProcessor

import nltk
import re
import regex

nltk.download('stopwords')

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

keyword_processor = KeywordProcessor()
for word in stopwords.words('english'):
    keyword_processor.add_keyword(word, '__EMPTY__')


def flashtext_replacement(text):
    text = keyword_processor.replace_keywords(text)
    pattern = r"__EMPTY__\s?"
    text = re.sub(pattern, '', text)

    return text.strip()


def lemmatize(token_list):
    """lemmatize every token in a given list of tokens
    """
    return [lemmatizer.lemmatize(token.lower(), tag_map[tag[0]]) for token, tag in nltk.pos_tag(token_list)]

def stem(token_list):
    """stem every token in a given list of tokens
    """
    return [stemmer.stem(token.lower()) for token in token_list]

def lemmatize_and_stem(token_list):
    """combine both of the functions above. idk, in case this is faster
    """
    return [stemmer.stem(lemmatizer.lemmatize(token.lower(), tag_map[tag[0]])) for token, tag in nltk.pos_tag(token_list)]

def sanitise(long_string):
    """Tokenize the a string text into a list of tokens and remove any non-alphanumeric characters and stop words, except for
    currency characters and numbers with punctuation (100,000.00). Use as a substitute for word_tokenize().

    Args:
        long_string (str): the text

    Returns:
        list: list of filtered tokens
    """
    # Tokenize
    word_list = nltk.tokenize.word_tokenize(long_string)

    # Remove non-alphanumeric characters
    sanitised_list = [sanitise_word(string) for string in word_list]
    token_list = " ".join(sanitised_list).split()

    # Remove stop words
    if (REMOVE_STOPWORDS):
        removed_list = flashtext_replacement(keyword_processor.replace_keywords(" ".join(token_list))).split()
        token_list = removed_list

    # Apply stemming and/or lemmatization
    if (USE_LEMMATIZER and USE_STEMMER):
        token_list = lemmatize_and_stem(token_list)
    else:
        # This line is if you want to do lemmatization (prefer to do this before stemming, as stemming might not return a real word)
        if (USE_LEMMATIZER):
            token_list = lemmatize(token_list)

        # This line is if you want to do stemming after or instead
        if (USE_STEMMER):
            token_list = stem(token_list)

    return token_list


def sanitise_word(string):
    if not is_numeric(string):
        return regex.sub(r'[^a-zA-Z0-9\_\-\p{Sc}]', ' ', string)
    else:
        return string


def is_numeric(string):
    if regex.match(r'[0-9]+[^0-9][0-9]+', string):
        return True
    else:
        return False
