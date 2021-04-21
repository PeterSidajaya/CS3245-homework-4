from nltk.corpus import wordnet as wn
from collections import defaultdict

import nltk
import regex

def lemmatize(token_list):
    """lemmatize every token in a given list of tokens
    """
    lmtzr = nltk.stem.WordNetLemmatizer()
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    return [lmtzr.lemmatize(token.lower(), tag_map[tag[0]]) for token, tag in nltk.pos_tag(token_list)]

def stem(token_list):
    """stem every token in a given list of tokens
    """
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(token.lower()) for token in token_list]

def sanitise(long_string):
    """Tokenize the a string text into a list of tokens and remove any non-alphanumeric characters, except for
    currency characters and numbers with punctuation (100,000.00). Use as a substitute for word_tokenize().

    Args:
        long_string (str): the text

    Returns:
        list: list of filtered tokens
    """
    word_list = nltk.tokenize.word_tokenize(long_string)
    sanitised_list = [sanitise_word(string) for string in word_list]
    second_tokenization_list = [nltk.tokenize.word_tokenize(string) for string in sanitised_list]
    return [token for token_list in second_tokenization_list for token in token_list]
    

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
