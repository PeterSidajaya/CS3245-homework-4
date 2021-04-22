from nltk.corpus import wordnet as wn
from collections import defaultdict
from constants import *
from nltk.corpus import stopwords

import nltk
import regex

nltk.download('stopwords')

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


def lemmatize(token_list):
    """Lemmatize every token in a given list of tokens

    Args:
        token_list (list(str)): List of tokens
    Returns:
        list(str): lemmatized list of tokens
    """
    return [lemmatizer.lemmatize(token.lower(), tag_map[tag[0]]) for token, tag in nltk.pos_tag(token_list)]

def stem(token_list):
    """Stem every token in a given list of tokens
    
    Args:
        token_list (list(str)): List of tokens
    Returns:
        list(str): stemmed list of tokens
    """
    return [stemmer.stem(token.lower()) for token in token_list]

def lemmatize_and_stem(token_list):
    """Lemmatize then stem every token in a given list of tokens.
    
    Args:
        token_list (list(str)): List of tokens
    Returns:
        list(str): lemmatized and stemmed list of tokens
    """
    return [stemmer.stem(lemmatizer.lemmatize(token.lower(), tag_map[tag[0]])) for token, tag in nltk.pos_tag(token_list)]

def sanitise(long_string):
    """Tokenize and sanitize a long string text.
    
    Tokenize the string text into a list of tokens and remove any non-alphanumeric 
    characters and stop words, except for currency characters and numbers with 
    punctuation (100,000.00). Use as a substitute for word_tokenize().

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
        removed_list = [token for token in token_list if token not in stopwords.words('english')]
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
    """Remove any non-alphanumeric characters,
    
    Args:
        string (str): The string to sanitize
    Returns:
        str: The string with non-alphanumeric characters removed
    """
    if not is_numeric(string):
        return regex.sub(r'[^a-zA-Z0-9\_\-\p{Sc}]', ' ', string)
    else:
        return string

def is_numeric(string):
    """Check if a string is numeric.

    Args:
        string (str): The string to check.
    Returns:
        True if the string is purely numeric, false otherwise.
    """
    if regex.match(r'[0-9]+[^0-9][0-9]+', string):
        return True
    else:
        return False
