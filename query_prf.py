from constants import *
from collections import Counter
from index_helper import get_word_list
from query_util import stem_clauses, normalize_list, QueryType, tag_results, get_avg_score
from nltk import word_tokenize

import math

def get_term_idf(term, dictionary, no_of_documents):
    """
    Given a term, calculate the idf score of the term.
    """
    if term not in dictionary:
        return 0

    term_freq, _ = dictionary[term]

    # IDF formula is from Okapi BM25 IDF
    idf_num = no_of_documents - term_freq + 0.5
    idf_denom = term_freq + 0.5
    idf = math.log((idf_num / idf_denom) + 1)
    return idf

def prf_impt_words(ranked_list, dictionary):
    """
    Extract important keywords from the documents ranked at the top of
    ranked_list.

    The candidate of important keywords are extracted at indexing
    stage, where we define important keywords to be the most frequent
    keywords (excluding stopwords). Then, the candidate are assigned
    idf_score. The words with highest idf are selected as the final
    output.

    Returns list of words.
    """
    # Only take the top ranked results
    best_docs = ranked_list[:PRF_NUM_OF_RESULTS]
    no_of_documents = len(dictionary[DOCUMENT_LENGTH_KEYWORD])

    # Get all saved important words
    # Important words are already sanitised stemmed and/or lemma during indexing
    impt_words = set()
    for doc_id in best_docs:
        impt_words.update(dictionary[IMPT_KEYWORD][doc_id])

    # Get top important words across all documents
    impt_words_with_idf = list(map(lambda x: (x, get_term_idf(x, dictionary, no_of_documents)), impt_words))
    impt_words_with_idf.sort(key=lambda x: x[1])

    flat_top_impt_words = impt_words_with_idf[:PRF_NUM_OF_RESULTS]
    avg_idf_score_words = get_avg_score(flat_top_impt_words)

    # Filter out weaker results
    new_query_words = filter(lambda x: x[1] > avg_idf_score_words, flat_top_impt_words)
    new_query_words = map(lambda x: x[0], new_query_words)

    return new_query_words
