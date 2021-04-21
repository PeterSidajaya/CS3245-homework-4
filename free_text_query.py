from constants import *
from collections import Counter
from index_helper import get_word_list
from scoring import rank_document_ids, get_results_for_vector
from query_util import get_query_term_vector

import math
import pickle

def free_text_search(query_list, dictionary, posting_file, tagged_prio_list, do_ranking=True):
    """rank the list of document based on the query given.

    query_list is the list of sanitized tokens in the input query clause. We compute a query
    vector from this query, and retrieve the document vectors for the documents in the index
    with relation to this query_list, and if do_ranking is true, we sort the document vectors
    by cosine score with the query vector, otherwise we just return all documents where
    the at least one word in the query_list appears.

    Args:
        query_list (list): the list of query string to be ranked against
        dictionary (dictionary): dictionary of the posting lists
        posting_file (str): address to the posting file list
        tagged_prio_list (set): set of valid doc_id from phrasal queries in the given query text
        do_ranking (bool): Whether ranking should be performed, or an unsorted list is sufficient
    Returns:
        if do_ranking:
            list(int), list(float) The list of doc_id's sorted by score, and query_term_vector
        else:
            list(int): List of doc_id's sorted by score
    """
    query_counter = Counter(query_list)
    query_keys = list(query_counter.keys())
    query_term_vector = get_query_term_vector(query_keys, query_counter, dictionary)

    results = get_results_for_vector(query_term_vector, query_keys, dictionary, posting_file, tagged_prio_list, do_ranking)

    if do_ranking:
        # Return query_term_vector alongside results, can be used for other techniques e.g. PRF
        return results, query_term_vector
    else:
        return results
