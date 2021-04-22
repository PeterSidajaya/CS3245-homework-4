from constants import *
from collections import Counter
from index_helper import get_word_list
from scoring import rank_document_ids
from query_util import get_query_term_vector

import math
import pickle

def free_text_search(query_list, dictionary, posting_file, tagged_prio_list, do_ranking=True):
    """
    Rank the list of document based on the query given.

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
            list(int): The list of doc_id's sorted by score
        else:
            list(int): List of doc_id's from the free text search
    """
    query_counter = Counter(query_list)
    query_keys = list(query_counter.keys())
    query_term_vector = get_query_term_vector(query_keys, query_counter, dictionary)

    # dictionary["LENGTH"] is the normalize denominator for a particular document_id which precomputed in index stage
    ranking_list = []
    potential_document_id = set()
    document_term_dict = {}

    # initialize the dictionary
    for term in query_keys:
        document_term_dict[term] = {}

    # calculate tf_score for each term (if it exists in the dictionary)
    for term in query_keys:
        tf_score = 0
        posting_list = get_word_list(term, dictionary, posting_file)

        for (doc_id, term_freq, _) in posting_list:
            tf_score = 1 + math.log(term_freq, 10)  # tf
            document_term_dict[term][doc_id] = tf_score / dictionary[DOCUMENT_LENGTH_KEYWORD][doc_id]  # normalize score
            potential_document_id.add(doc_id)

    # With ranking
    if (do_ranking):
        # Calculate score for each document
        for doc_id in potential_document_id:
            score = []

            # Iterate for each term score
            for i in range(len(query_keys)):
                term = query_keys[i]

                if (term not in document_term_dict or doc_id not in document_term_dict[term]):
                    score.append(0)
                else:
                    score.append(document_term_dict[term][doc_id] * query_term_vector[i])

            # Final score for document
            score = sum(score)
            ranking_list.append((doc_id, score))

        ranking_list = rank_document_ids(ranking_list, tagged_prio_list)
        return [x for x, y in ranking_list]

    # Without ranking
    else:
        return list(potential_document_id)
