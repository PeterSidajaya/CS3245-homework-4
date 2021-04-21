from constants import *
from collections import Counter
from index_helper import get_word_list
from scoring import rank_document_ids, get_bm25_score

import math
import pickle

def free_text_search(query_list, dictionary, posting_file, tagged_prio_list, do_ranking=True):
    """rank the list of document based on the query given

    Args:
        query_list (list): the list of query string to be ranked against
        dictionary (dictionary): dictionary of the posting lists
        posting_file (str): address to the posting file list
        accepted_doc_id (set): set of valid doc_id from phrasal queries in the given query text

    Returns:
        str: search rank result
    """
    query_counter = Counter(query_list)
    query_keys = list(query_counter.keys())

    no_of_document = len(dictionary[DOCUMENT_LENGTH_KEYWORD])
    avg_dl = dictionary[DOCUMENT_AVG_LENGTH_KEYWORD]

    ranking_list = []
    potential_document_id = set()
    document_term_dict = {}
    n_of_doc_containing_term = {}

    # initialize the dictionary
    for term in query_keys:
        document_term_dict[term] = {}

    # calculate tf_score for each term (if it exists in the dictionary)
    for term in query_keys:
        tf_score = 0
        posting_list = get_word_list(term, dictionary, posting_file)
        n_of_doc_containing_term[term] = len(posting_list)

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
                    continue
                term_score = get_bm25_score(document_term_dict[term][doc_id], n_of_doc_containing_term[term], no_of_document, avg_dl)
                score.append(term_score)

            # Final score for the document
            score = sum(score)
            ranking_list.append((doc_id, score))

        ranking_list = rank_document_ids(ranking_list, tagged_prio_list)
        return [x for x, y in ranking_list]

    # Without ranking
    else:
        return potential_document_id


def normalize_list(lst, denominator):
    return list(map(lambda x: x/denominator, lst))
