from constants import *
from collections import Counter
from index_helper import get_word_list

import math
import pickle

def free_text_search(query_list, dictionary, posting_file, accepted_doc_id, do_ranking=True):
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

    query_length = 0
    query_term_vector = []

    no_of_document = len(dictionary[DOCUMENT_LENGTH_KEYWORD])

    # To get a faster quering, we precomute the value for tf_idf query vector
    # Next time, we only need to do dot product with each of the given document 
    for term in query_keys:
        tf_idf_score = 0

        if (term in dictionary):
            term_info = dictionary[term]
            term_df = term_info[0]

            tf_idf_score = (1 + math.log(query_counter[term], 10)) * math.log(no_of_document / term_df)
            query_length += (tf_idf_score ** 2)
        
        query_term_vector.append(tf_idf_score)

    normalize_denominator = math.sqrt(query_length)
    if (normalize_denominator != 0):
        # final precompute query vector
        query_term_vector = normalize_list(query_term_vector, normalize_denominator)

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
    
    # calculate cosine score
    for doc_id in potential_document_id:
        document_term_vector = []
        score = []

        # calculate cosine score
        for i in range(len(query_keys)):
            term = query_keys[i]

            if (term not in document_term_dict or doc_id not in document_term_dict[term]):
                score.append(0)
            else:
                score.append(document_term_dict[term][doc_id] * query_term_vector[i])

        # final cosine score for ranking
        score = sum(score)

        ranking_list.append((score, doc_id))

    if (do_ranking):
        ranking_list.sort(key=lambda x: x[0], reverse=True)

    if (accepted_doc_id == None):
        tf_idf_doc_list = [y for x, y in ranking_list]
    else:
        tf_idf_doc_list = [y for x, y in ranking_list if y in accepted_doc_id]
        
    return " ".join(str(doc_id) for doc_id in tf_idf_doc_list)


def normalize_list(lst, denominator):
    return list(map(lambda x: x/denominator, lst))