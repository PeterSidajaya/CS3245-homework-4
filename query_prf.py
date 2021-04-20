from constants import *
from collections import Counter
from index_helper import get_word_list
from free_text_query import normalize_list

import time
import heapq
import math
import pickle
import random

def prf_search(query_list, dictionary, posting_file, accepted_doc_id):
    """rank the list of document based on the query given, using prf

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
    doc_matrix = []

    # To get a faster quering, we precompute the value for tf_idf query vector
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

    # Retrieve the best documents with tf-idf based on this query vector
    ranking_list = get_results_for_vector(query_term_vector, query_keys, dictionary, posting_file)

    # These are the best docs that we will assume are relevant for PRF
    best_docs = ranking_list[:PRF_NUM_OF_RESULTS]
    set_of_best_doc_ids = set(map(lambda x: x[1], best_docs))

    # =======
    # Extend the query vector with words in the best docs.
    # This is tentative, we can just remove this and search in the titles.
    # Commenting this chunk out does not harm anything.
    print("Original query keys:", query_keys)
    extend_lists_as_much_as_possible(query_term_vector, query_keys, 
                                     set_of_best_doc_ids, PRF_TIME_LIMIT, dictionary, posting_file)
    
    print("Extended query keys:", query_keys)
    doc_matrix = get_tf_idf(set_of_best_doc_ids, query_keys, dictionary, posting_file)
    new_query_vec = get_mean_vector(doc_matrix, len(query_keys))
    new_query_vec = weighted_average(query_term_vector, PRF_QUERY_VEC_WEIGHT, 
                                     new_query_vec, PRF_QUERY_VEC_UPDATE_WEIGHT)
    # ====== OR
    # # Create a new query vector based on the best docs' vectors
    # new_query_vec = get_mean_vector(best_docs, len(query_term_vector))
    # new_query_vec = weighted_average(query_term_vector, PRF_QUERY_VEC_WEIGHT, 
    #                                  new_query_vec, PRF_QUERY_VEC_UPDATE_WEIGHT)
    # ======
    # Retrieve the best documents with tf-idf based on this modified query vector
    print("Original result:", list(map(lambda x: x[1], ranking_list[:15])))
    ranking_list = get_results_for_vector(new_query_vec, query_keys, dictionary, posting_file)
    print("New result:", list(map(lambda x: x[1], ranking_list[:15])))

    if (accepted_doc_id == None):
        tf_idf_doc_list = [y for x, y, z in ranking_list]
    else:
        tf_idf_doc_list = [y for x, y, z in ranking_list if y in accepted_doc_id]
        
    return tf_idf_doc_list

def extend_lists_as_much_as_possible(original_query_term_vector, query_keys, valid_doc_ids, time_limit, dictionary, posting_file):
    prev_time = time.perf_counter()
    total_time = 0
    no_of_words = len(dictionary)

    possible_words = list(dictionary.keys())
    num_checked = len(query_keys)
    # To add a hard limit for query vector size, uncomment this line instead
    # while total_time < time_limit and num_checked < no_of_words and len(query_keys) < 20:
    while total_time < time_limit and num_checked < no_of_words: # and len(query_keys) < 20:
        term = random.choice(possible_words)
        docs = get_word_list(term, dictionary, posting_file)
        for (doc_id, _, _) in docs:
            if doc_id in valid_doc_ids:
                original_query_term_vector.append(0)
                query_keys.append(term)
                break

        # Update timer
        curr_time = time.perf_counter()
        total_time += curr_time - prev_time
        prev_time = curr_time
        num_checked += 1

def get_mean_vector(best_docs, num_words):
    new_query_vec = []
    for i in range(num_words):
        term_value = 0
        for j in range(len(best_docs)):
            term_value += best_docs[j][2][i]
        term_value /= PRF_NUM_OF_RESULTS
        new_query_vec.append(term_value)
    return new_query_vec

def weighted_average(vector_1, weight_1, vector_2, weight_2):
    result_vec = []
    for i in range(len(vector_1)):
        result_vec.append(vector_1[i] * weight_1 + vector_2[i] * weight_2)
    return result_vec

def get_tf_idf(valid_doc_ids, query_keys, dictionary, posting_file):
    """Returns a document matrix, similar to get_results_for_vector but for valid doc ids.
    No dot product is done.
    """
    # dictionary["LENGTH"] is the normalize denominator for a particular document_id which precomputed in index stage
    tf_idf_list = []
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
    
    # calculate cosine score
    for doc_id in valid_doc_ids:
        document_term_vector = []
        doc_vec = []

        # calculate cosine score
        for i in range(len(query_keys)):
            term = query_keys[i]

            if (term not in document_term_dict or doc_id not in document_term_dict[term]):
                doc_vec.append(0)
            else:
                doc_vec.append(document_term_dict[term][doc_id])
        # We just need to make sure that doc_vec is in index 2 for backwards compatibility
        tf_idf_list.append((0, doc_id, doc_vec))

    return tf_idf_list

def get_results_for_vector(query_term_vector, query_keys, dictionary, posting_file):
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
        doc_vec = []

        # calculate cosine score
        for i in range(len(query_keys)):
            term = query_keys[i]

            if (term not in document_term_dict or doc_id not in document_term_dict[term]):
                doc_vec.append(0)
            else:
                doc_vec.append(document_term_dict[term][doc_id] * query_term_vector[i])

        # final cosine score for ranking
        score = sum(doc_vec)

        ranking_list.append((score, doc_id, doc_vec))

    ranking_list.sort(key=lambda x: x[0], reverse=True)
    return ranking_list
