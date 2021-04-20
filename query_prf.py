from constants import *
from collections import Counter
from index_helper import get_word_list
from free_text_query import normalize_list

import heapq
import math
import pickle

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

    ranking_list = get_results_for_vector(query_term_vector, query_keys, dictionary, posting_file)

    best_docs = ranking_list[:PRF_NUM_OF_RESULTS]
    new_query_vec = get_mean_vector(best_docs, len(query_term_vector))
    new_query_vec = weighted_average(query_term_vector, PRF_QUERY_VEC_WEIGHT, 
                                     new_query_vec, PRF_QUERY_VEC_UPDATE_WEIGHT)
    print(best_docs)
    print("Vector:",new_query_vec)

    if (accepted_doc_id == None):
        tf_idf_doc_list = [y for x, y, z in ranking_list]
    else:
        tf_idf_doc_list = [y for x, y, z in ranking_list if y in accepted_doc_id]
        
    return tf_idf_doc_list

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

def get_prf_clause(results):
    """
    Apply Pseudo Relevance Feedback(PRF) technique. PRF requires
    the query to be run at least once to get initial set of results.
    """
    # We only care about the top results
    top_results = results[:PRF_NUM_OF_RESULTS]

    # Look at the results document vector, and extract the impt words
    doc_vecs = list(map(lambda x: get_doc_vec(x), top_results))
    impt_words = map(lambda x: extract_k_impt_words(x, PRF_NUM_OF_WORDS_PER_DOC), doc_vecs)

    return ' '.join(impt_words)

############ HELPERS ############

def extract_k_impt_words(doc_vecs, k: int):
    """
    Extract k important words from the document vector. Important words
    are words with the highest tf-idf score within the document vector.
    """
    # Get the index of top tf-idf score
    heapq.heapify(doc_vecs)
    impt_words_index = heapq.nlargest(2, enumerate(doc_vecs), key=lambda x: x[1])

    # Convert the indices back to word
    return list(map(lambda x: get_word(x), impt_words_index))

def get_doc_vec(doc_id):
    """
    Get document vector based on its document-id
    """
    # Stubs
    return []

def get_word(doc_vec_id):
    """
    Get word based on the id of the document vector
    """
    # Stubs
    return ""
