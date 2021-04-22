from constants import *
from collections import Counter
from index_helper import get_word_list
from query_util import stem_clauses, normalize_list, QueryType, tag_results, get_avg_score
from nltk import word_tokenize

import time
import heapq
import math
import pickle
import random

def get_term_idf(term, dictionary, no_of_documents):
    if term not in dictionary:
        return 0

    term_freq, _ = dictionary[term]

    # IDF formula is from Okapi BM25 IDF
    idf_num = no_of_documents - term_freq + 0.5
    idf_denom = term_freq + 0.5
    idf = math.log((idf_num / idf_denom) + 1)
    return idf

def prf_impt_words(ranked_list, query, dictionary):
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

# def prf_search(ranked_list, query_keys, query_term_vector, dictionary, posting_file):
#     """Rank the list of document based on the query given, using prf.

#     We look at the top K documents and extend the query vector with the most common words of
#     each of them. We then compute the document vectors for each of our top K vectors corresponding
#     to this extended query vector, and compute the centroid. We make a new query vector from a
#     weighted average of the original query vector with this centroid.

#     We then perform a final search with the original top K documents as our priority list, with this
#     new query vector.

#     Args:
#         ranked_list_with_scores (list(doc_id, score)): The result of an innitial free text query (ranked)
#         query_keys (list(str)): The list of tokens in the query
#         query_term_vector (list(float)): The query vector with tf-idf scores corresponding to query_keys
#         dictionary (dictionary): dictionary of the posting lists
#         posting_file (str): address to the posting file list
#     Returns:
#         list(int): list of document id's, sorted by search rank result
#     """
#     # These are the best docs that we will assume are relevant for PRF
#     best_docs = ranked_list[:PRF_NUM_OF_RESULTS]
#     set_of_best_doc_ids = set(best_docs)

#     # Extend the query vector with most common words in the best docs.
#     extend_query_with_impt_words(query_term_vector, query_keys,
#                                      set_of_best_doc_ids, dictionary)

#     doc_matrix = get_tf_idf(set_of_best_doc_ids, query_keys, dictionary, posting_file)
#     new_query_vec = get_mean_vector(doc_matrix, len(query_keys))
#     new_query_vec = weighted_average(query_term_vector, PRF_QUERY_VEC_WEIGHT,
#                                      new_query_vec, PRF_QUERY_VEC_UPDATE_WEIGHT)

#     # Retrieve the best documents with tf-idf based on this modified query vector
#     tagged_best_docs = tag_results(best_docs, QueryType.FREE_TEXT)
#     prf_list = get_results_for_vector(new_query_vec, query_keys, dictionary, posting_file, tagged_best_docs, True)

#     return prf_list

# def extend_query_with_impt_words(original_query_term_vector, query_keys, valid_doc_ids, dictionary):
#     """Extend query_keys by the most common words in valid_doc_ids.

#     This function will extend the original_query_term_vector with the same number of 0's as words
#     were added to query_keys, whereby the words added are the most common words in the valid_doc_ids.
#     Note that this requires that the documents had stop words removed during indexing.

#     Args:
#         original_query_term_vector (list(float)) The query term vector
#         query_keys (list(str)) The unique words in the query
#         valid_doc_ids (set(int)) The documents to find the vectors for
#         dictionary (dictionary): dictionary of the posting lists
#     """
#     # Use this to prevent adding duplicate words
#     present_keys = set(query_keys)
#     for doc_id in valid_doc_ids:
#         impt_words = dictionary[IMPT_KEYWORD][doc_id]
#         # Add all tokens in the important keywords that were not already added, to the query vector
#         for token in impt_words:
#             if token not in present_keys and token in dictionary:
#                 # Prevent double adding words to the query vector
#                 present_keys.add(token)
#                 # Add it to the query vector.
#                 # The query vector value is 0 because it was not in the actual query
#                 original_query_term_vector.append(0)
#                 query_keys.append(token)

# def get_mean_vector(best_docs, num_words):
#     """From a document matrix, return the centroid.

#     best_docs is a list<triple>, where triple[2] is the document vector.
#     We return the mean of the document vectors.

#     Args:
#         best_docs (list((_, _, list(float))): The document matrix - a list of tuples containing document vectors
#         num_words (int): The length of each document vector
#     Returns:
#         list(float) The centroid of the document vectors
#     """
#     new_query_vec = []
#     for i in range(num_words):
#         term_value = 0
#         for j in range(len(best_docs)):
#             term_value += best_docs[j][2][i]
#         term_value /= PRF_NUM_OF_RESULTS
#         new_query_vec.append(term_value)
#     return new_query_vec

# def weighted_average(vector_1, weight_1, vector_2, weight_2):
#     """Create a new vector where v = v1*w1 + v2*w2.

#     Args:
#         vector_1 (list(float)) The first vector
#         weight_1 (float) The weight given to the first vector
#         vector_2 (list(float)) The second vector
#         weight_2 (float) The weight given to the second vector
#     Returns:
#         list(float) The weighted average of the input vectors
#     """
#     result_vec = []
#     for i in range(len(vector_1)):
#         result_vec.append(vector_1[i] * weight_1 + vector_2[i] * weight_2)
#     return result_vec

# def get_tf_idf(valid_doc_ids, query_keys, dictionary, posting_file):
#     """Returns a document matrix, similar to get_results_for_vector but for only the valid doc ids.

#     Cosine scores are not computed, just raw tf-idf scores. This is a "light version" of
#     get_results_for_vector, so the output format is similar to it for compatibility.

#     The result is a list of document id's and their corresponding document vectors for
#     the given words in the query_keys.

#     Args:
#         valid_doc_ids (set(int)) The documents to find the vectors for
#         query_keys (list(str)) The unique words in the query
#         dictionary (dictionary): dictionary of the posting lists
#         posting_file (str): address to the posting file list
#     Returns:
#         list((0, doc_id, doc_vec)) List of document vectors tagged with doc_id (and dummy value)
#     """
#     # dictionary["LENGTH"] is the normalize denominator for a particular document_id which precomputed in index stage
#     tf_idf_list = []
#     document_term_dict = {}

#     # initialize the dictionary
#     for term in query_keys:
#         document_term_dict[term] = {}

#     # calculate tf_score for each term (if it exists in the dictionary)
#     for term in query_keys:
#         tf_score = 0
#         posting_list = get_word_list(term, dictionary, posting_file)

#         for (doc_id, term_freq, _) in posting_list:
#             tf_score = 1 + math.log(term_freq, 10)  # tf
#             document_term_dict[term][doc_id] = tf_score / dictionary[DOCUMENT_LENGTH_KEYWORD][doc_id]  # normalize score

#     # Create the document vectors
#     for doc_id in valid_doc_ids:
#         document_term_vector = []
#         doc_vec = []
#         # Do it in order of the query_keys
#         for i in range(len(query_keys)):
#             term = query_keys[i]
#             if (term not in document_term_dict or doc_id not in document_term_dict[term]):
#                 doc_vec.append(0)
#             else:
#                 doc_vec.append(document_term_dict[term][doc_id])
#         # We just need to make sure that doc_vec is in index 2 for backwards compatibility
#         tf_idf_list.append((0, doc_id, doc_vec))

#     return tf_idf_list

