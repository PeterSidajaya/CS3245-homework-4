from constants import *
from collections import Counter
from index_helper import get_word_list
from free_text_query import normalize_list
from query_util import stem_clauses
from nltk import word_tokenize

import time
import heapq
import math
import pickle
import random

def prf_search(query_list, dictionary, posting_file, accepted_doc_id, stemmer, lemmatzr):
    """rank the list of document based on the query given, using prf.

    We first perform a tf-idf similar to that in free_text_search, but after ranking them, we take
    the first PRF_NUM_OF_RESULTS documents, and take all the words in their titles, and recompute a 
    new query vector that is a weighted average of the original query vector and the tf-idf of each 
    of the docs with the words in the titles, and perform another tf-idf on this extended query vector.

    Args:
        query_list (list): the list of query strings (stemmed/lemmatized) to be ranked against
        dictionary (dictionary): dictionary of the posting lists
        posting_file (str): address to the posting file list
        accepted_doc_id (set): set of valid doc_id from phrasal queries in the given query text
        stemmer: The stemmer to use on titles
        lemmatzr: The lemmatizer to use on titles
    Returns:
        str: search rank result
    """
    query_counter = Counter(query_list)
    query_keys = list(query_counter.keys())

    query_length = 0
    query_term_vector = []

    no_of_document = len(dictionary[DOCUMENT_LENGTH_KEYWORD])

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
    extend_lists_as_much_as_possible(query_term_vector, query_keys, 
                                     set_of_best_doc_ids, PRF_TIME_LIMIT, dictionary, posting_file)
    # Alternative is to extend with titles, but the titles are actually useless from our experiments
    # extend_query_by_title(query_term_vector, query_keys, set_of_best_doc_ids, 
    #                       dictionary, posting_file, stemmer, lemmatzr)
    doc_matrix = get_tf_idf(set_of_best_doc_ids, query_keys, dictionary, posting_file)
    new_query_vec = get_mean_vector(doc_matrix, len(query_keys))
    new_query_vec = weighted_average(query_term_vector, PRF_QUERY_VEC_WEIGHT, 
                                     new_query_vec, PRF_QUERY_VEC_UPDATE_WEIGHT)

    # Retrieve the best documents with tf-idf based on this modified query vector
    ranking_list = get_results_for_vector(new_query_vec, query_keys, dictionary, posting_file)

    if (accepted_doc_id == None):
        tf_idf_doc_list = [y for x, y, z in ranking_list]
    else:
        tf_idf_doc_list = [y for x, y, z in ranking_list if y in accepted_doc_id]
        
    return tf_idf_doc_list

def extend_query_by_title(original_query_term_vector, query_keys, valid_doc_ids, dictionary, 
    posting_file, stemmer, lemmatzr):
    """Extend query_keys by words in the titles of valid_doc_ids.

    This function will extend the original_query_term_vector with the same number of 0's as words
    were added to query_keys, whereby the words added are words in the titles of the valid_doc_ids.

    This is done until time_limit, since the number of words in the documents reaches infinity.

    Args:
        original_query_term_vector (list(float)) The query term vector
        query_keys (list(str)) The unique words in the query
        valid_doc_ids (set(int)) The documents to find the vectors for
        dictionary (dictionary): dictionary of the posting lists
        posting_file (str): address to the posting file list
        stemmer: Stemmer to use on title
        lemmatzr: Lemmatizer to use on title
    """
    present_keys = set(query_keys)
    for doc_id in valid_doc_ids:
        # Get the document's title and tokenize it
        title = dictionary[TITLE_KEYWORD][doc_id] 
        tokens = word_tokenize(title)
        # Stem word by word
        stemmed_tokens = []
        if (USE_LEMMATIZER):
            stemmed_tokens = list(map(lambda x: lemmtzr.lemmatize(x).lower(), tokens))
        if (USE_STEMMER):
            stemmed_tokens = list(map(lambda x: stemmer.stem(x).lower(), tokens))

        # Add all tokens in the title that were not already added, to the query vector
        for token in stemmed_tokens:
            if token not in present_keys and token in dictionary:
                # Prevent double adding words to the query vector
                present_keys.add(token)
                # Add it to the query vector. 
                # The query vector value is 0 because it was not in the actual query
                original_query_term_vector.append(0) 
                query_keys.append(token)

def extend_query_as_much_as_possible(original_query_term_vector, query_keys, valid_doc_ids, time_limit, dictionary, posting_file):
    """Extend query_keys by random words in valid_doc_ids.

    This function will extend the original_query_term_vector with the same number of 0's as words
    were added to query_keys, whereby the words added are randomly selected words of the valid_doc_ids.

    This is done until time_limit, since the number of words in the documents reaches infinity. The
    reason for this random method is that we cannot store an inverted index due to memory constraints.

    Args:
        original_query_term_vector (list(float)) The query term vector
        query_keys (list(str)) The unique words in the query
        valid_doc_ids (set(int)) The documents to find the vectors for
        time_limit (float) The amount of time that is allowed to be spent on extension.
        dictionary (dictionary): dictionary of the posting lists
        posting_file (str): address to the posting file list
    """
    prev_time = time.perf_counter()
    total_time = 0
    no_of_words = len(dictionary)

    possible_words = list(dictionary.keys())
    num_checked = len(query_keys)
    # To add a hard limit for query vector size, uncomment this line instead
    # while total_time < time_limit and num_checked < no_of_words and len(query_keys) < 20:
    while total_time < time_limit and num_checked < no_of_words: # and len(query_keys) < 20:
        term = random.choice(possible_words)
        if term == DOCUMENT_LENGTH_KEYWORD or term == TITLE_KEYWORD:
            num_checked += 1
            continue
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
    """From a document matrix, return the centroid.
    
    For best_docs, this is a list of triples, where triple[2] is the document vector.
    We return the mean of the document vectors.

    Args:
        best_docs (list((_, _, list(float))) The document matrix
    Returns:
        list(float) The centroid of the document vectors
    """
    new_query_vec = []
    for i in range(num_words):
        term_value = 0
        for j in range(len(best_docs)):
            term_value += best_docs[j][2][i]
        term_value /= PRF_NUM_OF_RESULTS
        new_query_vec.append(term_value)
    return new_query_vec

def weighted_average(vector_1, weight_1, vector_2, weight_2):
    """Create a new vector where v = v1*w1 + v2*w2.
    
    Args:
        vector_1 (list(float)) The first vector
        weight_1 (float) The weight given to the first vector
        vector_2 (list(float)) The second vector
        weight_2 (float) The weight given to the second vector
    Returns:
        list(float) The weighted average of the input vectors
    """
    result_vec = []
    for i in range(len(vector_1)):
        result_vec.append(vector_1[i] * weight_1 + vector_2[i] * weight_2)
    return result_vec

def get_tf_idf(valid_doc_ids, query_keys, dictionary, posting_file):
    """Returns a document matrix, similar to get_results_for_vector but for only the valid doc ids.
    
    Cosine scores are not computed, just raw tf-idf scores. This is a "light version" of 
    get_results_for_vector, so the output format is similar to it.

    The result is a list of document id's and their corresponding document vectors for
    the given words in the query_keys.

    Args:
        valid_doc_ids (set(int)) The documents to find the vectors for
        query_keys (list(str)) The unique words in the query
        dictionary (dictionary): dictionary of the posting lists
        posting_file (str): address to the posting file list
    Returns:
        list((0, doc_id, doc_vec)) List of document vectors tagged with doc_id (and dummy value)
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
    
    # Create the document vectors
    for doc_id in valid_doc_ids:
        document_term_vector = []
        doc_vec = []
        # Do it in order of the query_keys
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
    """Find the document vectors for the given query term vector.

    The result is a list of document id's and their corresponding document vectors for
    the given words in the query_keys, sorted by their cosine distance with query_term_vector.

    Args:
        query_term_vector (list(int)) The tf-idf of the query terms in query_keys
        query_keys (list(str)) The unique words in the query
        dictionary (dictionary): dictionary of the posting lists
        posting_file (str): address to the posting file list
    Returns:
        list((score, doc_id, doc_vec)) Sorted list of document vectors scored by cosine distance
                                       from query_term_vector.
    """
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
