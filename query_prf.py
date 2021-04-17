from constants import *

import heapq

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