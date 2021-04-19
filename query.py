from collections import Counter
from phrasal_query import get_phrasal_query_doc_id
from query_util import categorise_query, QueryType
import nltk
import pickle
import math

# this function will be used to run if there is phrasal query
def intersect_document_ids(doc_list):
    if (len(doc_list) == 1):
        return doc_list[0]

    # sort by lower domain first
    doc_list.sort(key=len)
    return set(doc_list[0]).intersection(*doc_list[1:])


def process_query(query_string, dictionary, posting_file):
    # Categorise query
    query_clauses = categorise_query(query_string)
    phrasal_results = []

    # We only do search on phrasal queries first
    contains_phrasal_query = False

    for query_clause in query_clauses:
        if query_clause[1] == QueryType.PHRASAL:
            phrasal_query_docs = get_phrasal_query_doc_id(query_clause[0], dictionary, posting_file)
            phrasal_results.append(phrasal_query_docs)
            contains_phrasal_query = True

    # From the list of phrasal results, we do free text search
    query_list = " ".join([word for word, _, _ in query_clauses]).split(" ")

    if contains_phrasal_query:
        # Combine phrasal results
        phrasal_result_doc_id = intersect_document_ids(phrasal_results)
        final_result = free_text_search(query_list, dictionary, posting_file, phrasal_result_doc_id)
    else:
        final_result = free_text_search(query_list, dictionary, posting_file, None)

    return final_result


def free_text_search(query_list, dictionary, posting_file, accepted_doc_id):
    """rank the list of document based on the query given

    Args:
        query_list (list): the list of query string to be ranked against
        dictionary (dictionary): dictionary of the posting lists
        posting_file (str): address to the posting file list
        accepted_doc_id (set): set of valid doc_id from phrasal queries in the given query text

    Returns:
        str: search rank result
    """
    # all tokenization should consistent with the one from index.py
    stemmer = nltk.stem.PorterStemmer()
    token_list = list(map(lambda x: stemmer.stem(x).lower(), query_list))
    query_counter = Counter(token_list)
    query_keys = list(query_counter.keys())

    query_length = 0
    query_term_vector = []

    # To get a faster quering, we precomute the value for tf_idf query vector
    # Next time, we only need to do dot product with each of the given document 
    for term in query_keys:
        tf_idf_score = 0

        # dictionary is in the form of:
        # dict = {
        #   "term1": (df1, pointer_to_posting_file),
        #   "term2": (df2, pointer_to_posting_file),
        #   ...
        # }
        #
        # posting_list is in the form of:
        # [(docID_1, tf1), (docID2, tf2), ...]
        if (term in dictionary):
            term_info = dictionary[term]
            term_df = term_info[0]
            term_pointer = term_info[1]
            no_of_document = len(dictionary["LENGTH"])

            posting_file.seek(term_pointer)
            posting_list = pickle.load(posting_file)

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

    # calculate tf_idf_score for each term (if it exists in the dictionary)
    for term in query_keys:
        tf_idf_score = 0

        if (term in dictionary):
            term_info = dictionary[term]
            term_pointer = term_info[1]
            
            posting_file.seek(term_pointer)
            posting_list = pickle.load(posting_file)
            
            for (doc_id, term_freq, _) in posting_list:
                tf_idf_score = 1 + math.log(term_freq, 10)  # tf
                document_term_dict[term][doc_id] = tf_idf_score / dictionary["LENGTH"][doc_id]  # normalize score
                potential_document_id.add(doc_id)
    
    # sort the list in case two or more document_id score the same
    potential_document_id = sorted(list(potential_document_id))

    # calculate and rank document_id based on cosine score
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
        ranking_list.sort(key=lambda x: x[0], reverse=True)

    if (accepted_doc_id == None):
        tf_idf_doc_list = [y for x, y in ranking_list]
    else:
        tf_idf_doc_list = [y for x, y in ranking_list if y in accepted_doc_id]
        
    return " ".join(str(doc_id) for doc_id in tf_idf_doc_list)


def normalize_list(lst, denominator):
    return list(map(lambda x: x/denominator, lst))