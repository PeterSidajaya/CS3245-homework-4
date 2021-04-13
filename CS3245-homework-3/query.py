from collections import Counter
import nltk
import pickle
import math


def search(query, dictionary, postings_file):
    """rank the list of document based on the query given

    Args:
        query (str): the query string to be ranked against
        dictionary (dictionary): dictionary of the posting lists
        postings_file (str): address to the posting file list

    Returns:
        str: search rank result
    """
    posting_file = open(postings_file, 'rb')

    # all tokenization should consistent with the one from index.py
    stemmer = nltk.stem.PorterStemmer()
    token_list = list(map(lambda x: stemmer.stem(x).lower(), query.split(" ")))
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
            
            for (doc_id, term_freq) in posting_list:
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

        # maintain the top k results and store it in ranking_list
        k = 10

        if (not ranking_list):
            ranking_list.append((score, doc_id))
        else:
            if (score > ranking_list[-1][0]):
                for x in range(len(ranking_list)):
                    if (ranking_list[x][0] < score):
                        ranking_list.insert(x, (score, doc_id))
                        break
                
                if (len(ranking_list) > k):
                    ranking_list = ranking_list[:k]
            else:
                if (len(ranking_list) < k):
                    ranking_list.append((score, doc_id))

    return " ".join([str(y) for x, y in ranking_list])


def normalize_list(lst, denominator):
    return list(map(lambda x: x/denominator, lst))