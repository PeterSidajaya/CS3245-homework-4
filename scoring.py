from constants import *
from index_helper import get_word_list
from query_util import QueryType

import math

def rank_document_ids(results_with_score, tagged_prio_list=None):
    """
    Rank document ids, given two lists:
        1. results_with_score: the list of results with score but no tag
        2. tagged_prio_list: the list of results with tags but no score

    Tag here refers to QueryType enum.

    In this ranking process, score of the results that comes from phrasal search and
    results that comes from tagged_prio_list will be multiplied with weight
    (defined in constants.py).
    """
    # Weed out the weakest results, extreme low score lowers the overall benchmark
    initial_benchmark = get_avg_score(results_with_score) * FILTER_STRENGTH
    filtered_score_list = list(filter(lambda x: x[1] > initial_benchmark, results_with_score))

    # Default score will only be assigned to doc_ids in tagged_prio_list
    default_score = get_avg_score(filtered_score_list) * PRIORITY_WEIGHT
    # Default tag will only be assigned to doc_ids in results_with_score
    default_tag = QueryType.FREE_TEXT

    # Combine the scored results and tagged result
    combined_list = combine_score_and_tag(filtered_score_list, tagged_prio_list, default_score, default_tag)

    # Apply weighting for results that comes from phrasal query
    weighted_list = []
    for elem in combined_list:
        doc_id, score, clause_type = elem
        if clause_type == QueryType.PHRASAL:
            score *= PHRASAL_WEIGHT
        weighted_list.append((doc_id, score))

    # Weed out weaker results
    weighted_avg_score = get_avg_score(weighted_list)
    weighted_list = list(filter(lambda x: x[1] > (weighted_avg_score * FILTER_STRENGTH), weighted_list))

    # Sort
    weighted_list.sort(key=lambda x: x[1], reverse=True)
    return weighted_list

def combine_score_and_tag(scored_list, tagged_list, default_score, default_tag):
    """
    Output the merging of the scored_list and tagged_list.

    The method output a list, where each element is (doc_id, score, tag).

    The 2 lists are:
        1. scored_list: the list of results with score but no tag
        2. tagged_list: the list of results with tags but no score

    The merging is as follows:
        1. If a result is contained within both scored_list and tagged_list,
           then the result will inherit the score from scored_list & tag from
           tagged_list.
        2. If a result is only contained in scored_list,
           then the result will inherit the score from scored_list & tag from
           default_tag.
        3. If a result is only contained in tagged_list,
           then the result will inherit the score from default_score & tag from
           tagged_list.
    """
    tagged_score = []

    # Tagged_prio_dict has the format of dict[doc_id] = (score, clause_type)
    tagged_score_dict = {}
    print(tagged_list)

    # Tagged_prio_list has the format of (doc_id, clause_type)
    for elem in tagged_list:
        tagged_score_dict[elem[0]] = (default_score, elem[1])

    # Results_with_score has the format of (doc_id, score)
    for elem in scored_list:
        if elem[0] in tagged_score_dict:
            tagged_score_dict[elem[0]] = (elem[1], tagged_score_dict[elem[0]][1])
        else:
            tagged_score_dict[elem[0]] = (elem[1], default_tag)

    # Final output has the format of (doc_id, score, tag)
    return [(k, v[0], v[1]) for k, v in tagged_score_dict.items()]

def get_avg_score(results_with_score):
    """
    Get the average score of the results.
    ASsumes that the format of each element is (doc_id, score)
    """
    score_sum = 0
    for res in results_with_score:
        score_sum += res[1]

    if len(results_with_score) > 0:
        return score_sum / len(results_with_score)
    else:
        return 0

def get_results_for_vector(query_term_vector, query_keys, dictionary, posting_file, tagged_prio_list, do_ranking):
    """
    Find the document vectors for the given query term vector.

    The result is a list of document id's and their corresponding document vectors for
    the given words in the query_keys, sorted by their cosine distance with query_term_vector.

    # do_ranking should be one of the QUERY MODES in constants. If NO_RANKING is selected, the
    # valid documents are returned in arbitrary order. GET_RANKING will get them sorted by cosine score,
    # and GET_RANKING_AND_VECTORS will return the full document matrix (include document vectors)

    Args:
        query_term_vector (list(int)) The tf-idf of the query terms in query_keys
        query_keys (list(str)) The unique words in the query
        dictionary (dictionary): dictionary of the posting lists
        posting_file (str): address to the posting file list
        do_ranking (bool): Whether the returned list is sorted or not
    Returns:
        if do_ranking:
            list(doc_id, score) List of document id's for the query vector and scores
        else:
            list(doc_id) List of document ids
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