from query_util import QueryType
from constants import *

def rank_document_ids(results_with_score, tagged_prio_list=None):
    # Default score will only be assigned to doc_ids in tagged_prio_list
    default_score = get_avg_score(results_with_score) * PRIORITY_WEIGHT
    # Default tag will only be assigned to doc_ids in results_with_score
    default_tag = QueryType.FREE_TEXT

    # Combine the scored results and tagged result
    combined_list = combine_score_and_tag(results_with_score, tagged_prio_list, default_score, default_tag)    

    # Apply weighting for results that comes from phrasal query
    weighted_list = []
    for elem in combined_list:
        doc_id, score, clause_type = elem
        if clause_type == QueryType.PHRASAL:
            score *= PHRASAL_WEIGHT
        weighted_list.append((doc_id, score))

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
    """
    score_sum = 0
    for res in results_with_score:
        score_sum += res[1]
    return score_sum / len(results_with_score)
