from enum import Enum
from constants import *

import re
import math

class QueryType(Enum):
    FREE_TEXT = 0
    PHRASAL = 1

class BooleanOp(Enum):
    AND = 0
    OR = 1

def categorise_query(query: str):
    """
    Output a list of clauses, where each element is a tuple
    of the clause, its type, and the boolean operator to connect with the next element.
    
    Each clause will be tagged with the clause type and its boolean operator.
    Boolean operator is used to connect to the next clause.

    This method reads the query from left to right; stopping at every keywords ('"', AND)
    and determine that the substring in between as keywords as clauses.

    e.g. Input: "little puppy" AND chihuahua 
         Output: [('little puppy', <QueryType.PHRASAL>, <BooleanOp.AND>), ('chihuahua', <QueryType.FREE_TEXT>, <BooleanOp.OR>)]
    """
    if (len(query) < 1):
      return
    
    curr_str_idx = 0
    query_clauses = []
    is_last_keyword_quote = False

    while curr_str_idx != -1 and curr_str_idx < len(query):
        spliced_query = query[curr_str_idx:]
        clause = ""
        clause_type = None
        closest_keyword_pos = 0
        closest_keyword = ""

        if (is_last_keyword_quote):
            # Find the next double quote
            closest_keyword_pos = spliced_query.find(DOUBLE_QUOTE_KEYWORD)
            closest_keyword = DOUBLE_QUOTE_KEYWORD

            is_last_keyword_quote = False
            clause_type = QueryType.PHRASAL
        
        else:
            # Finding closest query keyword
            keywords_pos = list(map(lambda keyword: spliced_query.find(keyword), QUERY_KEYWORDS))
            closest_keyword_idx = find_closest_idx(keywords_pos)
            closest_keyword_pos = keywords_pos[closest_keyword_idx]
            closest_keyword = QUERY_KEYWORDS[closest_keyword_idx]

            is_last_keyword_quote = closest_keyword == DOUBLE_QUOTE_KEYWORD
            clause_type = QueryType.FREE_TEXT
        
        # If there is no more keyword, we splice until the end of the string
        next_idx = len(spliced_query) if closest_keyword_pos == -1 else closest_keyword_pos
        clause = spliced_query[:next_idx].strip()

        if (len(clause) > 0):
            op_type = BooleanOp.AND if closest_keyword == AND_KEYWORD else BooleanOp.OR
            query_clauses.append((clause, clause_type, BooleanOp.OR))

        # When we encounter AND, the previous clause is connected by AND
        if (closest_keyword == AND_KEYWORD and len(query_clauses) > 0):
            query_clauses[-1] = (query_clauses[-1][0], query_clauses[-1][1], BooleanOp.AND)

        # Update position
        curr_str_idx = closest_keyword_pos if closest_keyword_pos == -1 else curr_str_idx + next_idx + len(closest_keyword)

    return query_clauses

def intersect_document_ids(doc_list1, doc_list2):
    doc_list = [doc_list1, doc_list2]

    # sort by lower domain first
    doc_list.sort(key=len)
    return list(set(doc_list[0]).intersection(*doc_list[1:]))

def union_document_ids(doc_list1, doc_list2):
    doc_list = [doc_list1, doc_list2]

    return list(set(doc_list[0]).union(*doc_list[1:]))

############ HELPERS ############

def find_closest_idx(keywords_pos):
    """
    Given a list of integer, returns the smallest index that is above -1.
    """
    smallest_idx = math.inf
    smallest_pos = math.inf

    for idx, pos in enumerate(keywords_pos):
        if pos > -1 and pos < smallest_pos:
            smallest_idx = idx
            smallest_pos = pos

    smallest_idx = -1 if math.isinf(smallest_idx) else smallest_idx
    return smallest_idx
