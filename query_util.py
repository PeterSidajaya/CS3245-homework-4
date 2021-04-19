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
    of the clause and its type.
    
    Multiple clause implies that each clause is connected via AND operator.

    e.g. Input: "little puppy" AND chihuahua 
         Output: [<little puppy, QueryType.PHRASAL>, <chihuahua, QueryType.FREE_TEXT>]
    """
    if (len(query) < 1):
      return
    
    curr_str_idx = 0
    query_clauses = []
    is_last_keyword_quote = False

    # i = 0
    while curr_str_idx != -1 and curr_str_idx < len(query):
        spliced_query = query[curr_str_idx:]
        clause = ""
        closest_keyword_pos = 0
        closest_keyword = ""

        print("Spliced query: {}".format(spliced_query))
        if (is_last_keyword_quote):
            # Find the next double quote
            closest_keyword_pos = spliced_query.find(DOUBLE_QUOTE_KEYWORD)
            closest_keyword = DOUBLE_QUOTE_KEYWORD

            is_last_keyword_quote = False
            print("Finding the next DQUOTE: {}".format(closest_keyword_pos))
        
        else:
            # Finding closest query keyword
            keywords_pos = list(map(lambda keyword: spliced_query.find(keyword), QUERY_KEYWORDS))
            closest_keyword_idx = find_closest_idx(keywords_pos)
            closest_keyword_pos = keywords_pos[closest_keyword_idx]
            closest_keyword = QUERY_KEYWORDS[closest_keyword_idx]

            is_last_keyword_quote = closest_keyword == DOUBLE_QUOTE_KEYWORD
            print("Finding the next KEYWORD: {}".format(closest_keyword_pos))
        
        # If there is no more keyword, we splice until the end of the string
        next_idx = len(spliced_query) if closest_keyword_pos == -1 else closest_keyword_pos
        clause = spliced_query[:next_idx].strip()
        print("Curr idx: {}, next_idx: {}".format(curr_str_idx, next_idx))
        if (len(clause) > 0):
          print("Clause: {}".format(clause))

        # Update position
        curr_str_idx = closest_keyword_pos if closest_keyword_pos == -1 else curr_str_idx + next_idx + len(closest_keyword)
        print("New curr Idx: {}".format(curr_str_idx))
        print()

    return query_clauses

############ HELPERS ############

def find_closest_idx(keywords_pos):
    smallest_idx = math.inf
    for idx, pos in enumerate(keywords_pos):
        if pos > -1 and pos < smallest_idx:
            smallest_idx = idx

    smallest_idx = -1 if math.isinf(smallest_idx) else smallest_idx
    return smallest_idx