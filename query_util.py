from enum import Enum
from constants import *

import re

class QueryType(Enum):
    FREE_TEXT = 0
    PHRASAL = 1


def categorise_query(query: str):
    """
    Output a list of clauses, where each element is a tuple
    of the clause and its type.
    
    Multiple clause implies that each clause is connected via AND operator.

    e.g. Input: "little puppy" AND chihuahua 
         Output: [<little puppy, QueryType.PHRASAL>, <chihuahua, QueryType.FREE_TEXT>]
    """
    # Try to split by boolean op
    exp_split = query.split("AND")

    # Check for each clause its type and append it
    query_clauses = []
    for exp in exp_split:
        phrasal_search = re.search(PHRASAL_REGEX, exp)
        if (phrasal_search == None):
            query_clauses.append((exp.strip(), QueryType.FREE_TEXT))
        else:
            query_clauses.append((phrasal_search.group(1).strip(), QueryType.PHRASAL))
    
    return query_clauses


# TODO: What is this?
def stem_clause(clause: (str, QueryType), stemmer):
    """
    """
    return (stemmer.stem(clause[0]).lower(), clause[1]) 