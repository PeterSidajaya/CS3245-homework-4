from enum import Enum
from constants import *
from nltk import word_tokenize

import re
import math

class QueryType(Enum):
    FREE_TEXT = 0
    PHRASAL = 1

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
    
    and_clauses = re.split(AND_KEYWORD, query)
    query_clauses = []

    for i in range(len(and_clauses)):
        processed_and_clause = []
        and_clause = and_clauses[i]
        curr_str_idx = 0
        is_last_keyword_quote = False

        while curr_str_idx != -1 and curr_str_idx < len(and_clause):
            spliced_query = and_clause[curr_str_idx:]
            clause = ""
            closest_keyword_pos = 0
            
            # Finding closest double quote
            closest_keyword_pos = spliced_query.find(DOUBLE_QUOTE_KEYWORD)
            
            if (is_last_keyword_quote):
                is_last_keyword_quote = False
                clause_type = QueryType.PHRASAL
            else:
                is_last_keyword_quote = True
                clause_type = QueryType.FREE_TEXT

            # If there is no more keyword, we splice until the end of the string
            next_idx = len(spliced_query) if closest_keyword_pos == -1 else closest_keyword_pos
            clause = spliced_query[:next_idx].strip()

            if (len(clause) > 0):
                processed_and_clause.append((clause, clause_type))

            # Update position
            curr_str_idx = closest_keyword_pos if closest_keyword_pos == -1 else curr_str_idx + next_idx + len(DOUBLE_QUOTE_KEYWORD)

        query_clauses.append(processed_and_clause)
    
    return query_clauses

def intersect_document_ids(doc_list1, doc_list2):
    """
    Returns the intersection between doc_list1 and doc_list2.

    Both doc_list1 and doc_list2 has the following format for each elem:
    (doc_id, clause_type)

    Special case: when a document ID is contained both in doc_list1 and doc_list2
    BUT with different tags, we will prioritise QueryType.PHRASAL.

    This is so because results tagged with QueryType.PHRASAL is deemed
    to have higher importance.

    e.g. doc_list1 = [(1, QueryType.PHRASAL), (2, QueryType.FREE_TEXT)]
         doc_list2 = [(1, QueryType.FREE_TEXT)]

         Outputs [(1, QueryType.PHRASAL)]
    """
    # TODO: See whether sorting step is necessary or not
    doc_list1.sort(key=lambda x: x[0])
    doc_list2.sort(key=lambda x: x[0])

    result = []
    idx0, idx1 = 0, 0
    while idx0 < len(doc_list1) and idx1 < len(doc_list2):
        # Matching element
        if doc_list1[idx0][0] == doc_list2[idx1][0]:
            is_phrasal = doc_list1[idx0][1] == QueryType.PHRASAL or doc_list2[idx1][1] == QueryType.PHRASAL
            clause_type = QueryType.PHRASAL if is_phrasal else QueryType.FREE_TEXT
            result.append((doc_list1[idx0][0], clause_type))
            idx0 += 1
            idx1 += 1

        # doc_list1[pointer] is smaller, advance the pointer
        elif doc_list1[idx0][0] < doc_list2[idx1][0]:
            idx0 += 1
        
        # doc_list2[pointer] is smaller, advance the pointer
        else: 
            idx1 += 1
    return result

def union_document_ids(doc_list1, doc_list2):
    """
    Returns the union between doc_list1 and doc_list2.

    Both doc_list1 and doc_list2 has the following format for each elem:
    (doc_id, clause_type)

    Special case: when a document ID is contained both in doc_list1 and doc_list2
    BUT with different tags, we will prioritise QueryType.PHRASAL.

    This is so because results tagged with QueryType.PHRASAL is deemed
    to have higher importance.

    e.g. doc_list1 = [(1, QueryType.PHRASAL), (2, QueryType.FREE_TEXT)]
         doc_list2 = [(1, QueryType.FREE_TEXT)]

         Outputs [(1, QueryType.PHRASAL), (2, QueryType.FREE_TEXT)]
    """
    # TODO: See whether sorting step is necessary or not
    doc_list1.sort(key=lambda x: x[0])
    doc_list2.sort(key=lambda x: x[0])

    result = []
    idx0, idx1 = 0, 0

    # Iterate while both lists are alive
    while idx0 < len(doc_list1) and idx1 < len(doc_list2):
        # Matching element
        if doc_list1[idx0][0] == doc_list2[idx1][0]:
            is_phrasal = doc_list1[idx0][1] == QueryType.PHRASAL or doc_list2[idx1][1] == QueryType.PHRASAL
            clause_type = QueryType.PHRASAL if is_phrasal else QueryType.FREE_TEXT
            result.append((doc_list1[idx0][0], clause_type))
            idx0 += 1
            idx1 += 1

        # doc_list1[pointer] is smaller, advance the pointer
        elif doc_list1[idx0][0] < doc_list2[idx1][0]:
            result.append(doc_list1[idx0])
            idx0 += 1

        # doc_list2[pointer] is smaller, advance the pointer
        else: 
            result.append(doc_list2[idx1])
            idx1 += 1
    
    # List one has still elements 
    if idx0 < len(doc_list1):
        result.extend(doc_list1[idx0:])
    
    # List two has still elements
    if idx1 < len(doc_list2):
        result.extend(doc_list2[idx1:])

    return result

def stem_clauses(query_clauses, stemmer, lemmtzr):
    stemmed_clauses = []
    for and_clause in query_clauses:
        stemmed_and_clause = []

        # Iterate through each clause under the and clause
        for or_clause in and_clause:
            clause, clause_type = or_clause
            
            tokens = word_tokenize(clause)

            # Stem word by word
            stemmed_tokens = tokens
            if (USE_LEMMATIZER):
                stemmed_tokens = list(map(lambda x: lemmtzr.lemmatize(x).lower(), stemmed_tokens))
            if (USE_STEMMER):
                stemmed_tokens = list(map(lambda x: stemmer.stem(x).lower(), stemmed_tokens))

            stemmed_words = " ".join(stemmed_tokens)
            stemmed_and_clause.append((stemmed_words, clause_type))
        
        stemmed_clauses.append(stemmed_and_clause)
    return stemmed_clauses

def get_words_from_clauses(query_clauses):
    list_of_words = []
    for and_clause in query_clauses:
        and_clause_words = " ".join([clause_word for clause_word, clause_type in and_clause]).split(" ")
        list_of_words.extend(and_clause_words)
    return list_of_words
