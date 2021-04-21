from enum import Enum
from constants import *
from nltk import word_tokenize
from word_processing import lemmatize, stem, sanitise

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
    doc_list = [doc_list1, doc_list2]

    # sort by lower domain first
    doc_list.sort(key=len)
    return list(set(doc_list[0]).intersection(*doc_list[1:]))

def union_document_ids(doc_list1, doc_list2):
    doc_list = [doc_list1, doc_list2]

    return list(set(doc_list[0]).union(*doc_list[1:]))

def stem_clauses(query_clauses, stemmer, lemmtzr):
    stemmed_clauses = []
    for and_clause in query_clauses:
        stemmed_and_clause = []

        # Iterate through each clause under the and clause
        for or_clause in and_clause:
            clause, clause_type = or_clause
            
            # Use the same method as in indexing
            tokens = sanitise(clause)

            # Stem word by word
            stemmed_tokens = tokens
            
            # Do lemmatization before stemming
            if (USE_LEMMATIZER):
                stemmed_tokens = lemmatize(stemmed_tokens)
            if (USE_STEMMER):
                stemmed_tokens = stem(stemmed_tokens)

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
