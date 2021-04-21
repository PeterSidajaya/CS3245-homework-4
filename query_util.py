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
    """Split a query into a list of clauses.

    A query can be thought of multiple subqueries connected by AND operators, and each
    subquery can consist of multiple clauses, which are either free text queries or phrasal queries.
    The clauses in a subquery are implicitly connected via OR operators.
    
    Each element in the result is a subquery, which is a list of tuples, where each tuple
    is a clause, its type, and the boolean operator to connect with the next element.

    This method reads the query from left to right; stopping at every keywords ('"', AND)
    and determine that the substring in between as keywords as clauses.

    e.g. Input: "little puppy" AND chihuahua
         Output: [('little puppy', <QueryType.PHRASAL>), ('chihuahua', <QueryType.FREE_TEXT>)]

    Args: 
        query(str): The raw query
    Returns:
        list(list(clause, QueryType)): The list of subqueries resulting from splitting the query,
                                                  where each subquery is a list of clauses
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
    """Returns the intersection between doc_list1 and doc_list2.

    Both doc_list1 and doc_list2 has the following format for each elem:
    (doc_id, clause_type)

    Special case: when a document ID is contained both in doc_list1 and doc_list2
    BUT with different tags, we will prioritise QueryType.PHRASAL.

    This is so because results tagged with QueryType.PHRASAL is deemed
    to have higher importance.

    e.g. doc_list1 = [(1, QueryType.PHRASAL), (2, QueryType.FREE_TEXT)]
         doc_list2 = [(1, QueryType.FREE_TEXT)]

         Outputs [(1, QueryType.PHRASAL)]
        
    Args:
        doc_list1 (list(doc_id, QueryType)): The first list of clauses
        doc_list2 (list(doc_id, QueryType)): The second list of clauses
    Returns:
        list(doc_id, QueryType): The merged list of clauses
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
    """Returns the union between doc_list1 and doc_list2.

    Both doc_list1 and doc_list2 has the following format for each elem:
    (doc_id, clause_type)

    Special case: when a document ID is contained both in doc_list1 and doc_list2
    BUT with different tags, we will prioritise QueryType.PHRASAL.

    This is so because results tagged with QueryType.PHRASAL is deemed
    to have higher importance.

    e.g. doc_list1 = [(1, QueryType.PHRASAL), (2, QueryType.FREE_TEXT)]
         doc_list2 = [(1, QueryType.FREE_TEXT)]

         Outputs [(1, QueryType.PHRASAL), (2, QueryType.FREE_TEXT)]
    
    Args:
        doc_list1 (list(doc_id, QueryType)): The first list of clauses
        doc_list2 (list(doc_id, QueryType)): The second list of clauses
    Returns:
        list(doc_id, QueryType): The merged list of clauses
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
    """Stem each of the subqueries in query_clauses.

    query_clauses should be a list of subqueries created by categorize_query, where each subquery is
    a list of tuples consisting of a string with the clause contents and its query type (phrasal or free text).
    
    The result is that all the clauses will have their clause contents sanitized and stemmed.

    Args:
        query_clauses (list(list(query_string, QueryType))): The list of query_clauses
        stemmer: The stemmer to use
        lemmtzr: The lemmatizer to use
    Returns:
        list(list(query_string, QueryType)): query_clauses with its contents sanitized
    """
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
            stemmed_words = " ".join(stemmed_tokens)
            stemmed_and_clause.append((stemmed_words, clause_type))

        stemmed_clauses.append(stemmed_and_clause)
    return stemmed_clauses

def get_words_from_clauses(query_clauses):
    """Retrieve all words from a list of lists of query clauses.
    
    Each query clause consists of a tuple with the clause contents and the query type.

    Args:
        query_clauses (list(list(string, QueryType))): A list of list of tuples 
    Returns:
        str: A string consisting of all the words in the query clauses
    """
    list_of_words = []
    for and_clause in query_clauses:
        and_clause_words = " ".join([clause_word for clause_word, clause_type in and_clause]).split(" ")
        list_of_words.extend(and_clause_words)
    return list_of_words

def get_query_term_vector(query_keys, query_counter, dictionary):
    """retrieve the tf-idf of the query vector.

    Args:
        query_keys (list(str)) The terms in the query
        query_counter (dict(str:int)) The number of occurences of each string in query_keys
        dictionary (dict) The dictionary of the posting lists
    Returns:
        list(float) The tf-idf query vector corresponding to query_keys
    """
    query_term_vector = []
    query_length = 0
    no_of_document = len(dictionary[DOCUMENT_LENGTH_KEYWORD])

    for term in query_keys:
        tf_idf_score = 0

        if (term in dictionary):
            term_info = dictionary[term]
            term_df = term_info[0]

            tf_idf_score = (1 + math.log(query_counter[term], 10)) * math.log(no_of_document / term_df)
            query_length += (tf_idf_score ** 2)

        query_term_vector.append(tf_idf_score)

    normalize_denominator = math.sqrt(query_length)
    if (normalize_denominator != 0):
        # final precompute query vector
        query_term_vector = normalize_list(query_term_vector, normalize_denominator)

    return query_term_vector

def normalize_list(lst, denominator):
    """Return a new list which is lst with every element divided by denominator.
    
    Args:
        lst (list(float)): The vector to normalize
        denominator (float): The normalizing factor
    Returns:
        The normalized vector
    """
    return list(map(lambda x: x/denominator, lst))

def tag_results(results, tag):
    """Returns a list of results where each element is (result, tag).

    Arguments:
        results (list(int)): A list of doc Id's
        tag (QueryType): The tag to tag with
    Returns:
        list(int, tag): The list of tagged results
    """
    return list(map(lambda x: (x, tag), results))
