from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
from collections import Counter

from free_text_query import free_text_search
from phrasal_query import get_phrasal_query_doc_id
from query_expansion import expand_clause
from query_util import *
from constants import *
from query_prf import *

def process_query(query_string, dictionary, posting_file, use_prf=False, prf_clause=None):
    """Perform a search based on the query_string.

    This method peimpt_wordstion/stemming/lemmatization of the query string,
    then splits it into subqueries joined by AND consisting of clauses joined by implicit
    OR operators. On each clause, it processes the query with the method corresponimpt_wordss type,
    either a free text search or a phrasal search. It performs unions on all
    clauses within a subquery, and intersects all subqueries. It then performs a ranking
    impt_wordsrity based scoring system. If PRF is enabled, it finally performs PRF.
    It then prints out the final result of the query, sorted by score. 
    
    For more details, refer to README.

    Arguments:
        query_string (str): The raw query string
        dictionary (dict): The dictionary to the posting lists
        posting_file: The posting file handler
    Returns:
        String containing the result, which is the sorted list of doc ID's corresponding 
        to the query.
    """
    # Categorise query
    query_clauses = categorise_query(query_string)

    # Stem
    stemmed_query_clauses = stem_clauses(query_clauses)

    # If prf_clause is given, extend the last OR clause with the prf_clause.
    #
    # NOTE:
    # We do not want to add it as a separate AND clause as it will restrict the original query.
    # Additionally, prf_clause is already sanitised & lemma/stemmed, hence must not be passed into stem_clauses
    # to avoid over-stemming of the words.
    if not prf_clause is None:
        if len(stemmed_query_clauses) > 0:
            stemmed_query_clauses[-1].extend(prf_clause)
        else:
            # Edge case for empty query
            stemmed_query_clauses.append(prf_clause)

    all_and_results = []
    expanded_words = []
    for and_clause in stemmed_query_clauses:
        and_clause_result = []

        # Handle each clause in the and clause
        for or_clause in and_clause:
            clause_word, clause_type = or_clause
            curr_result = []

            # Get result for the current clause
            if clause_type == QueryType.PHRASAL:
                curr_result = get_phrasal_query_doc_id(clause_word, dictionary, posting_file)
                curr_result = tag_results(curr_result, QueryType.PHRASAL)
            elif clause_type == QueryType.FREE_TEXT:
                # If we want to use clause expansion, use this
                clause_word = expand_clause(clause_word)

                free_text_list = word_tokenize(clause_word)
                expanded_words.extend(free_text_list)

                curr_result = free_text_search(free_text_list, dictionary, posting_file, None, do_ranking=False)
                curr_result = tag_results(curr_result, QueryType.FREE_TEXT)

            # Combine with the existing results
            and_clause_result = union_document_ids(and_clause_result, curr_result)

        all_and_results.append(and_clause_result)

    # Perform intersection between and clauses
    combined_result = all_and_results[0]
    for and_clause_result in all_and_results:
        combined_result = intersect_document_ids(combined_result, and_clause_result)

    query_list = get_words_from_clauses(stemmed_query_clauses)
    query_list.extend(expanded_words)
    query_list = list(set(query_list))

    # Score and rank
    final_result = free_text_search(query_list, dictionary, posting_file, combined_result, do_ranking=True)

    if use_prf:
        # Get new words from PRF
        impt_words = prf_impt_words(final_result, query_string, dictionary)
        impt_clause = categorise_query(" ".join(impt_words))[0]
        # Perform the search again with important words, but without PRF (only do it once)
        return process_query(query_string, dictionary, posting_file, use_prf=False, prf_clause=impt_clause)

    else:
        # Omit scores for final output
        return " ".join(str(doc_id) for doc_id in final_result)
