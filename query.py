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
    """
    Perform a search based on the query_string.

    This method perform preprocessing on the query string (tokenization, stemming, lemmatization based
    on the settings) and categorises them. For details on how a query is categorised,
    please refer to query_util.categorise_query(). Once categorised, we perform searches
    based on its type (PHRASAL or FREE_TEXT search).

    Phrasal search is done via positional index search. For details please look at phrasal_query.py.
    Free text search is done via normal search on each free text word. For details please look at free_text_query.py.

    Within a subquery (OR clauses), it performs unions on all the results,
    and intersects all the results between subqueries (AND clauses).

    It then performs a ranking based scoring system. Please refer to scoring.py for details.

    If PRF is enabled, it finally performs PRF.

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

            # Apply query expansion
            expanded_clause_word = expand_clause(clause_word)
            free_text_list = word_tokenize(expanded_clause_word)
            expanded_words.extend(free_text_list)

            # Get result for the current clause
            if clause_type == QueryType.PHRASAL:
                # For phrasal, we first perform the phrasal search
                phrasal_result = get_phrasal_query_doc_id(clause_word, dictionary, posting_file)
                phrasal_result = tag_results(phrasal_result, QueryType.PHRASAL)

                # Followed by the free text search on expanded words
                expanded_result = free_text_search(free_text_list, dictionary, posting_file, None, do_ranking=False)
                expanded_result = tag_results(expanded_result, QueryType.FREE_TEXT)

                # Result is union of the two
                curr_result = union_document_ids(phrasal_result, expanded_result)

            elif clause_type == QueryType.FREE_TEXT:
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
        impt_words = prf_impt_words(final_result, dictionary)
        impt_clause = categorise_query(" ".join(impt_words))[0]
        # Perform the search again with important words, but without PRF (only do it once)
        return process_query(query_string, dictionary, posting_file, use_prf=False, prf_clause=impt_clause)

    else:
        # Omit scores for final output
        return " ".join(str(doc_id) for doc_id in final_result)
