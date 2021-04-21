from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
from collections import Counter

from free_text_query import free_text_search
from phrasal_query import get_phrasal_query_doc_id
from query_expansion import expand_clause
from query_util import *
from constants import *
from query_prf import *

def process_query(query_string, dictionary, posting_file):
    """
    Perform search based on the query_string.
    """
    stemmer = PorterStemmer()
    lemmatzr = WordNetLemmatizer()

    # Categorise query
    query_clauses = categorise_query(query_string)

    # Stem
    stemmed_query_clauses = stem_clauses(query_clauses, stemmer, lemmatzr)

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

                curr_result  = free_text_search(free_text_list, dictionary, posting_file, None, do_ranking=False)
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
    final_result, query_term_vector = free_text_search(query_list, dictionary, posting_file, combined_result, do_ranking=True)

    if USE_PRF:
        final_result = prf_search(final_result, query_list, query_term_vector, dictionary, posting_file)

    # Omit scores for final output
    return " ".join(str(doc_id) for doc_id in final_result)
