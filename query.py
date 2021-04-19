from nltk.stem import WordNetLemmatizer
from collections import Counter

from free_text_query import free_text_search
from phrasal_query import get_phrasal_query_doc_id
from query_expansion import expand_clause
from query_util import *
from constants import *

def process_query(query_string, dictionary, posting_file):
    lemmatzr = WordNetLemmatizer()

    # Categorise query
    query_clauses = categorise_query(query_string)
    
    # Lemmatise
    lemmatized_query_clauses = list(map(lambda clause: (lemmatzr.lemmatize(clause[0]).lower(), clause[1], clause[2]), query_clauses))

    last_op_type = BooleanOp.OR
    results = []
    
    for query_clause in lemmatized_query_clauses:
        clause_word, clause_type, op_type = query_clause
        curr_result = []

        # Get result for the current clause
        if clause_type == QueryType.PHRASAL:
            curr_result = get_phrasal_query_doc_id(clause_word, dictionary, posting_file)
        elif clause_type == QueryType.FREE_TEXT:
            # If we want to use clause expansion, use this
            # clause_word = expand_clause(query_clause)
            
            free_text_list = clause_word.split(" ")
            curr_result = free_text_search(free_text_list, dictionary, posting_file, None, do_ranking=False)

        # Combine with the existing results
        if last_op_type == BooleanOp.AND:
            results = intersect_document_ids(results, curr_result)
        elif last_op_type == BooleanOp.OR:
            results = union_document_ids(results, curr_result)

        # Update for next clause
        last_op_type = op_type

    # Score and rank
    query_list = " ".join([clause_word for clause, clause_type, op_type in query_clauses]).split(" ")
    final_result = free_text_search(query_list, dictionary, posting_file, results, do_ranking=True)

    return " ".join(str(doc_id) for doc_id in final_result)
