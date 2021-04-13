from collections import deque
from spimi import merge_lists
from config import make_pointer
import nltk
import functools
import pickle
import math

def query_shunting(query):
    """
    Process the query using the shunting-yard algorithm

    Args:
        query (string): the query to be processed

    Returns:
        queue: the query queue in reverse polish notation (RPN)
    """
    stemmer = nltk.stem.PorterStemmer()
    operators = {'AND', 'OR', 'NOT'}
    brackets = {'(', ')'}
    output_queue, operator_stack = deque([]), deque([])
    query = deque(query.split())
    
    while query:
        token = query.popleft()

        # separate the brackets from the word
        if token[0] == '(':
            query.appendleft(token[1:])
            token = '('
        elif token != ')' and token[-1] == ')':
            query.appendleft(')')
            token = token[:-1]

        # shunting-yard algortihm, see wikipedia for full (note that NOT is right-associative unary operator)
        if (token not in operators) and (token not in brackets):
            output_queue.append(stemmer.stem(token).lower())
        elif token in operators:
            while ((len(operator_stack) != 0) and
                (precedence(operator_stack[0], token)) and
                (operator_stack[0] != '(')):
                output_queue.append(operator_stack.popleft())
            operator_stack.appendleft(token)
        elif token == '(':
            operator_stack.appendleft(token)
        elif token == ')':
            while (operator_stack[0] != '('):
                output_queue.append(operator_stack.popleft())
            if (operator_stack[0] == '('):
                operator_stack.popleft()
    
    while (len(operator_stack) != 0):
        output_queue.append(operator_stack.popleft())
    
    return output_queue


def precedence(op1, op2):
    operators = ['(', ')', 'NOT', 'AND', 'OR']
    return operators.index(op1) < operators.index(op2)


def get_intersection(left, right):
    """
    Get the intersection between two posting lists, with the help of skip pointers

    Args:
        left (list): the first list
        right (list): the second list

    Returns:
        list: intersection of the lists
    """

    # left/right value example: [(1, True, 2), (5, False, None), (8, True, 4), (9, False, None),
    #                            (10, True, 6), (13, False, None), (15, False, None)]
    #
    # left/right tuple identity: (docID, has_skip_pointer, skip_pointer_location)
    def get_doc_id(postings_list, curr_pointer):
        return postings_list[curr_pointer][0]

    def has_next_pointer(postings_list, curr_pointer):
        return postings_list[curr_pointer][1]

    def get_next_pointer(postings_list, curr_pointer):
        return postings_list[curr_pointer][2]

    
    left_pointer = 0
    right_pointer = 0

    result = []

    while (left_pointer < len(left) and right_pointer < len(right)):
        left_docID = get_doc_id(left, left_pointer) 
        right_docID = get_doc_id(right, right_pointer) 

        if (left_docID == right_docID):
            result.append(left_docID)
            left_pointer += 1
            right_pointer += 1
        elif (left_docID < right_docID):
            if (has_next_pointer(left, left_pointer) and
                get_doc_id(left, get_next_pointer(left, left_pointer)) <= get_doc_id(right, right_pointer)):
                while (has_next_pointer(left, left_pointer) and
                       get_doc_id(left, get_next_pointer(left, left_pointer)) <= get_doc_id(right, right_pointer)):
                    left_pointer = get_next_pointer(left, left_pointer)
                
                continue
            left_pointer += 1
        else:
            if (has_next_pointer(right, right_pointer) and
                get_doc_id(right, get_next_pointer(right, right_pointer)) <= get_doc_id(left, left_pointer)):
                while (has_next_pointer(right, right_pointer) and
                       get_doc_id(right, get_next_pointer(right, right_pointer)) <= get_doc_id(left, left_pointer)):
                    right_pointer = get_next_pointer(right, right_pointer)
                continue

            right_pointer += 1
    return make_pointer(result)


def get_union(left, right):
    """get the union between two posting lists

    Args:
        left (list): the first list
        right (list): the second list

    Returns:
        list: union of the lists
    """
    left_val = [i for i, j, k in left]
    right_val = [i for i, j, k in right]
    final_list = merge_lists(left_val, right_val)
    return make_pointer(final_list)


def get_complement(left, right):
    """get the complement between two posting lists (left \ right)

    Args:
        left (list): the first list
        right (list): the second list

    Returns:
        list: complement of the lists (left \ right)
    """
    left_val = [i for i, j, k in left]
    right_val = [i for i, j, k in right]
    i = 0
    j = 0
    tmp_list = []
    while i < len(left_val) and j < len(right_val):
        if(left_val[i] < right_val[j]):
            tmp_list.append(left_val[i])
            i += 1
        elif(left_val[i] > right_val[j]):
            j += 1
        elif(left_val[i] == right_val[j]):
            i += 1
            j += 1
    tmp_list.extend(left_val[i:])
    return make_pointer(tmp_list)


def search(query, dictionary, postings_file):
    """process the query and return the resulting posting list

    Args:
        query (queue): the query queue in reverse polish notation (RPN)
        dictionary (dictionary): dictionary of the posting lists
        postings_file (str): address to the posting file list

    Returns:
        str: search result
    """
    query_queue = query_shunting(query)
    posting_file = open(postings_file, 'rb')
    operators = ['(', ')', 'NOT', 'AND', 'OR']
    eval_stack = deque([])

    # read out the query_queue
    while len(query_queue) != 0:
        token = query_queue.popleft()
        if token not in operators:
            if token in dictionary:
                pointer = dictionary[token][1]
                posting_file.seek(pointer)
                eval_stack.append(pickle.load(posting_file))
            else:
                eval_stack.append([])
        elif token == 'NOT':
            operand = eval_stack.pop()
            list_diff = get_complement(dictionary['ALL POSTING'], operand)
            eval_stack.append(list_diff)
        elif token == 'AND':
            operand_1 = eval_stack.pop()
            operand_2 = eval_stack.pop()
            intersect = get_intersection(operand_1, operand_2)
            eval_stack.append(intersect)
        elif token == 'OR':
            operand_1 = eval_stack.pop()
            operand_2 = eval_stack.pop()
            union = get_union(operand_1, operand_2)
            eval_stack.append(union)

    if len(eval_stack[0]) != 0:
        get_val = sorted([i for i, j, k in eval_stack[0]])
        return " ".join([str(i) for i in get_val])
    else:
        return ""