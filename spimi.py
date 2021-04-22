from constants import *
from collections import Counter
from index_helper import index_text

import pickle
import os
import math


def invert(multiple_doc_list, dictionary_file_add, posting_file_add):
    """Invert a term stream to dictionary and posting files

    Args:
        multiple_doc_list (<list<doc_id, token_list>>): list consisting of (doc_id, token_list) tuples
        dictionary_file_add (str): address of the dictionary file
        posting_file_add (str): address of the posting file
    """
    dictionary = {}
    dictionary[DOCUMENT_LENGTH_KEYWORD] = {}        # This is where we'll store the length of each docs
    dictionary[IMPT_KEYWORD] = {}

    for doc_id, token_list in multiple_doc_list:
        content_dict = index_text(token_list)
        length = 0

        # Calculate and precompute df and length
        for term, tup in content_dict.items():
            tf, position_list = tup
            if term not in dictionary:
                # First entry is df, the rest is the posting list, which consist of the doc id, tf, and position list
                dictionary[term] = (1, [(int(doc_id), tf, position_list)])
            else:
                df = dictionary[term][0] + 1
                posting_list = dictionary[term][1] + [(int(doc_id), tf, position_list)]
                dictionary[term] = (df, posting_list)
            # Document length is calculated from tf
            length += (1 + math.log(tf, 10)) ** 2

        # Calculate document length for document normalization in search
        dictionary[DOCUMENT_LENGTH_KEYWORD][int(doc_id)] = math.sqrt(length)

        # Add most frequent words in the document to the dictionary (for PRF)
        token_counter = Counter(token_list)
        frequent_tokens = list(map(lambda x: x[0], token_counter.most_common(PRF_NUM_OF_WORDS_PER_DOC)))
        dictionary[IMPT_KEYWORD][int(doc_id)] = frequent_tokens

    # This part onwards will store the dictionary and posting list to the files

    dictionary_file = open(dictionary_file_add, 'wb')
    posting_file = open(posting_file_add, 'wb')

    # Delete the existing contents, if any
    dictionary_file.truncate(0)
    posting_file.truncate(0)

    # Store posting_lists and dictionary to files
    for term, value in dictionary.items():
        if term == DOCUMENT_LENGTH_KEYWORD:     # LENGTH is not a term, it will be stored in the dictionary
            continue
        if term == IMPT_KEYWORD:                # IMPT_KEYWORD is not a term, it will be stored in the dictionary
            continue
        df, posting_list = value
        pointer = posting_file.tell()           # find our current position in the posting file
        pickle.dump(posting_list, posting_file) # dump the posting list into the posting file
        dictionary[term] = (df, pointer)        # keep track of the pointer

    pickle.dump(dictionary, dictionary_file)
    posting_file.close()
    dictionary_file.close()


def merge_files(dictionary_file_1_add, posting_file_1_add, dictionary_file_2_add, posting_file_2_add,
    output_dictionary_add, output_posting_add):
    """Merge two pairs of dictionary and posting files

    Args:
        dictionary_1_add (str): add of the first dictionary file
        posting_file_1_add (str): address of the first posting file
        dictionary_file_2_add (str): address of the second dictionary file
        posting_file_2_add (str): address of the second posting file
        output_dictionary_add (str): address of the output dictionary file
        output_posting_add (str): address of the output posting file
    """
    # Open the dictionary files
    infile = open(dictionary_file_1_add, 'rb')
    dictionary_1 = pickle.load(infile)
    infile.close()
    infile = open(dictionary_file_2_add, 'rb')
    dictionary_2 = pickle.load(infile)
    infile.close()

    # Open the posting list files
    posting_file_1 = open(posting_file_1_add, 'rb')
    posting_file_2 = open(posting_file_2_add, 'rb')

    # Open the output files
    dictionary_output = open(output_dictionary_add, 'wb')
    posting_output = open(output_posting_add, 'wb')

    # Delete the existing contents, if any
    dictionary_output.truncate(0)
    posting_output.truncate(0)

    dictionary = {}

    # Iterate through dictionary 1
    for term in dictionary_1.keys():
        if term == DOCUMENT_LENGTH_KEYWORD:     # LENGTH is not a term, it will be stored in the dictionary
            continue
        if term == IMPT_KEYWORD:     # LENGTH is not a term, it will be stored in the dictionary
            continue
        df_1 = df_2 = 0
        df_1, pointer_1 = dictionary_1[term]
        posting_file_1.seek(pointer_1)
        posting_list_1 = pickle.load(posting_file_1)
        if term in dictionary_2:                            # check if the term exist in the other dictionary
            df_2, pointer_2 = dictionary_2[term]
            posting_file_2.seek(pointer_2)
            posting_list_2 = pickle.load(posting_file_2)
            # Using the old function should still work, as tuple comparison checks from first index
            posting_list = merge_lists(posting_list_1, posting_list_2)
        else:
            posting_list = posting_list_1

        # Dump into the output file
        pointer = posting_output.tell()
        pickle.dump(posting_list, posting_output)
        dictionary[term] = (df_1 + df_2, pointer)

    # For terms in dictionary_2 but not in dictionary_1
    for term in dictionary_2.keys():
        if term == DOCUMENT_LENGTH_KEYWORD:     # LENGTH is not a term, it will be stored in the dictionary
            continue
        if term == IMPT_KEYWORD:                # IMPT_KEYWORD is not a term, it will be stored in the dictionary
            continue
        if term not in dictionary:
            df_2, pointer_2 = dictionary_2[term]
            posting_file_2.seek(pointer_2)
            posting_list_2 = pickle.load(posting_file_2)

            # Dump into the output file
            pointer = posting_output.tell()
            pickle.dump(posting_list_2, posting_output)
            dictionary[term] = (df_2, pointer)

    # Combine the LENGTH
    length_dict1 = dictionary_1[DOCUMENT_LENGTH_KEYWORD]
    length_dict2 = dictionary_2[DOCUMENT_LENGTH_KEYWORD]
    length_dict = {**length_dict1, **length_dict2}          # merge two dictionaries
    dictionary[DOCUMENT_LENGTH_KEYWORD] = length_dict

    # Combine the IMPT
    impt_dict1 = dictionary_1[IMPT_KEYWORD]
    impt_dict2 = dictionary_2[IMPT_KEYWORD]
    impt_dict = {**impt_dict1, **impt_dict2}             # merge two dictionaries
    dictionary[IMPT_KEYWORD] = impt_dict

    # Dump the dictionary
    pickle.dump(dictionary, dictionary_output)

    # Close the files
    posting_file_1.close()
    posting_file_2.close()
    dictionary_output.close()
    posting_output.close()

    # Remove the old files
    os.remove(dictionary_file_1_add)
    os.remove(dictionary_file_2_add)
    os.remove(posting_file_1_add)
    os.remove(posting_file_2_add)


def merge_lists(list_1, list_2):
    """Merge two sorted lists into one sorted list with an OR operation.

    Args:
        list_1 (list<int>) The first sorted list
        list_2 (list<int>) The second sorted list
    """
    i = 0
    j = 0
    list = []
    while i < len(list_1) and j < len(list_2):
        if list_1[i] < list_2[j]:
            list.append(list_1[i])
            i += 1
        elif list_1[i] > list_2[j]:
            list.append(list_2[j])
            j+= 1
        elif list_1[i] == list_2[j]:
            list.append(list_1[i])
            i += 1
            j += 1
    list.extend(list_1[i:])
    list.extend(list_2[j:])
    return list
