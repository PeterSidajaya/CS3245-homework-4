#!/usr/bin/python3
from index_helper import *
from constants import *

import re
import nltk
import sys
import getopt
import os
import pickle
import math
import string
import csv

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")
    

# main function
def build_index(doc_id, out_dict, out_postings):
    """
    build index from the csv file,
    then output the dictionary file and postings file
    """
    stemmer = nltk.stem.PorterStemmer()

    # For testing - this is the limit of the entry we are taking
    sample_limit = 5                               
    dictionary = {}
    # This is where we'll store the length of each docs
    dictionary[DOCUMENT_LENGTH_KEYWORD] = {}        

    max_int = sys.maxsize
    while True:
        # decrease the max_int value by factor 10 
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int/10)

    # Opens the csv
    with open(doc_id, newline='', encoding='UTF-8') as f:
        reader = csv.reader(f)
        i = 0

        for row in reader:
            i += 1
            if i == 1:                  # skip first row
                continue
            if i == sample_limit + 2:
                break

            doc_id, title, content, date_posted, court = row

            word_list = nltk.tokenize.word_tokenize(content)
            filtered_list = [text for text in word_list if text not in string.punctuation]
            token_list = list(map(lambda x: stemmer.stem(x).lower(), filtered_list))

            content_dict = index_text(token_list)
            length = 0

            # Calculate and precompute df and length
            for term, tup in content_dict.items():
                tf, position_list = tup
                if term not in dictionary:
                    # First entry is df
                    dictionary[term] = (1, [(int(doc_id), tf, position_list),])      
                else:
                    df = dictionary[term][0] + 1
                    posting_list = dictionary[term][1] + [(int(doc_id), tf, position_list),]
                    dictionary[term] = (df, posting_list)
                # Document length is calculated from tf
                length += (1 + math.log(tf, 10)) ** 2       
            
            # Calculate document length for document normalization in search
            dictionary[DOCUMENT_LENGTH_KEYWORD][int(doc_id)] = math.sqrt(length)

    # Deals with pickles
    posting_file = open(out_postings, 'wb')
    dictionary_file = open(out_dict, 'wb')
    
    # Delete existing contents
    posting_file.truncate(0)                    
    dictionary_file.truncate(0)

    # Store posting_lists and dictionary to files
    for term, value in dictionary.items():
        if term == DOCUMENT_LENGTH_KEYWORD:     # LENGTH is not a term
            continue
        df, posting_list = value
        pointer = posting_file.tell()           # find our current position in the posting file
        pickle.dump(posting_list, posting_file) # dump the posting list into the posting file
        dictionary[term] = (df, pointer)        # keep track of the pointer
        
    pickle.dump(dictionary, dictionary_file)    # dump the dictionary into the dictionary file
    posting_file.close()
    dictionary_file.close()

input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

print("start indexing...")
build_index(input_directory, output_file_dictionary, output_file_postings)

# Testing purposes
# posting_file = open('postings.txt', 'rb')
# dictionary_file = open('dict.txt', 'rb')
# dictionary = pickle.load(dictionary_file)
# posting_list = get_list('destroy', dictionary, posting_file)
# print(posting_list[0])