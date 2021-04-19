#!/usr/bin/python3
from constants import *
from spimi import invert, merge_files

import re
import nltk
import sys
import getopt
import os
import pickle
import string
import csv
import shutil
import math


def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")
    

# main function
def build_index(doc_id, out_dict, out_postings):
    """
    build index from the csv file,
    then output the dictionary file and postings file
    """
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    dictionary = {}

    # # For testing purposes
    # limit = 1000

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

    if not os.path.exists(POSTING_DIR):
        os.mkdir(POSTING_DIR)
    else:
        shutil.rmtree(POSTING_DIR)
        os.mkdir(POSTING_DIR)

    # Opens the csv
    with open(doc_id, newline='', encoding='UTF-8') as f:
        reader = csv.reader(f)

        # this is the tokens that will be passed through to the invert() function
        multiple_doc_list = []

        num_of_blocks = 0
        files_in_block = 0

        for idx, row in enumerate(reader):
            # Skip first row
            if idx == 0:                  
                continue

            # # End if limit is reached, for testing purposes
            # if idx == limit:
            #     break

            doc_id, title, content, date_posted, court = row

            word_list = nltk.tokenize.word_tokenize(content)
            filtered_list = [text for text in word_list if text not in string.punctuation]

            # This line is if you want to do stemming
            token_list = list(map(lambda x: stemmer.stem(x.lower()), filtered_list))

            # This line is if you want to do lemmatization instead
            # token_list = list(map(lambda x: lemmatizer.lemmatize(x.lower()), filtered_list))

            multiple_doc_list.append((int(doc_id), token_list))
            files_in_block += 1

            # If the number of files scanned has reach block size, then invert first
            if files_in_block == SPIMI_BLOCK_SIZE:
                print('Inverting block number ' + str(num_of_blocks + 1))
                multiple_doc_list.sort()
                invert(multiple_doc_list, POSTING_DIR + 'temp_dictionary_0_' + str(num_of_blocks) + '.txt', POSTING_DIR + 'temp_posting_0_' + str(num_of_blocks) + '.txt')
                num_of_blocks += 1
                files_in_block = 0
                multiple_doc_list = []
        
    # Invert the remaining block
    if (files_in_block != 0):
        print('Inverting block number ' + str(num_of_blocks + 1))
        multiple_doc_list.sort()
        invert(multiple_doc_list, POSTING_DIR + 'temp_dictionary_0_' + str(num_of_blocks) + '.txt', POSTING_DIR + 'temp_posting_0_' + str(num_of_blocks) + '.txt')
        num_of_blocks += 1
        multiple_doc_list = []


    # MERGING STAGE, binary merging (iterate until height of binary tree)
    for i in range(math.ceil(math.log(num_of_blocks, 2))): 
        # Generation: #i
        k = 0
        for j in range(0, num_of_blocks, 2):
            if j + 1 < num_of_blocks:
                # do the merging process
                # Merging block: j and j+1
                merge_files(POSTING_DIR + 'temp_dictionary_' + str(i) + '_' + str(j) + '.txt', POSTING_DIR + 'temp_posting_'+str(i) + '_' + str(j) + '.txt',
                            POSTING_DIR + 'temp_dictionary_' + str(i) + '_' + str(j+1) + '.txt', POSTING_DIR + 'temp_posting_'+str(i) + '_' + str(j+1) + '.txt',
                            POSTING_DIR + 'temp_dictionary_' + str(i+1) + '_' + str(k) + '.txt', POSTING_DIR + 'temp_posting_'+str(i+1) + '_' + str(k) + '.txt')
            else:
                # when the number is odd (left only the last data), copy the final block instead
                # Copying block: j
                shutil.move(POSTING_DIR + 'temp_dictionary_' + str(i) + '_' + str(j) + '.txt', POSTING_DIR + 'temp_dictionary_' + str(i+1) + '_' + str(k) + '.txt')
                shutil.move(POSTING_DIR + 'temp_posting_' + str(i) + '_' + str(j) + '.txt', POSTING_DIR + 'temp_posting_' + str(i+1) + '_' + str(k) + '.txt')
            k += 1
        num_of_blocks = k

    try:
        # rename the merged posting and dictionary files, for clarity
        shutil.move(POSTING_DIR + 'temp_posting_' + str(i+1) + '_' + str(k-1) + '.txt', out_postings)
        shutil.move(POSTING_DIR + 'temp_dictionary_' + str(i+1) + '_' + str(k-1) + '.txt', out_dict)
    except Exception as ex:
        # this is to prevent when we only want to index 1 file
        shutil.move(POSTING_DIR + 'temp_posting_0_0.txt', out_dict)
        shutil.move(POSTING_DIR + 'temp_dictionary_0_0.txt', out_postings)


# Main function starts here
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