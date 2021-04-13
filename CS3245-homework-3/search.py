#!/usr/bin/python3
import re
from typing import Deque
import nltk
import sys
import getopt
import pickle
from query import search

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')

    infile = open(dict_file, 'rb')
    new_dict = pickle.load(infile)
    infile.close()

    in_file = open(queries_file, 'r', encoding="utf8")
    out_file = open(results_file, 'w', encoding="utf8")
    query_list = in_file.read().splitlines()

    while query_list:
        query = query_list.pop(0)
        if (not query):
            out_file.write("")
        else:
            out_file.write(search(query, new_dict, postings_file))
        
        if query_list:
            out_file.write('\n')

    in_file.close()
    out_file.close()


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
