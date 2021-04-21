### CONFIG ###

# Query processing
USE_STEMMER = True
USE_LEMMATIZER = False 

# Result ranking
PHRASAL_WEIGHT = 1.7    # the higher it is, the more weight phrasal results has
PRIORITY_WEIGHT = 1.1   # the higher it is, the more weight initial results has
FILTER_STRENGTH = 1.3  # the smaller it is, the more forgiving is the filter (thus more results)

# BM25 parameters
BM25_B = 0.75
BM25_K = 1.2

### INDEX ###
DOCUMENT_LENGTH_KEYWORD = "LENGTH"
DOCUMENT_AVG_LENGTH_KEYWORD = "LENGTH_AVG"
SPIMI_BLOCK_SIZE = 1000
POSTING_DIR = "temp_postings_result_dir/" 

### QUERY KEYWORD ###
AND_KEYWORD = "AND"
DOUBLE_QUOTE_KEYWORD = "\""
QUERY_KEYWORDS = [AND_KEYWORD, DOUBLE_QUOTE_KEYWORD]

### QUERY EXPANSION ###
EXPAND_NUM_OF_SYNONYMS = 3

### QUERY PSEUDO RELEVANCE FEEDBACK ###
PRF_NUM_OF_RESULTS = 5
PRF_NUM_OF_WORDS_PER_DOC = 1

### PHRASAL QUERIES ###
DOC_ID = 0 
TF = 1 
POS_LIST = 2