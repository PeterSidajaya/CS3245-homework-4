### CONFIG ###

# Indexing
REMOVE_STOPWORDS = True

# Query processing
USE_STEMMER = True
USE_LEMMATIZER = True
USE_PRF = True

# Result ranking
PHRASAL_WEIGHT = 1.7    # the higher it is, the more weight phrasal results has
PRIORITY_WEIGHT = 1.1   # the higher it is, the more weight initial results has
FILTER_STRENGTH = 1.2   # the smaller it is, the more forgiving is the filter (thus more results)

### INDEX ###
DOCUMENT_LENGTH_KEYWORD = "LENGTH"
IMPT_KEYWORD = "IMPORTANT"
SPIMI_BLOCK_SIZE = 1000
POSTING_DIR = "temp_postings_result_dir/"

### QUERY KEYWORD ###
AND_KEYWORD = "AND"
DOUBLE_QUOTE_KEYWORD = "\""
QUERY_KEYWORDS = [AND_KEYWORD, DOUBLE_QUOTE_KEYWORD]

### QUERY EXPANSION ###
EXPAND_NUM_OF_SYNONYMS = 4

### QUERY PSEUDO RELEVANCE FEEDBACK ###
PRF_NUM_OF_RESULTS = 30
PRF_NUM_OF_WORDS_PER_DOC = 5 # requires re-indexing

### PHRASAL QUERIES ###
DOC_ID = 0
TF = 1
POS_LIST = 2
