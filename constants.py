### CONFIG ###

# Query processing
USE_STEMMER = True
USE_LEMMATIZER = False
USE_PRF = True

# Result ranking
PHRASAL_WEIGHT = 1.7    # the higher it is, the more weight phrasal results has
PRIORITY_WEIGHT = 1.1   # the higher it is, the more weight initial results has
FILTER_STRENGTH = 1.3  # the smaller it is, the more forgiving is the filter (thus more results)

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
EXPAND_NUM_OF_SYNONYMS = 2

### QUERY PSEUDO RELEVANCE FEEDBACK ###
PRF_NUM_OF_RESULTS = 10
PRF_NUM_OF_WORDS_PER_DOC = 5
PRF_QUERY_VEC_WEIGHT = 0.1
PRF_QUERY_VEC_UPDATE_WEIGHT = 0.9
PRF_TIME_LIMIT = 8

### PHRASAL QUERIES ###
DOC_ID = 0
TF = 1
POS_LIST = 2

### QUERY MODES ###
NO_RANKING = 0
GET_RANKING = 1
GET_RANKING_AND_VECTORS = 2
