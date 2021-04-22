This is the README file for A0172307L-A0184586M-A0168113L-A0170766X submission
E0196721@u.nus.edu, E0313575@u.nus.edu, E0201877@u.nus.edu, E0176546@u.nus.edu

== Python Version ==

We're using Python Version <3.8.7> for this assignment.

== General Notes about this assignment ==

=== INDEXING ===

The indexing process is contained in index.py, with spimi.py, index_helper.py and word_processing.py containing the helper functions.
We use SPIMI as the index is too large for a single pass. We create blocks of 1000 documents each, and merge them
in a binary merge after all blocks are created.

For the indexing of each document, we perform sanitization (word_processing.py) by first performing word_tokenization,
removing non-alphanumeric characters and stop words, then performing lemmatization then stemming (configurable).
The process continues by calculating the doc_freq of each term from the document list and save the postings list (doc ID, term freq) to postings.txt.
We also need to store information at indexing time about the document length, in order to do document normalization for tf-idf calculations.
We also store the top 5 most common words in the document to facilitate PRF.

=== SEARCHING ===

The searching process is started in search.py, calling query.py which contains the high level logic, that
calls the rest of our query processing functions.

We first split the query into subqueries (query_util.py) by splitting the query by "AND" keywords, 
and categorizing the subqueries' clauses into phrasal and free text queries. Clauses within subqueries are
implicitly treated with "OR" operations.

We then sanitize the clauses (with the same process as in indexing) and stem/lemmatize them (configurable) (word_processing.py).
We process each of the clauses and union (OR) the results of clauses in the subqueries, and intersect (AND) the results of subqueries.
The clauses are processed by free_text_search (free_text_query.py) or get_phrasal_query_doc_id (phrasal_query.py), depending
on how they were categorized.

    free_text_search is done by first getting the tf-idf vector of the query (ltc), as well as the the tf-idf scores for
    each of the documents in the index (scoring.py) (lnc, normalization factors were stored during indexing). We compute the 
    cosine scores with the query vector for each document vector and rank them. For ranking (scoring.py), we prune extremely
    low scores, and sort the documents in descending order of cosine scores.
        Ranking is an option - we can choose not to rank and simply return an unsorted list of potential doc_id's. As an 
        optimization, we do not rank when computing the clause/subquery results, and only rank after intersections of all subqueries.

    phrasal queries are done by first finding documents that contain all words in the phrase, and for those documents,
    we check if their posting lists contain the relevant consecutive indices, e.g. the phrase "a b c" should have some x, 
    such that x appears in posting_list_a, x+1 appears in posting_list_b, and x+2 appears in posting_list_c. 
    And if this is true we add the document to the result.

Query expansion is applied on each clauses, both for the phrasal and free text clause. For phrasal queries, we still separate
between (i) the results obtained through phrasal search on the original phrasal queries and (ii) the results obtained through search on the
expanded phrasal queries. An example use case on this design decision can be found under `EXPANSION ON PHRASAL QUERY` section.

After processing and intersecting all subqueries, we retrieve all unique words in the clauses and the query expansion
of each clause, and perform a final free text search on this "query_list". e.g. if our original query was '"a b" AND c d e AND "f g"',
we will perform the final free text search on query_list 'a b c d e f g' (plus query expansion).

    This final free_text_search will be done with a priority list. The priority list consists of all the documents that 
    passed the original AND intersections, and they are tagged with whether they were also the result of a phrasal query or free text query.
    We perform a similar tf-idf to our query_list first as in the regular free_text_search, but for scoring, we will add
    additional weight to those that were present in phrasal queries, and ensure that those in the priority
    list are given a score at least that of the average score in our query_list result. We finally sort by this weighted score.

This is the end of our query processing. We can then perform PRF on this query result, which is explained below,
but in our final submission we do not turn this on.

=== QUERY REFINEMENT ===

Query Expansion:
    For our query, we expand the query by adding the top K synonyms of each word in our original query to the query.

    The logic of expansion is found in query_expansion.py

Pseudo-Relevance Feedback (bonus):
    During indexing, we stored the 5 most common words (stop words removed) that occurred in each document in our dictionary.

    When we turn on PRF, we first perform our regular query processing pipeline once through. We then take 
    the results, and we extract all 5 most common words (excluding stopwords) from the top k (20) results. For all of the commmon
    words, we calculate the idf score for each word, and choose the words with highest idf scores.

    Afterwards, we perform another query search (using the regular processing pipeline) with the original query augmented with
    these words.

    In our final submission, this is DISABLED, because the most common words in the documents tend not to be related to the
    search query, and throw off the results.
    
    The logic is found in query_prf.py

=== EXPANSION ON PHRASAL QUERY ===

This design decision is derived from our observation on the given q1.txt from LumiNUS: with the query `quiet phone call` and
one of the expected result of document id 6807771. From the document 6807771 itself, the document has multiple "telephone call" phrases
instead of the `phone call`.

This implies that if we perform a query search on `quiet AND "phone call"`, document 6807771 will not appear as a result. However, we deem
that document 6807771 is still relevant as it contains many "telephone call" phrases, although will not ranked high as it does not exactly match
the given phrasal query "phone call". This lead us to still apply query expansion on the phrasal query "phone call" to still include document
6807771.

== Files included with this submission ==

index.py  : main file for indexing logic (mostly via spimi.py)
search.py : handles IO for queries and initiates query handling (via query.py)

Used in both indexing and search:
    constants.py:       : runtime constants and configuration settings
    word_processing.py  : functions to sanitize, lemmatize and stem text (for both indexing and searching)
    index_helper.py     : helpers to format token lists into dictionary entries, as well as to retrieve
                          posting lists from the posting file (used by both indexing and searching)

Used in indexing:
    spimi.py            : helper functions for indexing (writing blocks to files, and merging blocks)

Used in search:
    query.py            : high level logic for query handling
    phrasal_query.py    : handle phrasal queries
    free_text_query.py  : handle free text queries
    query_expansion.py  : perform query expansion using synonyms
    query_prf.py        : perform pseudo-relevance feedback on results from initial query handling to refine results
    query_util.py       : helpers for handling queries
    scoring.py          : perform scoring for documents based on tf-idf from queries, weighted with priority list
                          (priority is given to queries fulfilling AND clauses) and weightage to phrasal queries

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] I/We, A0172307L and A0184586M and A0168113L and A0170766X, certify that I/we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason: -

We suggest that we should be graded as follows: -

== References ==

- Lecture notes
- CS3245 Piazza forum (e.g. need to filter punctuation)
- NLTK website, stackoverflow (for query expansion and similarity between synonyms)
- Wikipedia (for variants of tf-idf/scoring system)
- Wikipedia Okapi BM25