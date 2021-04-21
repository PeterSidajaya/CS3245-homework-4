This is the README file for A0172307L-A0184586M-A0168113L-A0170766X submission
E0196721@u.nus.edu, E0313575@u.nus.edu, E0201877@u.nus.edu, E0176546@u.nus.edu

== Python Version ==

We're using Python Version <3.8.7> for this assignment.

== General Notes about this assignment ==

INDEXING

The indexing process is contained in index.py, with spimi.py, index_helper.py and word_processing.py containing the helper functions.
First, we do word_tokenization, followed by removing punctuation (as suggested in the forum), then stem it.
The process continues by calculating the doc_freq of each term from the document list and save the postings list (doc ID, term freq) to postings.txt.
We also need to store information at indexing time about the document length, in order to do document normalization.

SEARCHING

The searching process is contained in search.py, with query.py containing the helper functions.
In the query process, our purpose is to calculate cosine score shown in the lecture that will help us rank
our list of documents. We gather the precomputed data from INDEXING to do our searching. First, we
calculate and precompute the tf_idf score for query list for faster multiplication later for cosine score. Next for
each query term t, we calculate the tf score of the possible document id, normalize it and store it in a dictionary. 
Finally we do cross multiplication between our query_vector with document_vector to yield our cosine score.
Maintain the top 10 score of the document_id for our final results.

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

- CS3245 Piazza forum for comparing results of our search engine