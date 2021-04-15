from enum import Enum
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import re
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

WHITESPACE_REGEX = "\s+"

class QueryType(Enum):
    FREE_TEXT = 0
    PHRASAL = 1

def penn_to_wordnet(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('V'):
        return wordnet.VERB
    return None

def tokenise_and_get_synsets(expression: str, lemmatzr):
    tagged = pos_tag(word_tokenize(expression))
    syn_sets = []

    for token in tagged:
        # Assign tag, whether its noun/verb/etc
        word, tag = token
        wn_tag = penn_to_wordnet(tag)
        if not wn_tag:
            continue

        # Apply stemmer and format it to synset format word.nn.pos
        lemma = lemmatzr.lemmatize(word, pos=wn_tag)
        syn_sets.append(wordnet.synsets(lemma, pos=wn_tag))

    return syn_sets

def get_top_k_synonyms(syn_sets, k: int):
    # Syn_sets are ordered by frequency, so first element is the most probable word w/o context
    syn_to_compare = syn_sets[0]
    print("Extracting from the syn_sets of {}".format(syn_sets))
    print("Comparing against {}".format(syn_to_compare))

    # Create a new list where each element is (syn_set, similarity score)
    sim_score = [(syn_sets[i], syn_to_compare.path_similarity(syn_sets[i])) for i in range(len(syn_sets))]

    # Sort from highest to lowest score
    sorted(sim_score, key=lambda x: x[1] if x[1] != None else 0, reverse=True)
    
    # Return top k synonyms
    return list(map(lambda x: x[0], sim_score[:k]))

def expand_clause(clause: (str, QueryType), lemmatzr):
    # We dont expand phrasal clause
    if (clause[1] == QueryType.PHRASAL):
        return

    expression = clause[0]    
    token_syn_sets = tokenise_and_get_synsets(expression, lemmatzr)   
    print("ALL_SYN_SETS {}\n\n".format(token_syn_sets))
    print("Length of ALL_SYN_SETS {}".format(len(token_syn_sets)))

    for token_syn_set in token_syn_sets:
      synonyms = get_top_k_synonyms(token_syn_set, 3) 
      print("Expression: {}".format(token_syn_set[0]))
      print("Synonyms: {}".format(synonyms))
      print("---------\n\n")

test_clause = ("running dog", QueryType.FREE_TEXT)
lemmatzr = WordNetLemmatizer()

expand_clause(test_clause, lemmatzr)

# Sources
# https://stackoverflow.com/questions/27591621/nltk-convert-tokenized-sentence-to-synset-format
# https://stackoverflow.com/questions/59355529/is-there-any-order-in-wordnets-synsets
# https://www.nltk.org/howto/wordnet.html#similarity