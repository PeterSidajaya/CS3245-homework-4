from enum import Enum
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet

from query_util import QueryType
from constants import *

def expand_clause(expression: str):
    """Apply query expansion to the given expression.

    We first tokenize the expression, then we add the top k synonyms for each word in the expression to expand the expression.

    Args:
        expression (str): The expression to expand
    Returns:
        str: The expanded expression
    """
    # Tokenise and get all possible synonyms
    token_list = word_tokenize(expression)
    synsets_token = get_synsets(token_list)   

    expanded_tokens = []
    for i in range(len(synsets_token)):
        # The first synonym is always the word itself, so add 1 more
        synonyms = get_top_k_synonyms(synsets_token[i], EXPAND_NUM_OF_SYNONYMS + 1)

        # Make sure original word is included, add as first element
        synonym_names = [synonym.lemma_names()[0] for synonym in synonyms]
        if (token_list[i] not in synonym_names):
            synonym_names.insert(0, token_list[i])

        # Concat everything
        expanded_token = ' '.join(synonym_names)
        expanded_tokens.append(expanded_token)
    
    return ' '.join(expanded_tokens)

############ HELPERS ############

def pos_to_wordnet(tag):
    """Replaces the Part-of-Speech(POS) tag with wordnet tag, to be compatible
       with synset format of word.nn.pos. 
    
    Args:
        tag (str): The POS tag
    Returns:
        A wordnet tag corresponding to the provided tag
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('V'):
        return wordnet.VERB
    return None

def get_synsets(tokens):
    """Given a list of tokens, get the synonyms for each token.
    
    Returns a list of list, where each element is the synonyms of the corresponding token in tokens.
    Synonyms are in synsets format of word.nn.pos.

    e.g. Given expression 'running dog', it will be tokenised to
         'running' and 'dog', then stemmed to possibly 'run' and 'dog'.

         The final output will be [[list of synonyms of 'run'...], [list of synonyms of 'dog'...]]

         Actual sample output is [[Synset('run.v.01'), Synset('scat.v.01'), Synset('operate.v.01'), ...],
                                  [Synset('dog.n.01'), Synset('frump.n.01'), Synset('cad.n.01'), ...]]
    Args:
        tokens (str): The list of tokens
    Returns
        list(list): List of list of synsets
    """
    tagged = pos_tag(tokens)
    synsets = []

    for token in tagged:
        # Assign tag, whether its noun/verb/etc
        word, tag = token
        wn_tag = pos_to_wordnet(tag)
        if not wn_tag:
            continue

        # Format is to synset format, remove duplicate and add it to the list
        synsets.append(remove_duplicate_synsets(wordnet.synsets(word, pos=wn_tag)))

    return synsets

def remove_duplicate_synsets(synsets):
    """Remove duplicate from synsets. 
    
    Synsets must be in format of word.nn.pos.
    Duplicate is detected from its lemmas_names, and not from its nn and pos.

    Order is preserved.

    e.g. Given synset of [Synset('dog.n.01'), Synset('frump.n.01'), Synset('dog.n.02'), Synset('dog.v.01')],
         returns [Synset('dog.n.01'), Synset('frump.n.01')]
    
    Args:
        synsets (list): List of synsets
    Returns:
        list of synsets with duplicates removed
    """
    words_encountered = {}
    unique_synsets = []

    for synset in synsets:
        word_name = synset.lemma_names()[0]
        if word_name in words_encountered:
            continue
        
        words_encountered[word_name] = True
        unique_synsets.append(synset)

    return unique_synsets

def get_top_k_synonyms(synsets, k: int):
    """Extract top k synonyms with the highest similarity score with the first synset. 
    
    We compare against the first synsets as synsets are ordered by frequency, 
    so first element is the most probable word w/o context.

    Will return at most k synonyms.

    For various similarity functions, please refer to:
    https://www.nltk.org/howto/wordnet.html#similarityery

    Args:
        synsets (list): A list of synsets
        k (int): The number of synonyms to extract
    Returns:
        list of synonyms
    """
    if (not synsets):
        return []

    # Syn_sets are ordered by frequency, so first element is the most probable word w/o context
    syn_to_compare = synsets[0]

    # Create a new list where each element is (syn_set, similarity score)
    sim_score = [(synsets[i], syn_to_compare.wup_similarity(synsets[i])) for i in range(len(synsets))]

    # Sort from highest to lowest score
    sorted(sim_score, key=lambda syn: syn[1] if syn[1] != None else 0, reverse=True)
    
    # Return top k synonyms
    return list(map(lambda x: x[0], sim_score[:k]))
