import pickle

def index_text(token_list):
    """
    Convert a stream of tokens into a dictionary of term, (frequency, position list) pairs

    Args:
        token_list (list): list of tokens

    Returns:
        freq_pos_dict: dictionary of term, (term frequency, position list) pairs
    """
    freq_pos_dict = {}
    index = 0
    for term in token_list:
        if term not in freq_pos_dict:
            freq_pos_dict[term] = (1, [index,])
        else:
            count, position_list = freq_pos_dict[term]
            count += 1
            position_list.append(index)
            freq_pos_dict[term] = (count, position_list)
        index += 1
    return freq_pos_dict


def get_word_list(term, dictionary, posting_file):
    """Returns the posting list for a given term in the posting_file.

    Args:
        term (str): the term to find the posting list of
        dictionary (dict): The dictionary containing pointers for the terms in the posting_file
        posting_file (python file object): use this format -> open(filename, 'rb')

    Returns:
        list: [(doc Id, term frequency, position list), ...]
    """
    if (term not in dictionary):
        return []

    pointer = dictionary[term][1]
    posting_file.seek(pointer)
    posting_list = pickle.load(posting_file)
    return posting_list
