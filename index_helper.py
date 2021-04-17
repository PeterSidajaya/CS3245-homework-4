import pickle

def index_text(token_list):
    """
    Convert a stream of tokens into a dictionary of term, (frequency, position list) pairs

    Args:
        token_list (list): list of tokens

    Returns:
        dict: dictionary of term, (frequency, position list) pairs
    """
    dict = {}
    index = 0
    for term in token_list:
        if term not in dict:
            dict[term] = (1, [index,])
        else:
            count, position_list = dict[term]
            count += 1
            position_list.append(index)
            dict[term] = (count, position_list)
        index += 1
    return dict

def get_list(term, dictionary, posting_file):
    """
    Returns you the posting list

    Args:
        term (str): the term you want to find the posting list of
        posting_file (python file object): use this format -> open(filename, 'rb')

    Returns:
        list: [(doc Id, term frequency, position list), ...]
    """

    pointer = dictionary[term][1]
    posting_file.seek(pointer)
    posting_list = pickle.load(posting_file)
    return posting_list
