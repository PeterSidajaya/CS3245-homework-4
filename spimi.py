def count_word(token_list):
    """convert a stream of tokens into list of dictionary of term, frequency pairs

    Args:
        token_list (list): list of tokens

    Returns:
        dict: dictionary of term, frequency pairs
    """
    count = {}
    for term in token_list:
        if term not in count:
            count[term] = 1
        else:
            count[term] += 1
    return count
