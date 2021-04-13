import pickle
import os

def invert(word_list, dictionary_file, posting_file):
    """invert a word stream to dictionary and posting files

    Args:
        word_list (list): word stream 
        dictionary_file (str): address of the dictionary file
        posting_file (str): address of the posting file
    """
    dictionary = {}
    for word, posting in word_list:
        if word not in dictionary:
            posting_list = [posting]
            dictionary[word] = posting_list
        else:
            posting_list = dictionary[word]
        
        if posting_list[-1] != posting:
            posting_list.append(posting) # this will also modified the original dictionary
    
    dictionary_file = open(dictionary_file, 'wb')
    posting_file = open(posting_file, 'wb')
    # these two lines are for clarity purposes only
    dictionary_file.truncate(0)
    posting_file.truncate(0)

    for word, posting_list in dictionary.items():
        pointer = posting_file.tell()
        pickle.dump(posting_list, posting_file)
        dictionary[word] = (len(posting_list), pointer)

    pickle.dump(dictionary, dictionary_file)


def merge_files(dictionary_file_1_address, posting_file_1_address, dictionary_file_2_address, posting_file_2_address,
    output_dictionary_address, output_posting_address):
    """merge two pairs of dictionary and posting files

    Args:
        dictionary_file_1_address (str): address of the first dictionary file
        posting_file_1_address (str): address of the first posting file
        dictionary_file_2_address (str): address of the second dictionary file
        posting_file_2_address (str): address of the second posting file
        output_dictionary_address (str): address of the output dictionary file
        output_posting_address (str): address of the output posting file
    """
    infile = open(dictionary_file_1_address, 'rb')
    dictionary_file_1 = pickle.load(infile)
    infile.close()
    infile = open(dictionary_file_2_address, 'rb')
    dictionary_file_2 = pickle.load(infile)
    infile.close()

    posting_file_1 = open(posting_file_1_address, 'rb')
    posting_file_2 = open(posting_file_2_address, 'rb')

    dictionary_output = open(output_dictionary_address, 'wb')
    posting_output = open(output_posting_address, 'wb')
    # these two lines are for clarity purposes only
    dictionary_output.truncate(0)
    posting_output.truncate(0)

    dictionary = {}

    for word in dictionary_file_1.keys():
        posting_file_1.seek(dictionary_file_1[word][1])
        posting_data_1 = pickle.load(posting_file_1)
        if word in dictionary_file_2:
            posting_file_2.seek(dictionary_file_2[word][1])
            posting_data_2 = pickle.load(posting_file_2)
            posting_list = merge_lists(posting_data_1, posting_data_2)
        else:
            posting_list = posting_data_1
        pointer = posting_output.tell()
        pickle.dump(posting_list, posting_output)
        dictionary[word] = (len(posting_list), pointer)
    
    # for terms in dictionary_file_2 but not in dictionary_file_1
    for word in dictionary_file_2.keys():
        if word not in dictionary:
            posting_file_2.seek(dictionary_file_2[word][1])
            posting_list = pickle.load(posting_file_2)
            pointer = posting_output.tell()
            pickle.dump(posting_list, posting_output)
            dictionary[word] = (len(posting_list), pointer)
    
    pickle.dump(dictionary, dictionary_output)
    # remove the old files
    os.remove(dictionary_file_1_address)
    os.remove(dictionary_file_2_address)
    os.remove(posting_file_1_address)
    os.remove(posting_file_2_address)


def merge_lists(list_1, list_2):
    """merge two lists into one
    """
    i = 0
    j = 0
    list = []
    while i < len(list_1) and j < len(list_2):
        if list_1[i] < list_2[j]:
            list.append(list_1[i])
            i += 1
        elif list_1[i] > list_2[j]:
            list.append(list_2[j])
            j+= 1
        elif list_1[i] == list_2[j]:
            list.append(list_1[i])
            i += 1
            j += 1
    list.extend(list_1[i:])
    list.extend(list_2[j:])
    return list

