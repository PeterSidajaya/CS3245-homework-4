from index_helper import get_word_list
import nltk

def stem_word(word):
	"""
	Stemming stub - should perform case folding then stemming.
	"""
	stemmer = nltk.PorterStemmer()
	return stemmer.stem(word.lower())


def get_phrasal_query_doc_id(query_string, dictionary, posting_file):
	""" 
	Takes a query string and returns the value of the phrasal query.
	
	Format of the query string will be "a b" or "a b c".

	Arguments:
		query_string 	The query.
	Returns:
		list[docId] 	A list of integers representing the doc Ids.
	"""
	print("Query: \"{}\" - ".format(query_string), end="")
	words = query_string.split()
	words = list(map(lambda word: stem_word(word), words))

	if len(words) == 2:
		word_list_1 = get_word_list(words[0], dictionary, posting_file)
		word_list_2 = get_word_list(words[1], dictionary, posting_file)
		
		print("Two words: {}, {}".format(words[0], words[1]))
		return two_word_phrasal_query(word_list_1, word_list_2)
	elif len(words) == 3:
		word_list_1 = get_word_list(words[0], dictionary, posting_file)
		word_list_2 = get_word_list(words[1], dictionary, posting_file)
		word_list_3 = get_word_list(words[2], dictionary, posting_file)

		print("Three words: {}, {}, {}".format(words[0], words[1], words[2]))
		return three_word_phrasal_query(word_list_1, word_list_2, word_list_3)
	# Just handling an edge case
	elif len(words) == 1:
		return get_word_list(words[0], dictionary, posting_file)
	# The phrasal query with more than 3 words is illegal
	else:
		return []


def two_word_phrasal_query(word_list_1, word_list_2):
	"""
	Given two words, return the docId's which result from the phrasal query.

	Arguments:
		word1 		The first word in the phrase
		word2 		The second word in the phrase
	Returns:
		list[docId] A list of integers representing the doc Ids containing a phrase
	"""
	DOC_ID, TF, POS_LIST = 0, 1, 2

	idx1, idx2 = 0, 0
	result = []
	while idx1 < len(word_list_1) and idx2 < len(word_list_2):
		# The DOC_ID is the same at our two list pointers
		if word_list_1[idx1][DOC_ID] == word_list_2[idx2][DOC_ID]:
			# If the document contains the phrase, add it to the result
			if two_list_phrasal_query(word_list_1[idx1][POS_LIST],
								   word_list_2[idx2][POS_LIST]):
				result.append(word_list_1[idx1][DOC_ID])
			idx1 += 1
			idx2 += 1
		# The DOC_ID's are different - increment the lesser one
		elif word_list_1[idx1][DOC_ID] < word_list_2[idx2][DOC_ID]:
			idx1 += 1
		else:
			idx2 += 1

	return result


def two_list_phrasal_query(list1, list2, complete=False):
	"""
	Given two lists of position indices, return true if there exists x where list1 contains x and list2 contains (x+1).
	
	If complete is True, return a full list of indices instead.

	Arguments:
		list1		The list of positions containing the first word in a phrase
		list2 		The list of positions containing the second word in a phrase
		complete	If this is False, we return a boolean, otherwise a list[docId]
	Returns:
		bool 		The phrase exists in the two lists
	or
		list[docId] The list of positions x, where word x in the document is the phrase "word1 word2"
	"""
	idx1, idx2 = 0, 0
	result = []

	while idx1 < len(list1) and idx2 < len(list2):
		# The phrase is found
		if list1[idx1] + 1 == list2[idx2]:
			if not complete:
				return True
			result.append(list1[idx1])
			idx1 += 1
			idx2 += 1

		elif list1[idx1] >= list2[idx2]:
			idx2 += 1
		else:
			idx1 += 1

	return result if complete else False


def three_word_phrasal_query(word_list_1, word_list_2, word_list_3):
	"""
	Given two words, return the docId's which result from the phrasal query.

	Arguments:
		word1 		The first word in the phrase
		word2 		The second word in the phrase
	Returns:
		list[docId] A list of integers representing the doc Ids containing a phrase
	"""
	DOC_ID, TF, POS_LIST = 0, 1, 2

	idx1, idx2, idx3 = 0, 0, 0
	result = []
	while idx1 < len(word_list_1) and idx2 < len(word_list_2) and idx3 < len(word_list_3):
		# The DOC_ID is the same at our two list pointers
		if word_list_1[idx1][DOC_ID] == word_list_2[idx2][DOC_ID] and\
			word_list_1[idx1][DOC_ID] == word_list_3[idx3][DOC_ID]:
			# If the document contains the phrase, add it to the result
			if three_list_phrasal_query(word_list_1[idx1][POS_LIST],
								   word_list_2[idx2][POS_LIST],
								   word_list_3[idx3][POS_LIST]):
				result.append(word_list_1[idx1][DOC_ID])
			idx1 += 1
			idx2 += 1
			idx3 += 1
		# The DOC_ID's are different - increment the least one
		elif word_list_1[idx1][DOC_ID] <= word_list_2[idx2][DOC_ID] and\
			word_list_1[idx1][DOC_ID] <= word_list_3[idx3][DOC_ID]:
			idx1 += 1
		elif word_list_2[idx2][DOC_ID] <= word_list_1[idx1][DOC_ID] and\
			word_list_2[idx2][DOC_ID] <= word_list_3[idx3][DOC_ID]:
			idx2 += 1
		else:
			idx3 += 1

	return result


def three_list_phrasal_query(list1, list2, list3, complete=False):
	"""
	Given two lists of position indices, return true if there exists x where list1 contains x,
		list2 contains (x+1), and list3 contains (x+2).
	
	If complete is True, return a full list of indices instead.

	Arguments:
		list1		The list of positions containing the first word in a phrase
		list2 		The list of positions containing the second word in a phrase
		list3 		The list of positions containing the third word in a phrase
		complete	If this is False, we return a boolean, otherwise a list[docId]
	Returns:
		bool 		The phrase exists in the two lists
	or
		list[docId] The list of positions x, where word x in the document is the phrase "word1 word2 word3"
	"""
	idx1, idx2, idx3 = 0, 0, 0
	result = []

	while idx1 < len(list1) and idx2 < len(list2) and idx3 < len(list3):
		# The phrase is found
		if list1[idx1] + 1 == list2[idx2] and list1[idx1] + 2 == list3[idx3]:
			if not complete:
				return True

			result.append(list1[idx1])
			idx1 += 1
			idx2 += 1
			idx3 += 1

		# idx2 should be ahead of idx1
		elif list1[idx1] >= list2[idx2]:
			idx2 += 1
		# idx3 should be the furthest ahead
		elif list1[idx1] >= list3[idx3] or list2[idx2] >= list3[idx3]:
			idx3 += 1
		# They are in increasing order, so increment idx1
		else:
			idx1 += 1

	return result if complete else False
