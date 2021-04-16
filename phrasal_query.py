import nltk

def Stem(word):
	"""
	Stemming stub - should perform case folding then stemming.
	"""
	stemmer = nltk.PorterStemmer()
	return stemmer.stem(word.lower())

def GetList(word):
	"""
	Retrieval Stub - Getting the lsit of position lists for a word.

	These should be in the format 
		[[docId1, termFreq1, posList1], [docId2, termFreq2, posList2], ...]
	"""
	return []

def phrasal_query(queryString):
	""" 
	Takes a query string and returns the value of the phrasal query.
	
	Format of the query string will be "a b" or "a b c".

	Arguments:
		queryString 	The query.
	Returns:
		list[docId] 	A list of integers representing the doc Ids.
	"""
	print("Query: \"{}\" - ".format(queryString), end="")
	words = queryString.split()
	words = list(map(lambda word: Stem(word), words))

	if len(words) == 2:
		print("Two words: {}, {}".format(words[0], words[1]))
		return two_word_phrasal_query(words[0], words[1])
	elif len(words) == 3:
		print("Three words: {}, {}, {}".format(words[0], words[1], words[2]))
		return three_word_phrasal_query(words[0], words[1], words[2])
	else:
		return []

def two_word_phrasal_query(word1, word2):
	"""
	Given two words, return the docId's which result from the phrasal query.

	Arguments:
		word1 		The first word in the phrase
		word2 		The second word in the phrase
	Returns:
		list[docId] A list of integers representing the doc Ids containing a phrase
	"""
	DOC_ID, TF, POS_LIST = 0, 1, 2

	listOfTriples1 = GetList(word1)
	listOfTriples2 = GetList(word2)

	idx1, idx2 = 0, 0
	result = []
	while idx1 < len(listOfTriples1) and idx2 < len(listOfTriples2):
		# The DOC_ID is the same at our two list pointers
		if listOfTriples1[idx1][DOC_ID] == listOfTriples[idx2][DOC_ID]:
			# If the document contains the phrase, add it to the result
			if two_list_phrasal_query(listOfTriples1[idx1][POS_LIST],
								   listOfTriples2[idx2][POS_LIST]):
				result.append(listOfTriples1[idx1][DOC_ID])
			idx1 += 1
			idx2 += 1
		# The DOC_ID's are different - increment the lesser one
		elif listOfTriples1[idx1][DOC_ID] < listOfTriples[idx2][DOC_ID]:
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

def three_word_phrasal_query(word1, word2, word3):
	"""
	Given two words, return the docId's which result from the phrasal query.

	Arguments:
		word1 		The first word in the phrase
		word2 		The second word in the phrase
	Returns:
		list[docId] A list of integers representing the doc Ids containing a phrase
	"""
	DOC_ID, TF, POS_LIST = 0, 1, 2

	listOfTriples1 = GetList(word1)
	listOfTriples2 = GetList(word2)
	listOfTriples3 = GetList(word3)

	idx1, idx2, idx3 = 0, 0, 0
	result = []
	while idx1 < len(listOfTriples1) and idx2 < len(listOfTriples2) and idx3 < len(listOfTriples3):
		# The DOC_ID is the same at our two list pointers
		if listOfTriples1[idx1][DOC_ID] == listOfTriples[idx2][DOC_ID] and \
			listOfTriples1[idx1][DOC_ID] == listOfTriples[idx3][DOC_ID]:
			# If the document contains the phrase, add it to the result
			if two_list_phrasal_query(listOfTriples1[idx1][POS_LIST],
								   listOfTriples2[idx2][POS_LIST]):
				result.append(listOfTriples1[idx1][DOC_ID])
			idx1 += 1
			idx2 += 1
			idx3 += 1
		# The DOC_ID's are different - increment the least one
		elif listOfTriples1[idx1][DOC_ID] < listOfTriples2[idx2][DOC_ID] and\
			listOfTriples1[idx1][DOC_ID] < listOfTriples2[idx2][DOC_ID]:
			idx1 += 1
		elif listOfTriples2[idx2][DOC_ID] < listOfTriples1[idx1][DOC_ID] and\
			listOfTriples2[idx2][DOC_ID] < listOfTriples3[idx3][DOC_ID]:
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
