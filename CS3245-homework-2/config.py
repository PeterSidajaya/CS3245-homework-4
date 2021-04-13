import math

# check if the given posting_list has next pointer
def has_skip(pointer, jump, total_length):
    if (pointer % jump == 0 and pointer + jump < total_length):
        return True
    
    return False


# function that convert list to pointer_list
# pointer_list is in the form of list of tuples where the tuples contain (docID, has_skip_pointer, skip_pointer_location)
# 
# example: [1,5,8,9,10,13,15]
# left/right : [(1, True, 2), (5, False, None), (8, True, 4), (9, False, None),
#               (10, True, 6), (13, False, None), (15, False, None)]
def make_pointer(posting_list):
    jump = int(math.sqrt(len(posting_list)))
    result = []

    for i in range(len(posting_list)):
        if (has_skip(i, jump, len(posting_list))):
            result.append((posting_list[i], True, i + jump))
        else:
            result.append((posting_list[i], False, None))

    return result