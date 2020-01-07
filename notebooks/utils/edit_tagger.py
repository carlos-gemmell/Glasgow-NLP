import numpy as np

def lev(matrix, i, j, compare):
    m00 = matrix[i-1, j-1]
    if not compare:
        m00 += 1
    
    m10 = matrix[i, j-1] + 1
    m01 = matrix[i-1, j] + 1
    
    matrix[i,j] = min([m00, m10, m01])

def build_matrix (tokens1, tokens2):
    tokens1 = ["_"] + tokens1
    tokens2 = ["_"] + tokens2
    m,n = len(tokens1), len(tokens2)
    
    lev_matrix = np.zeros([m,n])
    lev_matrix[0] = list(range(n))
    lev_matrix[:,0] = list(range(m))
    
    for i in range(1,m):
        for j in range(1,n):
            lev(lev_matrix, i, j, tokens1[i] == tokens2[j])
    
    return lev_matrix

# Delete, Insert, Replace, Keep
#
def tagger(matrix, i, j):
    steps = [matrix[i, j-1], matrix[i-1, j-1], matrix[i-1, j]]
    
    if i == 0:
        return i, j-1, "I"
    elif j == 0:
        return i-1, j, "D"
    elif matrix[i-1, j-1] == min(steps):
        if matrix[i-1, j-1] == matrix[i,j]:
            return i-1, j-1, "K"
        else:
            return i-1, j-1, "R"
    elif matrix[i, j-1] == min(steps):
        return i, j-1, "I"
    elif matrix[i-1, j] == min(steps):
        return i-1, j, "D"
        

def get_tags(matrix):
    m,n = matrix.shape
    sequence = []
    m -= 1; n -= 1
    
    while not (m == 0 and n == 0):
        m,n, tag = tagger(matrix, m,n)
        sequence.append(tag)
        
    sequence.reverse()
    return sequence

def single_step_edits(s1, s2, token_insertions=-1, pad_token="<pad>"):
    '''
    This function returns a sequence of edit commands to convert s1 to s1' that is closer to s2.
    The outputs are lists of the same length as s1.
    
    Args:
        s1: List, string list to start edits from
        s2: List, target string list string
        token_insertions: Int, cap on number of Insert (I) tokens to give to string, 
                          if -1 insert_locations will autofill to correct levenstein distance
    Outputs:
        edit_commands: list with only Keep (K), Delete (D), Replace (R) opperations are allowed
        insert_locations: list with number of generic tokens to insert after each indexed position
        token_replacements: list containing new replacement tokens in "R" is present, "<N>" otherwise
    '''
    matrix = build_matrix(s1, s2)
    s2 = s2.copy()
    tags = get_tags(matrix)
    
    if tags[0] == "I":
        raise Exception('Carlos\' code error: Cannot insert generic token at first position, only after')
        
    
    edit_commands = [x for x in tags if x != "I"]
    assert len(edit_commands) == len(s1)
    
    token_replacements = []
    for i in range(len(tags)):
        if tags[i] == "R":
            token_replacements.append(s2[i])
        elif tags[i] == "K":
            token_replacements.append(pad_token)
        elif tags[i] == "D":
            token_replacements.append(pad_token)
            s2[i:i] = ["<temp_gen>"]
        
    
    insert_locations = []
    insertion_count = 0
    for i in range(len(tags)):
        if tags[i] != "I":
            insert_locations.append(0)
        else:
            insert_locations[-1] += 1
    
    if token_insertions > 0:
        insert_locations = [token_insertions if x!=0 else 0 for x in insert_locations]
    
    return edit_commands, insert_locations, token_replacements
    
    
def perform_edits(s, edits, gen_tok_id=4):
    edit_commands, insert_locations, token_replacements = edits
    
    # replace tokens first and flag for deletion
    for i in range(len(token_replacements)):
        if edit_commands[i] == "R":
            s[i] = token_replacements[i]
        if edit_commands[i] == "D":
            s[i] = "<DELETE_ME>"
    
    ofset = 0
    
    for i in range(len(insert_locations)):
        if insert_locations[i] > 0:
            insertion_index = i+1+ofset
            s[insertion_index:insertion_index] = [gen_tok_id] * insert_locations[i]
            ofset += insert_locations[i]
    
    s = [x for x in s if x != "<DELETE_ME>"]
    
    return s
    
    
    
    
    