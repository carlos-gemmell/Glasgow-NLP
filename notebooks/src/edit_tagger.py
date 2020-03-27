import numpy as np
import random

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

def single_step_edits(s1, s2, token_insertions=-1, max_insertions=10, pad_token="<pad>"):
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
    
    insert_locations = [np.minimum(max_insertions, ins) for ins in insert_locations]
    
    return edit_commands, insert_locations, token_replacements
    
    
def perform_edits(s, edits, gen_tok_id=4):
    s = s.copy()
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
    
    
    
def lcs(X, Y): 
    m = len(X) 
    n = len(Y) 
    L = [[0 for x in range(n+1)] for x in range(m+1)] 
    
  
    # Following steps build L[m+1][n+1] in bottom up fashion. Note 
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]  
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0: 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1] + 1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
  
    # Following code is used to print LCS 
    index = L[m][n] 
    
    return index, L
    
def lcs_traceback(L, X, Y):
    '''
    L is the traceback matrix.
    X and Y are the strings
    '''
    actions = []
    i = len(X)
    j = len(Y)
    index = L[i][j] 
    while i > 0 and j > 0: 
  
        # If current character in X[] and Y are same, then 
        # current character is part of LCS 
        if X[i-1] == Y[j-1]: 
            i-=1
            j-=1
            index-=1
  
        # If not same, then find the larger of two and 
        # go in the direction of larger value 
        elif L[i-1][j] >= L[i][j-1]: 
            i-=1
            actions.append(("Delete", X[i],i))
        else: 
            j-=1
            actions.append(("Insert", Y[j],i))
    return actions

def bi_directional_traceback(start, target):
    if len(start) == 0:
        return list(set([("Insert", t, 0) for t in target]))
    if len(target) == 0:
        return list(set([("Delete", t, 0) for t in start]))
    if start == target:
        return [("None", None, None)]
    score, matrix = lcs(start, target) 
    actions = lcs_traceback(matrix, start, target)
    actions = [(i,j,min(len(start),k)) for i, j, k in actions]

    r_start = start[::-1]
    r_target = target[::-1]
    score, matrix = lcs(r_start, r_target) 
    reverse_actions = lcs_traceback(matrix, r_start,r_target)
    corrected_actions = [(i,j,max(0,len(start)-k-1)) for i, j, k in reverse_actions]

    return list(set(actions + corrected_actions))

def perform_edit(old_seq, edit):
    seq = list(old_seq)
    action, tok, pos = edit
    if action == "Insert":
        seq.insert(pos,tok)
    if action == "Delete" and pos < len(seq):
        del seq[pos]
    return seq

def random_edit(original_string, v_chars):
    seq = list(original_string)
    action = random.choice(["Insert", "Delete"])
    tok = random.choice(v_chars)
    pos = random.choice(list(range(len(seq)+1)))
    edit = (action, tok, pos)
    return perform_edit(seq, edit)