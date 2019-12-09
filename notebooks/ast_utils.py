from collections import deque 
import ast
import astor
import importlib
import re

def to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
    
  
def get_square_index(s, i): 
    if s[i] != '[': 
        return -1
  
    d = deque() 
    for k in range(i, len(s)): 
        if s[k] == ']': 
            d.popleft() 

        elif s[k] == '[': 
            d.append(s[i]) 

        if not d: 
            return k 
    return -1

def get_paren_index(s, i): 
    if s[i] != '(': 
        return -1
  
    d = deque() 
    for k in range(i, len(s)): 
        if s[k] == ')': 
            d.popleft() 

        elif s[k] == '(': 
            d.append(s[i]) 

        if not d: 
            return k 
    return -1



p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')


def canonicalize_code(code):
    code = code.strip()
    if p_elif.match(code):
        code = 'if True: pass\n' + code

    if p_else.match(code):
        code = 'if True: pass\n' + code

    if p_try.match(code):
        code = code + 'pass\nexcept: pass'
    elif p_except.match(code):
        code = 'try: pass\n' + code
    elif p_finally.match(code):
        code = 'try: pass\n' + code

    if p_decorator.match(code):
        code = code + '\ndef dummy(): pass'

    if code[-1] == ':':
        code = code + 'pass'

    return code


def de_canonicalize_code(code, ref_raw_code):
    ref_raw_code = ref_raw_code.strip()
    if code.endswith('def dummy():\n    pass'):
        code = code.replace('def dummy():\n    pass', '').strip()

    if p_elif.match(ref_raw_code):
        # remove leading if true
        code = code.replace('if True:\n    pass', '').strip()
    elif p_else.match(ref_raw_code):
        # remove leading if true
        code = code.replace('if True:\n    pass', '').strip()

    # try/catch/except stuff
    if p_try.match(ref_raw_code):
        code = code.replace('except:\n    pass', '').strip()
    elif p_except.match(ref_raw_code):
        code = code.replace('try:\n    pass', '').strip()
    elif p_finally.match(ref_raw_code):
        code = code.replace('try:\n    pass', '').strip()

    # remove ending pass
    if code.endswith(':\n    pass\n') or code.endswith(':\n    pass'):
        code = code.replace('\n    pass', '').strip()

    return code


def dump_to_ast(dump_string):
    first_paren = dump_string.index("(")
    node_string = dump_string[:first_paren]
#     print(1, "creating node type:", node_string)
    module = importlib.import_module("ast")
    class_ = getattr(module, node_string)
    node = class_()
    inner_paren = dump_string[first_paren+1:-1]
    
    # handle variable length attributes
    while len(inner_paren) > 0:
#         print(2, "Attr string so far:", inner_paren)
        equal_idx = inner_paren.index("=")
        attr_string = inner_paren[:equal_idx]
#         print(3, "attribute name:", attr_string)
        
        # 5 cases: None, num, string, list, node
        # first check if None
#         print(4, "checking None:", inner_paren[equal_idx+1:equal_idx+5])
#         print(5, "checking number:", inner_paren[equal_idx+1], inner_paren[equal_idx+1].isnumeric())
#         print(6, "checking Quote or DQuote:", inner_paren[equal_idx+1] == "'" or inner_paren[equal_idx+1] == '"')
        
        if inner_paren[equal_idx+1:equal_idx+5] == "None":
            setattr(node, attr_string, None)
            inner_paren = inner_paren[equal_idx + 7:]
        
        elif inner_paren[equal_idx+1:equal_idx+5] == "True":
            setattr(node, attr_string, True)
            inner_paren = inner_paren[equal_idx + 7:]
            
        elif inner_paren[equal_idx+1:equal_idx+6] == "False":
            setattr(node, attr_string, False)
            inner_paren = inner_paren[equal_idx + 8:]
            
        # check if first character is a number
        elif inner_paren[equal_idx+1].isnumeric():
            number_finder = re.compile("([0-9]+(.?[0-9]*))( *([a-zA-Z]+))*")
            groups = number_finder.match(inner_paren[equal_idx+1:]).groups()
#             print(6, "Number groups identified:", groups)
            num = to_num(groups[0])
#             print(7, "Number:", num)
            setattr(node, attr_string, num)
            inner_paren = inner_paren[equal_idx + len(groups[0])+2:]
        
        # check if first Character is quote or double quote to see if it's a string
        elif inner_paren[equal_idx+1] == "'" or inner_paren[equal_idx+1] == '"':
            string_matcher = re.compile("['\"](.*?)['\"]")
            string = string_matcher.findall(inner_paren[equal_idx:])[0]
            setattr(node, attr_string, string)
            inner_paren = inner_paren[equal_idx + len(string)+5:]
        
        # deal with list
        elif inner_paren[equal_idx+1] == "[":
            list_last_square = get_square_index(inner_paren, equal_idx+1)
            list_string = inner_paren[equal_idx+2:list_last_square]
            
            elem_list = []
            while len(list_string) > 0:
#                 print(8, "List string so far:", list_string)
                value_first_paren = list_string.index("(")
                value_last_paren = get_paren_index(list_string, value_first_paren)
                list_value_ast = dump_to_ast(list_string[:value_last_paren+1])
                elem_list.append(list_value_ast)
                list_string = list_string[value_last_paren+3:]
            setattr(node, attr_string, elem_list)
            inner_paren = inner_paren[list_last_square+3:]
        
        # deal with node
        else:
            value_first_paren = inner_paren.index("(")        
            value_last_paren = get_paren_index(inner_paren, value_first_paren)
#             print(9,"handling Node for: ",inner_paren[equal_idx+1:value_last_paren+1])
            value_ast = dump_to_ast(inner_paren[equal_idx+1:value_last_paren+1])
#             print(10, f"obtained AST for{attr_string}:", value_ast)
            setattr(node, attr_string, value_ast)
            inner_paren = inner_paren[value_last_paren+3:]
    
    
    return node