def my_function(X, Y): 
    return "Hello %s %s!" % (X, Y)

def string_py2_ast(string):
    import ast
    
    tree = ast.parse(string)
    return tree

def ast_py2_string(string):
    import ast
    import astor
    
    string = astor.to_source(tree)
    return string