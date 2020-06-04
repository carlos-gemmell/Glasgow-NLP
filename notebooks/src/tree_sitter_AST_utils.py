from IPython.display import Image, display
from pyformlang.finite_automaton import EpsilonNFA, State, Symbol, Epsilon
from tree_sitter import Language, Parser, Node
import networkx as nx
import json
import autopep8
import sre_parse
import traceback


python_newline_statements = ['assert_statement',
 'break_statement',
 'class_definition',
 'comment',
 'continue_statement',
 'decorated_definition',
 'decorator',
 'delete_statement',
 'except_clause',
 'exec_statement',
 'else_clause',
 'expression_statement',
 'for_statement',
 'function_definition',
 'future_import_statement',
 'global_statement',
 'if_statement',
 'import_from_statement',
 'import_statement',
 'nonlocal_statement',
 'pass_statement',
 'print_statement',
 'raise_statement',
 'return_statement',
 'try_statement',
 'while_statement',
 'with_statement']

python_spaced_nodes = ["def", "if", "while", "for", "in", "not", "as", "return",
                      "continue", "break", "from", "import", "raise", "global",
                      "class", "else"]

def sub_str_from_coords(string, start, end):
    lines = string.split("\n")
    s_line, s_char = start
    e_line, e_char = end
    assert s_line == e_line
    return lines[s_line][s_char:e_char]

class Tree_Sitter_ENFA(EpsilonNFA):
    def __init__(self, member_name, grammar):
        """
        This class builds the NFA from a tree sitter grammar member.
        It enforces the rules by only providing node sequences that are valid according 
        to the grammar in the grammar.json.
        """
        super().__init__()
        self.prefix = 0
        self.externals = [x["name"] for x in grammar["externals"]]
        self.punct_dict = {",":"COMMA", '\\':r"\\"}
        self.node_types = grammar["rules"]
        self.start_state = State("START")
        self.match_state = State("MATCH")
        self.add_start_state(self.start_state)
        self.add_final_state(self.match_state)
        self.add_transitions(self.node_types.get(member_name), self.start_state, self.match_state)
        
    def plot_dot(self):
        graph = self.to_networkx()
        for edge in graph.edges.data():
            try:
                if edge[2]["label"] in self.punct_dict:
                    edge[2]["label"] = self.punct_dict[edge[2]["label"]]
            except:
                pass
        pdot = nx.drawing.nx_pydot.to_pydot(graph)
        png_str = pdot.write_png("/tmp/graph.png")
        display(Image(filename='/tmp/graph.png'))
        
    def add_transitions(self, member, enter_state, exit_state):
        self.prefix += 1
        
        if member == None:
            self.add_transition(enter_state, Epsilon(), exit_state)
            return
        
        if member["type"] == "BLANK":
            self.add_transition(enter_state, Epsilon(), exit_state)

        if member["type"] == "SYMBOL":
            if member["name"][0] == "_" and member["name"] not in self.externals:
                self.add_transitions(self.node_types[member["name"]], enter_state, exit_state)
            else:
                symbol = Symbol(member["name"])
                self.add_transition(enter_state, symbol, exit_state)

        if member["type"] == "STRING":
            symbol = Symbol(member["value"])
            self.add_transition(enter_state, symbol, exit_state)
#             memb = regex_to_member(member["value"])
#             self.add_transitions(memb, enter_state, exit_state)

        if member["type"] == "PATTERN":
#             symbol = Symbol("regex:"+member["value"])
            memb = regex_to_member(member["value"])
            self.add_transitions(memb, enter_state, exit_state)
#             self.add_transition(enter_state, symbol, exit_state)

        if member["type"] in ["FIELD", "PREC","PREC_LEFT","PREC_RIGHT","ALIAS", "TOKEN"]:
            self.add_transitions(member["content"], enter_state, exit_state)

        if member["type"] == "SEQ":
            prev_state = enter_state
            for i, SEQ_member in enumerate(member["members"]):
                next_state = State(f"SEQ{self.prefix}_St_{i}")

                self.add_transitions(SEQ_member, prev_state, next_state)
                prev_state = next_state
            self.add_transition(next_state, Epsilon(), exit_state)

        if member["type"] == "CHOICE":
            for i, CHOICE_member in enumerate(member["members"]):
                choice_state = State(f"CH{self.prefix}_St_{i}")
                self.add_transitions(CHOICE_member, enter_state, choice_state)
                self.add_transition(choice_state, Epsilon(), exit_state)

        if member["type"] == "REPEAT":
            self.add_transition(enter_state, Epsilon(), exit_state)
            self.add_transition(exit_state, Epsilon(), enter_state)
            self.add_transitions(member["content"], enter_state, exit_state)

        if member["type"] == "REPEAT1":
            self.add_transition(exit_state, Epsilon(), enter_state)
            self.add_transitions(member["content"], enter_state, exit_state)        
    
    def validate_sequence(self, seq_node_seq):
        current_states = [self.start_state]
        current_states = self.eclose_iterable(current_states)
        for seq_node in seq_node_seq:
            current_states = self._get_next_states_iterable(current_states,Symbol(seq_node))
            current_states = self.eclose_iterable(current_states)
        possible_transitions = set()
        for s in current_states:
            if s in self._transition_function._transitions:
                transitions = [str(x) for x in list(self._transition_function._transitions[s])]
                possible_transitions.update(set(transitions))

        try:
            possible_transitions.remove("epsilon")
        except:
            pass
        
        if any([self.is_final_state(x) for x in current_states]):
            possible_transitions.update(['<REDUCE>'])
        return possible_transitions


class StringTSNode():
    def __init__(self, node_type, children=[], text=""):
        self.type = node_type
        self.children = children
        self.text = text
        

class PartialNode():
    def __init__(self, parent, node_type, ENFA, node_builder, is_textual, text=""):
        self.parent = parent
        self.type = node_type
        self.ENFA = ENFA
        self.children = []
        self.text = text
        self.is_textual = is_textual
        self.node_builder = node_builder
    
        
    @property
    def is_complete(self):
        child_sequence = [child.type for child in self.children]
        children_expansions = self.ENFA.validate_sequence(child_sequence)
        text_expansions = self.ENFA.validate_sequence(list(self.text))
        children_complete = "<REDUCE>" in children_expansions
        text_complete = "<REDUCE>" in text_expansions
        return children_complete or text_complete
    
    @property
    def explicit_expansions(self):
        # make a check for the is_textual property since the requirements are different for it.
        if self.is_textual:
            char_sequence = list(self.text)
            char_expansions = self.ENFA.validate_sequence(char_sequence)
            token_expansions = self.ENFA.validate_sequence([self.text])        # this needs to be fixed
            char_expansions.update(token_expansions)
            return char_expansions
        else:
            child_sequence = [child.type for child in self.children]
            expansions = self.ENFA.validate_sequence(child_sequence)
            return expansions
    
    def is_valid_pattern_expansion(self, expansion_token):
        if self.is_textual and expansion_token != "<REDUCE>":                
            char_sequence = list(self.text+expansion_token)
            char_expansions = self.ENFA.validate_sequence(char_sequence)
            token_expansions = self.ENFA.validate_sequence([expansion_token])
            char_expansions.update(token_expansions)
            if char_expansions != set():
                return True
            return False
        else:
            if expansion_token in self.explicit_expansions:
                return True
            return False
    
    def expansions_mask(self, expansion_vocab):
        return [self.is_valid_pattern_expansion(token) for token in expansion_vocab]
    
    def add_expansion(self, expansion_token):
        is_valid = self.is_valid_pattern_expansion(expansion_token)
        assert is_valid, f"{expansion_token} is not currently a valid expansion for {self.type}. "+\
                                                            f"possible explicit ones: {self.explicit_expansions}"
        if self.is_textual:
            self.text += expansion_token
#             self.children.append(transition_pattern)
            return self
        else:
            new_child = self.node_builder.create(self, expansion_token, "")
            self.children.append(new_child)
            return new_child
        
class PartialTree():
    def __init__(self, root_node_type, node_builder):
        self.full_seq = [root_node_type]
        self.node_builder = node_builder
        self.root = node_builder.create(None, root_node_type)
        self.pointer = self.root
        
    def add_action(self,expansion_token):
        is_valid = self.pointer.is_valid_pattern_expansion(expansion_token)
        assert is_valid, f"{expansion_token} is not currently a valid expansion for {self.pointer.type}. "+\
                                                            f"possible explicit ones: {self.pointer.explicit_expansions}"
        
        if expansion_token == "<REDUCE>" and self.pointer.parent == None:
            print("Tree generation terminated")
            return None
    
        if expansion_token == "<REDUCE>":
            self.pointer = self.pointer.parent
        else:
            new_child = self.pointer.add_expansion(expansion_token)
            self.pointer = new_child
        self.full_seq.append(expansion_token)

        return self.pointer
    
    @property
    def pointer_explicit_expansions(self):
        self.pointer.explicit_expansions
    
        

class NodeBuilder():
    def __init__(self, grammar):
        self.grammar = grammar
        rule_node_types = grammar["rules"].keys()
        self.node_ENFAs = {node_type: Tree_Sitter_ENFA(node_type, self.grammar) for node_type in rule_node_types}
        self.vocab, self.grammar_paterns = get_grammar_vocab(grammar)
        
        for node_type in self.vocab:
            if node_type not in self.node_ENFAs:
                self.node_ENFAs[node_type] = Tree_Sitter_ENFA(None, self.grammar)   
    @staticmethod            
    def member_has_pattern(member):
        if member == None:
            return False
        elif member["type"] in ["PATTERN","STRING","BLANK"]:
            return True
        elif member["type"] in ["SEQ", "CHOICE"]:
#             print([NodeBuilder.member_has_pattern(sub_member) for sub_member in member["members"]])
#             print(member)
            return all([NodeBuilder.member_has_pattern(sub_member) for sub_member in member["members"]])
        elif member["type"] in ["FIELD", "PREC","PREC_LEFT","PREC_RIGHT","ALIAS", "TOKEN", "REPEAT", "REPEAT1"]:
            return NodeBuilder.member_has_pattern(member["content"])
        
        return False
    
    def create(self, parent, node_type, text=""):
        is_textual = self.member_has_pattern(self.grammar["rules"].get(node_type))
        return PartialNode(parent, node_type, self.node_ENFAs[node_type], self, is_textual, text)


class Code_Parser():
    def __init__(self, grammar, language="python"):
        Language.build_library('build/my-languages.so',[
                'src/tree-sitter/tree-sitter-javascript',
                'src/tree-sitter/tree-sitter-python'])
        
        LANGUAGE = Language('build/my-languages.so', language)
        
        self.grammar = grammar
        
        self.TS_parser = Parser()
        self.TS_parser.set_language(LANGUAGE)
        self.node_builder = NodeBuilder(self.grammar)
    
    def code_to_sequence(self, code_str):
        tree = self.TS_parser.parse(bytes(code_str, "utf8"))
        root_node = tree.root_node
        sequence = self.TSTree_to_sequence(root_node, code_str)
        return sequence
    
    def TSTree_to_sequence(self, TSNode, code_str):
        node_sequence = [TSNode.type]
        if TSNode.type == "string":
            node_text = sub_str_from_coords(code_str, TSNode.start_point, TSNode.end_point)[1:-1]
            node_sequence += ["_string_start",'"',"<REDUCE>"]
            node_sequence += ["_string_content",node_text,"<REDUCE>"]
            node_sequence += ["_string_end",'"',"<REDUCE>"]
        elif TSNode.children == []:
            node_text = sub_str_from_coords(code_str, TSNode.start_point, TSNode.end_point)
            if TSNode.type != node_text:
                node_sequence.append(node_text)
        elif TSNode.children != []:
            for child in TSNode.children:
                node_sequence += self.TSTree_to_sequence(child, code_str)
        node_sequence.append("<REDUCE>")
        return node_sequence
        
    def sequence_to_partial_tree(self, sequence):
        first_node = sequence[0]
        partial_tree = PartialTree(first_node, self.node_builder)
        try:
            for expansion in sequence[1:]:
                partial_tree.add_action(expansion)
        except Exception as e:
            print("ERROR!")
            traceback.print_exc()
            print("-------")
        return partial_tree 
    
    
class Node_Processor():
    def __init__(self, language="python"):
        self.counter = 0
        if language == "python":
            global python_statements
            self.new_line_statements = python_newline_statements
            global pytohn_non_space_nodes
            self.spaced_nodes = python_spaced_nodes
            
    def to_sequence(self, node, tokenize_fn):
        node_sequence = [node.type]
        if node.is_textual: #node.children == []:
            node_sequence.append(tokenize_fn(node.text))
        else:
            for child in node.children:
                node_sequence += self.to_sequence(child, tokenize_fn)
        node_sequence.append("<REDUCE>")
        return node_sequence
        
    def to_string(self, node, indent=""):
        if node.children == []:
            if node.text == "":
                return node.type
            else:
                return node.text
        else:
            child_text = ""
            if node.type == "block":
                indent += " "*4
                child_text += "\n"
            for i, child in enumerate(node.children):
                if child.type in self.new_line_statements:
                    child_text += indent
                child_text += self.to_string(child, indent) 
                if child.type in self.spaced_nodes:
                    child_text += " "
                if child.type in self.new_line_statements:
                    child_text += "\n" 
            return child_text
    
    def pretty_print(self, node):
        code_string = self.to_string(node, indent="")
        return autopep8.fix_code(code_string)
    
    def plot_graph_fill(self, OG, parent, node):
        self.counter += 1

        ident_node = f"{self.counter}_{node.type}"
        OG.add_node(ident_node)
        OG.add_edge(parent, ident_node)

        if node.is_textual:
            node_text = node.text
            self.counter += 1
            node_label = f"{self.counter}_{node_text}"
            OG.add_node(node_label)
            OG.add_edge(ident_node, node_label)
        else:
            for child in node.children:
                self.plot_graph_fill(OG, ident_node, child)
        return OG
    
    def plot(self, node, path="/tmp/graph.png"):
        OG=nx.OrderedGraph()
        OG.add_node('ROOT')
        graph = self.plot_graph_fill(OG, 'ROOT', node)
        pdot = nx.drawing.nx_pydot.to_pydot(graph)
        png_str = pdot.write_png(path)
        display(Image(filename=path))
        
        
        
def member_vocab(member):
    if member["type"] in ["REPEAT", "REPEAT1", "ALIAS", "FIELD", 
                          "PREC","PREC_LEFT","PREC_RIGHT", "TOKEN"]:
        return member_vocab(member["content"])
    
    if member["type"] in ["SEQ", "CHOICE"]:
        grammar_tokens = set()
        grammar_patterns = set()
        for child in member["members"]:
            new_tokens, new_patterns = member_vocab(child)
            grammar_tokens.update(new_tokens)
            grammar_patterns.update(new_patterns)
        return grammar_tokens, grammar_patterns
        
    if member["type"] in ["STRING"]:
        return set([member["value"]]), set()
    
    if member["type"] in ["SYMBOL"]:
        return set([member["name"]]), set()
    
    if member["type"] == "PATTERN":
        return set(), set([member["value"]])
    return set(), set()

def get_grammar_vocab(grammar):
    vocab=set(grammar["rules"].keys())
    externals = [x["name"] for x in grammar["externals"]]
    grammar_patterns = set()
    token_list = list(vocab)
    for token in token_list:
        new_tokens, new_patterns = member_vocab(grammar["rules"][token])
        vocab.update(new_tokens)
        grammar_patterns.update(new_patterns)
    return vocab, grammar_patterns


def regex_to_member(regex_string):
    sre_tree = sre_parse.parse(regex_string)
    command = None # this is implicitly a sequence
    
    def sre_member_to_TS_member(command, content):
        if command == None:
            member = {"type": "SEQ",
                      "members": [sre_member_to_TS_member(sub_command, sub_content) for sub_command, sub_content in content]
                     }
            return member
        if str(command) in ["BRANCH"]:
            member = {"type": "CHOICE",
                      "members": [sre_member_to_TS_member(None, sub_content) for sub_content in content[1]]
                     }
            return member
        if str(command) in ["IN"]:
            member = {"type": "CHOICE",
                      "members": [sre_member_to_TS_member(sub_command, sub_content) for sub_command, sub_content in content]
                     }
            return member
        if str(command) in ["RANGE"]:
            member = {"type": "CHOICE",
                      "members": [{"type":"STRING","value":chr(ascii_val)} for ascii_val in range(content[0], content[1]+1)]
                     }
            return member
        if str(command) in ["LITERAL"]:
            member = {"type": "STRING",
                      "value": chr(content)
                     }
            return member
        
        if str(command) in ["MAX_REPEAT"]:
            # I'm currently not handling situations where a minimum number of repetitions need to happen or a maximum, this should be fixed here if necessary
            minimum_repeat, max_repeat, content = content
            if max_repeat != 0 and str(max_repeat) != "MAXREPEAT":
                member = {"type": "CHOICE",
                          "members": [{"type": "BLANK"},sre_member_to_TS_member(None, content)]
                         }
            else:
                member = {"type": "REPEAT" + ("1" if minimum_repeat == 1 else ""),
                          "content": sre_member_to_TS_member(None, content)
                         }
            return member
    
    return sre_member_to_TS_member(command, sre_tree)