from IPython.display import Image, display
from pyformlang.finite_automaton import EpsilonNFA, State, Symbol, Epsilon
from tree_sitter import Language, Parser, Node
import networkx as nx
import json
import autopep8


python_statements = ['print_statement',
 'continue_statement',
 'assert_statement',
 'decorated_definition',
 'return_statement',
 'for_statement',
 'expression_statement',
 'while_statement',
 'import_statement',
 'import_from_statement',
 'class_definition',
 'nonlocal_statement',
 'exec_statement',
 'with_statement',
 'if_statement',
 'break_statement',
 'global_statement',
 'function_definition',
 'delete_statement',
 'future_import_statement',
 'pass_statement',
 'try_statement',
 'raise_statement',
 'decorator']

pytohn_non_space_nodes = ["_string_start", "_string_content", "_string_end"]

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
        self.punct_dict = {",":"COMMA", '\"':"QUOTE","\'":"QUOTE"}
        self.node_types = grammar["rules"]
        self.start_state = State("START")
        self.match_state = State("MATCH")
        self.add_start_state(self.start_state)
        self.add_final_state(self.match_state)
        self.add_transitions(self.node_types[member_name], self.start_state, self.match_state)
        
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

        if member["type"] == "PATTERN":
            symbol = Symbol(member["value"])
            self.add_transition(enter_state, symbol, exit_state)

        if member["type"] in ["FIELD", "PREC","PREC_LEFT","PREC_RIGHT","ALIAS"]:
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
        return any([self.is_final_state(x) for x in current_states]), possible_transitions


class StringTSNode():
    def __init__(self, node_type, children=[], text=""):
        self.type = node_type
        self.children = children
        self.text = text


class Code_Parser():
    def __init__(self, language="python"):
        Language.build_library('build/my-languages.so',[
                'src/tree-sitter/tree-sitter-javascript',
                'src/tree-sitter/tree-sitter-python'])
        
        LANGUAGE = Language('build/my-languages.so', language)
        self.TS_parser = Parser()
        self.TS_parser.set_language(LANGUAGE)
        
    def parse_to_string_tree(self, code_str):
        tree = self.TS_parser.parse(bytes(code_str, "utf8"))
        root_node = tree.root_node
        return self.TSNode_to_StringTSNode(root_node, code_str)
        
    def TSNode_to_StringTSNode(self, TSNode, code_str):
        if TSNode.type == "string":
            node_text = sub_str_from_coords(code_str, TSNode.start_point, TSNode.end_point)
            children = [StringTSNode("_string_start", children=[], text='"'),
                        StringTSNode("_string_content", children=[], text=node_text[1:-1]),
                        StringTSNode("_string_end", children=[], text='"')]
            return StringTSNode("string", children=children, text="")
        
        if TSNode.children == []:
            node_text = sub_str_from_coords(code_str, TSNode.start_point, TSNode.end_point)
            node = StringTSNode(TSNode.type, children=[], text=node_text)
            return node
        else:
            parent_node = StringTSNode(TSNode.type)
            string_TSNode_children = []
            for child in TSNode.children:
                string_TSNode_children.append(self.TSNode_to_StringTSNode(child, code_str))
            parent_node.children = string_TSNode_children
            return parent_node
    
    
class Node_Processor():
    def __init__(self, language="python"):
        self.counter = 0
        if language == "python":
            global python_statements
            self.new_line_statements = python_statements
            global pytohn_non_space_nodes
            self.non_space_nodes = pytohn_non_space_nodes
        
    def to_string(self, node, indent=""):
        if node.children == []:
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
                if child.type not in self.non_space_nodes:
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

        if node.children == []:
            node_text = node.text
            if node_text != node.type:
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