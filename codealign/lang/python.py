"""Interact with tree_sitter to convert python code into a useful IR.
"""
from typing import Tuple
import collections
from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser, Node

from .c import print_immediate_children, print_types_recursively # for debugging
from .c import BlockSuccessorAssignment, Next, Continue, Break, SemanticError, clean_up_empty_blocks, error_check, ASSIGNMENT_SUBOPS
from ..ir import *

PYTHON_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PYTHON_LANGUAGE)
# parser.set_language(PYTHON_LANGUAGE)

ITERATOR_FN_NAME = "next"

class SyntaxError(Exception):
    pass

class Scope:
    def __init__(self, parent_scope: Optional['Scope'] = None):
        self.name2obj = {}
        self.parent = parent_scope
        self.temporary_index = 0
        self.global_scope: 'Scope' = self if parent_scope is None else parent_scope.global_scope

    def _find_scope(self, variable_name: str) -> Optional['Scope']:
        current = self
        while current is not None and variable_name not in current.name2obj:
            current = current.parent
        return current

    def variable_exists(self, variable_name: str) -> bool:
        assert isinstance(variable_name, str)
        return self._find_scope(variable_name) is not None

    def variable_read(self, variable_name: str) -> Variable:
        """Returns an existing codealign.ir.Variable object with this name if it has been defined before. If not, this method creates a new
        variable at the global scope with this name and returns it.
        """
        assert isinstance(variable_name, str)
        scope = self._find_scope(variable_name)
        if scope is not None:
            return scope.name2obj[variable_name]
        
        # Global as in "external to the function" not necessarily declared using the "global" keyword.
        # Making this a global variable will cause the alignment technique to assume this variable was defined at some point
        # outside the function before this function was called. Otherwise, a variable not previously defined would be assumed
        # to be uninitialized by the alignment technique.
        var = GlobalVariable(variable_name)
        self.global_scope.name2obj[variable_name] = var # without additional knowledge, assume the global scope.
        return var
    
    def variable_written(self, variable_name: str) -> Variable:
        """Returns an existing codealign.ir.Variable object with this name if it has been defined before in this scope. If not, this method creates a new
        variable at this scope with the provided name and returns it.
        """
        assert isinstance(variable_name, str)
        if variable_name in self.name2obj:
            return self.name2obj[variable_name]
        var = Variable(variable_name)
        self.name2obj[variable_name] = var
        return var
    
    def declare_global(self, variable_name: str):
        """Implements 'global x' statements.

        Fetches an existing variable from the global scope if there is one; creates a new one otherwise.
        Places that variable in this scope without removing it from the enclosing scope.
        """
        assert isinstance(variable_name, str)
        if variable_name in self.name2obj:
            # Declaring a variable before the 'global' statement is illegal in python. It causes a SyntaxError to be raised.
            raise SyntaxError(f"Variable {variable_name} is assigned before global declaration.")

        if variable_name in self.global_scope.name2obj:
            variable = self.global_scope.name2obj[variable_name]
        else:
            variable = GlobalVariable(variable_name)
            self.global_scope.name2obj[variable_name] = variable
        
        self.name2obj[variable_name] = variable

    def declare_nonlocal(self, variable_name: str):
        """Implements 'nonlocal x' statements.
        
        Fetch an existing variable from an outer scope and place it here in this scope without removing
        it from the enclosing scope.
        """
        assert isinstance(variable_name, str)
        scope = self
        while scope != self.global_scope: # Nonlocal cannot fetch from the global scope.
            if variable_name in scope.name2obj:
                variable = scope.name2obj[variable_name]
                self.name2obj[variable_name] = variable
                return variable
            else:
                scope = scope.parent
        
        # Trying to fetch a nonlocal variable that does not already exist is a SyntaxError in python.
        # nonlocal cannot fetch from the global scope.
        raise SyntaxError(f"No definition in enclosing nonglobal scope for nonlocal variable {variable_name}.")
    
    def delete_variable(self, variable_name: str):
        """Implements 'del x' statements.
        
        Removes the variable from the current scope if it exists; otherwise, it throws an exception.
        """
        if variable_name in self.name2obj:
            del self.name2obj[variable_name]
        else:
            raise SemanticError(f"{variable_name} is not defined in this scope; cannot del.")
         
    def create_parameter(self, parameter_name: str) -> Parameter:
        assert isinstance(parameter_name, str)
        p = Parameter(parameter_name)
        self.name2obj[parameter_name] = p
        return p
    
    def create_temporary(self) -> Variable:
        temporary_variable_name = f"t{self.temporary_index}"
        while self.variable_exists(temporary_variable_name):
            self.temporary_index += 1
            temporary_variable_name = f"t{self.temporary_index}"
        variable = Variable(temporary_variable_name, is_temporary=True)
        self.temporary_index += 1
        self.name2obj[temporary_variable_name] = variable
        return variable
    
    def __repr__(self):
        variables = [f"{type(v)}({v.name})" for _, v in self.name2obj.items()]
        return "Scope(" + ", ".join(variables) + (")" if self.parent is None else ")\n -> " + repr(self.parent))
    
### ProxyOperators represent partially converted AST nodes.

class ProxyOperator(ABC):
    """Represents a partially completed AST node.
    """
    pass

class ProxyVarOperator(ProxyOperator):
    """A partially built VarOperator, a ProxyVarOperator is missing its return value. VarOperators are intended
    to be immutable, and building them up over time conflicts with this ideal. ProxyVarOperators are a separate
    type that allows VarOperators to be built up in several steps without compromising the immutability of VarOperators.
    A ProxyVarOperator is still under construction; a VarOperator is complete.

    Call create_operator with the return value to build the complete operator.
    """
    def __init__(self, op, operands=List[VarOperator], ast_node: Optional[Node] = None):
        self.op = op
        self.operands = operands
        self.ast_node = ast_node

    def create_operator(self, result: Optional[Variable]) -> VarOperator:
        return VarOperator(self.op, result, self.operands, self.ast_node)
    
    def __repr__(self):
        op_names = [repr(op) for op in self.operands]
        return f"<proxy> {self.op} " + " ".join(op_names)

class ProxyFunctionVarOperator(ProxyVarOperator):
    """A partially built FunctionVarOperator, a ProxyFunctionVarOperator is missing its return value.

    Call create_operator with the return value to build the complete operator.
    """
    def __init__(self, name: Union[str, Variable], operands: List[VarOperand], kwargs: Optional[Dict[str, VarOperand]] = None, ast_node: Optional[Node] = None):
        super().__init__(FUNCTION_CALL_OP, operands)
        self.name = name
        self.kwargs = kwargs
        self.ast_node = ast_node
    
    def create_operator(self, result: Optional[Variable]) -> FunctionVarOperator:
        return FunctionVarOperator(self.name, result, self.operands, self.kwargs, self.ast_node)
    
    def __repr__(self):
        op_names = [repr(op) for op in self.operands]
        name_repr = f"(*{self.name})" if isinstance(self.name, Variable) else self.name
        if self.kwargs is not None:
            op_names += [f"{k}={v}" for k, v in self.kwargs.items()]
        return f"<proxy> {name_repr}(" + ", ".join(op_names) + ")"


class ProxyControlFlowExpression(ProxyOperator):
    """Represents an expression that decomposes into a collection of operators with internal control
    flow such as a ternary operator or list comprehension.
    """
    def __init__(self, result: Optional[Variable]):
        self.result = result

class ProxyComprehension(ProxyControlFlowExpression):
    class ForClause:
        def __init__(self, pre_iterator_block_operators: List[VarOperator], iterator_block_operators: List[VarOperator], body_operators: List[VarOperator]):
            self.pre_iterator_block_operators = pre_iterator_block_operators
            self.iterator_block_operators = iterator_block_operators
            self.body_operators = body_operators
    
    class IfClause:
        def __init__(self, condition_block_operators: List[VarOperator]):
            self.condition_block_operators = condition_block_operators
    
    def __init__(self, body: List[VarOperator], 
                body_result: Union[Variable, Tuple[Variable, VarOperand]],
                clauses: List[List[Union['ProxyComprehension.ForClause', 'ProxyComprehension.IfClause']]],
                temporary_variables: List[Variable], # For the results
                ast_node: Optional[Node] = None
                ):
        self.body = body
        self.body_result = body_result
        assert len(clauses) > 0
        assert isinstance(clauses[0], ProxyComprehension.ForClause)
        self.clauses = clauses
        self.temporary_variables = temporary_variables
        self.ast_node = ast_node
        super().__init__(None)
    
    def __repr__(self):
        ops_repr = lambda ops: "    " + "\n    ".join([repr(operator) for operator in ops]) + "\n"
        outstr = f"Body:\n{ops_repr(self.body)}    {self.body_result}\n"
        for clause in self.clauses:
            if isinstance(clause, ProxyComprehension.ForClause):
                outstr += "For Clause:\n" + ops_repr(clause.iterator_block_operators) + ops_repr(clause.body_operators)
            else:
                outstr += "If Clause:\n" + ops_repr(clause.condition_block_operators)
        return outstr


class ProxyListComprehension(ProxyComprehension):
    def __repr__(self):
        return "List Comprehension\n" + super().__repr__()

class ProxyDictionaryComprehension(ProxyComprehension):
    def __repr__(self):
        return "Dictionary Comprehension\n" + super().__repr__()
    
class ProxySetComprehension(ProxyComprehension):
    def __repr__(self):
        return "Set Comprehension\n" + super().__repr__()

class ProxyTernaryExpression(ProxyControlFlowExpression):
    def __init__(self, condition_block: List[VarOperator],
                 condition_result: VarOperand,
                 true_block: List[VarOperator],
                 true_result: VarOperand,
                 false_block: List[VarOperator],
                 false_result: List[VarOperator],
                 ast_node: Optional[Node] = None
                ):
        self.condition_block = condition_block
        self.condition_result = condition_result
        self.true_block = true_block
        self.true_result = true_result
        self.false_block = false_block
        self.false_result = false_result
        self.ast_node = ast_node
    
    def __repr__(self):
        outstr = "Proxy Ternary Operator\n"
        outstr += "Condition\n"
        outstr += "  " + "  \n".join([repr(op) for op in self.condition_block])
        outstr += f"\n  if {self.condition_result}"
        outstr += "\nTrue Branch\n"
        outstr += "  " + "  \n".join([repr(op) for op in self.true_block])
        outstr += f"\n  {self.true_result}"
        outstr += "\nFalse Branch\n"
        outstr += "  " + "  \n".join([repr(op) for op in self.false_block])
        outstr += f"\n  {self.false_result}"
        return outstr


ASSIGNMENT_SUBOPS = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
    "%=": "%",
    "<<=": "<<",
    ">>=": ">>",
    "&=": "&",
    "^=": "^",
    "|=": "|"
}

def get_assignment_subopcode(operator: Node) -> str:
    assert operator.type != "=", "'=' does not have a sub-opcode; other assignment operators do (e.g. +=)."
    assert operator.type in ASSIGNMENT_SUBOPS, f"{operator.type} not a valid python assignment operator."
    return ASSIGNMENT_SUBOPS[operator.type]
    
### Convert expressions to IR form.
### See the C parser for a comprehensive overview of the multi-way recursive functions bind_expression, expand_subexpression, and convert_expression.

def bind_expression(expression: Node, scope: Scope, expression_ops: List[VarOperator]) -> Variable:
    """Binds an expression to a variable. There are several cases to consider.
    If the expression is not an assignment expression, this method binds it to a new temporary variable.
    If the expression is an assignment expression, it will bind it to that variable if the left hand side (lhs) 
    is in fact a variable.
    If not, it will decompose the left hand side into a sequence of operations that ends in a temporary variable.
    This temporary variable will be used in a store operation.
    """
    expression = clean_expression(expression)
    if expression.type == "assignment":
        lhs = expression.child_by_field_name("left")
        if lhs.type == "identifier":
            result = scope.variable_written(lhs.text.decode("utf8"))
            proxy_operator = convert_expression(expression.child_by_field_name("right"), scope, expression_ops)
        else:
            result = bind_expression(lhs, scope, expression_ops)
            rhs_result = expand_subexpression(expression.child_by_field_name("right"), scope, expression_ops)
            proxy_operator = ProxyVarOperator(STORE_OP, [result, rhs_result], expression)
    elif expression.type == "augmented_assignment":
        lhs = expression.child_by_field_name("left")
        if lhs.type == "identifier": # NOT done through expand_subexpression becuase we want to call scope.variable_written rather than scope.variable_read
            lhs_result = scope.variable_written(lhs.text.decode("utf8"))
        else:
            # This could also be bind_expression because lhs should't be a leaf.
            assert check_expression_leaf(lhs, scope) == None, f"Cannot assign to a literal: {lhs.text.decode('utf8')}." # We know this is not an identifier because we would be in the if body, not the else.
            lhs_result = bind_expression(lhs, scope, expression_ops)
        rhs_result = expand_subexpression(expression.child_by_field_name("right"), scope, expression_ops)
        opcode = get_assignment_subopcode(expression.child_by_field_name("operator"))
        proxy_operator = ProxyVarOperator(opcode, [lhs_result, rhs_result], expression)
        result = lhs_result
    else: # Not an assignment.
        proxy_operator = convert_expression(expression, scope, expression_ops)
        result = scope.create_temporary()

    if isinstance(proxy_operator, ProxyVarOperator):
        expression_ops.append(proxy_operator.create_operator(result))
    else:
        assert isinstance(proxy_operator, ProxyControlFlowExpression)
        proxy_operator.result = result
        expression_ops.append(proxy_operator)
    
    return result

def check_expression_leaf(expression: Node, scope: Scope) -> Optional[Constant]:
    if expression.type == "identifier":
        return scope.variable_read(expression.text.decode("utf8"))
    elif expression.type == "integer":
        return IntegerConstant(expression.text.decode("utf8"))
    elif expression.type == "string":
        return StringLiteral(expression.text.decode("utf8"))
    elif expression.type == "true" or expression.type == "false":
        return BoolLiteral(expression.type)
    elif expression.type == "float":
        return FloatConstant(expression.text.decode("utf8"))
    elif expression.type == "none":
        return NullLiteral()
    elif expression.type == "ellipsis":
        return Ellipsis()
    elif expression.type == "lambda":
        parameters_node = expression.child_by_field_name("parameters")
        if parameters_node is not None:
            lambda_parameters = parameters_node.children
            assert all(map(lambda x: x.type == "identifier" or x.type == ",", lambda_parameters))
            num_parameters = (len(lambda_parameters) + 1) / 2 # parameter list includes commas
        else:
            num_parameters = 0
        return Lambda(num_parameters)
    
    return None

def clean_expression(expression: Node) -> Node:
    while expression.type == "parenthesized_expression":
        assert(expression.children[0].type == "(")
        assert(expression.children[-1].type ==")")
        assert(len(expression.children) == 3)
        expression = expression.children[1]
    return expression
    

def expand_subexpression(expression: Node, scope: Scope, expression_ops: List[VarOperator]) -> VarOperand:
    """Reduces an expression to a single VarOperand (a constant or variable) and returns it.
    """
    expression = clean_expression(expression)
    operand = check_expression_leaf(expression, scope)
    if operand is None:
        operand = bind_expression(expression, scope, expression_ops)
    return operand


def convert_expression(expression: Node, scope: Scope, expression_ops: List[VarOperator]) -> ProxyOperator:
    """Returns the most of the necessary ingredients to build a VarOperator, but does not create it. That is done by bind_expression,
    which supplies the last ingredient: the place to store the return value.
    """
    expression = clean_expression(expression)
    assert expression.type != "assignment" and expression.type != "augmented_assignment", "INTERNAL ERROR: Assignment should be handled by bind_expression."

    leaf = check_expression_leaf(expression, scope)
    if leaf is not None:
        return ProxyVarOperator(COPY_OP, [leaf], expression)
    elif expression.type == "call":
        # children: function, arguments

        ## Process function name
        function_name_node = expression.child_by_field_name("function")
        if function_name_node.type == "identifier":
            # This call could be using a function variable.
            function_name = function_name_node.text.decode("utf8")
            if scope.variable_exists(function_name):
                function_name = scope.variable_read(function_name)
        else:
            function_name = expand_subexpression(function_name_node, scope, expression_ops)

        ## Process function arguments
        arguments = expression.child_by_field_name("arguments")
        if arguments.type == "argument_list":
            assert arguments.children[0].type == "("
            assert arguments.children[-1].type == ")"
            arguments = arguments.children[1:-1]
        else:
            assert arguments.type == "generator_expression"
            arguments = [arguments]

        operands: List[VarOperand] = [] # the operands in this expression
        kwargs: Dict[str, VarOperand] = {}
        for argument in arguments:
            if argument.type == "," or argument.type == "comment":
                continue
            elif argument.type == "keyword_argument":
                # children: name, =, value
                arg_name_node = argument.child_by_field_name("name")
                assert arg_name_node.type == "identifier"
                kwargs[arg_name_node.text.decode("utf8")] = expand_subexpression(argument.child_by_field_name("value"), scope, expression_ops)
            else:
                operands.append(expand_subexpression(argument, scope, expression_ops))
        
        return ProxyFunctionVarOperator(function_name, operands, None if len(kwargs) == 0 else kwargs, expression)
    elif expression.type == "binary_operator":
        # children: left, operator, right
        left = expand_subexpression(expression.child_by_field_name("left"), scope, expression_ops)
        opcode = expression.child_by_field_name("operator").text.decode("utf8")
        right = expand_subexpression(expression.child_by_field_name("right"), scope, expression_ops)
        return ProxyVarOperator(opcode, [left, right], expression)
    elif expression.type == "unary_operator":
        # children: operator, argument
        opcode = expression.child_by_field_name("operator").text.decode("utf8")
        assert opcode in {"-", "+", "~"}
        operand = expand_subexpression(expression.child_by_field_name("argument"), scope, expression_ops)

        return ProxyVarOperator(opcode, [operand], expression)
    elif expression.type == "attribute":
        # Has two fields: object and attribute.
        attribute = expression.child_by_field_name("attribute")
        assert attribute.type == "identifier"
        obj_var = expand_subexpression(expression.child_by_field_name("object"), scope, expression_ops)

        return ProxyVarOperator(MEMBER_ACCESS_OP, [obj_var, Field(attribute.text.decode("utf8"))], expression)
    elif expression.type == "tuple" or expression.type == "expression_list":
        if expression.type == "tuple":
            assert expression.children[0].type == "("
            assert expression.children[-1].type == ")"
            children = expression.children[1:-1]
        else:
            children = expression.children

        operands: List[VarOperand] = []
        for child in children:
            if child.type == "," or child.type == "comment":
                continue
            operands.append(expand_subexpression(child, scope, expression_ops))

        return ProxyVarOperator(TUPLE_INITIALIZER_OP, operands, expression)
    elif expression.type == "dictionary":
        assert expression.children[0].type == "{"
        assert expression.children[-1].type == "}"

        dict_pairs: List[Tuple[str, VarOperand, VarOperand]] = []
        for child in expression.children[1:-1]:
            if child.type == "," or child.type == "comment":
                continue
            assert child.type == "pair"
            # the children of a pair are key, :, and value.
            key = child.child_by_field_name("key")
            value = child.child_by_field_name("value")
            dict_pairs.append((key.text.decode("utf8"), expand_subexpression(key, scope, expression_ops), expand_subexpression(value, scope, expression_ops)))
            
        # In dictionaries, order doesn't matter. (In later versions of python iteration order is guaranteed
        # to be the same as insertion order so this doesn't entirely hold.)
        # Because order doesn't matter, we'll sort dictionaries by key so that dictionaries that have equivalent
        # contents align as much as possible.
        dict_pairs.sort(key=lambda x: x[0])

        # Now that the pairs have been sorted, flatten the arguments out. Key-value "pair" abstractions are only used in 
        # dictionaries; the alignment mechanism will already know that this is a dictionary because its opcode
        # is DICTIONARY_INITIALIZER_OP. This strategy simplifies the implementation without loosing expressive 
        # power.
        operands: List[VarOperand] = []
        for _, k, v in dict_pairs:
            operands.append(k)
            operands.append(v)

        return ProxyVarOperator(DICTIONARY_INITIALIZER_OP, operands, expression)
    elif expression.type == "list":
        assert expression.children[0].type == "["
        assert expression.children[-1].type == "]"
        
        operands: List[VarOperand] = []
        for child in expression.children[1:-1]:
            if child.type == "," or child.type == "comment":
                continue
            operands.append(expand_subexpression(child, scope, expression_ops))

        return ProxyVarOperator(ARRAY_INITIALIZER_OP, operands, expression)
    elif expression.type == "set":
        assert expression.children[0].type == "{"
        assert expression.children[-1].type == "}"
        
        elements: List[Tuple[str, VarOperand]] = []
        for child in expression.children[1:-1]:
            if child.type == "," or child.type == "comment":
                continue
            elements.append((child.text.decode("utf8"), expand_subexpression(child, scope, expression_ops)))
        
        # In sets, order doesn't matter. Unlike with dictionaries, the insertion order is not preserved, so there 
        # is no drawback to sorting.
        elements.sort(key=lambda x: x[0])
        
        return ProxyVarOperator(SET_INITIALIZER_OP, [e[1] for e in elements], expression)
    elif expression.type == "subscript":
        # children: value, [, subscript, ].
        data_structure = expand_subexpression(expression.child_by_field_name("value"), scope, expression_ops)
        subscript = expand_subexpression(expression.child_by_field_name("subscript"), scope, expression_ops)

        return ProxyVarOperator(SUBSCRIPT_OP, [data_structure, subscript], expression)
    elif expression.type == "comparison_operator":
        # children None (lhs, no name), operators, None (rhs, no name)
        if len(expression.children) == 3:
            left = expand_subexpression(expression.children[0], scope, expression_ops)
            opcode = expression.children[1].text.decode("utf8")
            right = expand_subexpression(expression.children[2], scope, expression_ops)
            return ProxyVarOperator(opcode, [left, right], expression)
        elif len(expression.children) == 4:
            if expression.children[1].text.decode("utf8") == "not" and expression.children[2].text.decode("utf8") == "in":
                left = expand_subexpression(expression.children[0], scope, expression_ops)
                right = expand_subexpression(expression.children[3], scope, expression_ops)
                in_temporary = scope.create_temporary()
                expression_ops.append(VarOperator(MEMBERSHIP_OP, in_temporary, [left, right], expression))
                return ProxyVarOperator(NOT_OP, [in_temporary], expression)
            else:
                assert expression.children[1].text.decode("utf8") == "is" and expression.children[2].text.decode("utf8") == "not"
                left = expand_subexpression(expression.children[0], scope, expression_ops)
                right = expand_subexpression(expression.children[3], scope, expression_ops)
                return ProxyVarOperator("!=", [left, right], expression)
        else:
            assert len(expression.children) == 5
            # a compound comparison operator like 0 < k < 10
            left = expand_subexpression(expression.children[0], scope, expression_ops)
            lop = expression.children[1].text.decode("utf8")
            middle = expand_subexpression(expression.children[2], scope, expression_ops)
            
            ltemp = scope.create_temporary()
            expression_ops.append(VarOperator(lop, ltemp, [left, middle], expression))
            
            # Python does short-circuit the implied "and" in a compound comparison like this. We process "right" after
            # creating the first comparison to reflect this. 
            rop = expression.children[3].text.decode("utf8")
            right = expand_subexpression(expression.children[4], scope, expression_ops)
            
            rtemp = scope.create_temporary()
            expression_ops.append(VarOperator(rop, rtemp, [middle, right], expression))
            
            return ProxyVarOperator(AND_OP, [ltemp, rtemp], expression)
    elif expression.type == "not_operator":
        # children: not, argument
        argument = expand_subexpression(expression.child_by_field_name("argument"), scope, expression_ops)
        return ProxyVarOperator(NOT_OP, [argument], expression)
    elif expression.type == "boolean_operator":
        # children: left, and/or, right
        left = expand_subexpression(expression.child_by_field_name("left"), scope, expression_ops)
        opcode = expression.child_by_field_name("operator").text.decode("utf8")
        right = expand_subexpression(expression.child_by_field_name("right"), scope, expression_ops)
        canonical_op_mapping = {"or": OR_OP, "and": AND_OP}
        assert opcode in canonical_op_mapping

        return ProxyVarOperator(canonical_op_mapping[opcode], [left, right], expression)
    elif expression.type == "slice":
        # slices have form start:stop:step, but each of start, stop, and step can be omitted. If step 
        # is omitted, then the second colon isn't necessary.
        start = NullLiteral()
        stop = NullLiteral()
        step = NullLiteral()
        current_idx = 0
        if expression.children[current_idx].type != ":":
            start = expand_subexpression(expression.children[current_idx], scope, expression_ops)
            current_idx += 1
        # The first : is mandatory.
        assert expression.children[current_idx].type == ":"
        current_idx += 1
        if current_idx < len(expression.children) and expression.children[current_idx].type != ":":
            stop = expand_subexpression(expression.children[current_idx], scope, expression_ops)
            current_idx += 1
        # This : is not mandatory. If this slice expression doesn't have enough children, that means it's omitted and that's fine.
        # Use CNF form of an implication. current_idx < len(...) => ...type == ":"
        assert not current_idx < len(expression.children) or expression.children[current_idx].type == ":"
        current_idx += 1
        if current_idx < len(expression.children) and expression.children[current_idx].type != ":":
            step = expand_subexpression(expression.children[current_idx], scope, expression_ops)
        
        return ProxyVarOperator(SLICE_OP, [start, stop, step], expression)
    elif expression.type == "conditional_expression": # Ternary operator
        assert len(expression.children) == 5
        # children[0]: the true case
        assert expression.children[1].type == "if"
        # children[2]: the condition
        assert expression.children[3].type == "else"
        # children[4]: the false case

        condition_block = []
        condition_result = expand_subexpression(expression.children[2], scope, condition_block)

        true_block = []
        true_result = expand_subexpression(expression.children[0], scope, true_block)

        false_block = []
        false_result = expand_subexpression(expression.children[4], scope, false_block)

        return ProxyTernaryExpression(
            condition_block, condition_result,
            true_block, true_result,
            false_block, false_result,
            ast_node=expression
        )
    elif expression.type == "list_comprehension":
        assert expression.children[0].type == "["
        assert expression.children[-1].type == "]"

        # List comprehensions have their own internal scope
        comprehension_scope = Scope(scope)

        # Process the for and if clauses first so that variables defined in the for clauses are added to the scope first and thus reference the body correctly.
        clauses = convert_comprehension_clauses(expression.children[2:-1], comprehension_scope)

        body_ops = []
        body_result = expand_subexpression(expression.child_by_field_name("body"), comprehension_scope, body_ops)
        temporary_variables = [comprehension_scope.create_temporary(), comprehension_scope.create_temporary()]

        return ProxyListComprehension(body_ops, body_result, clauses, temporary_variables, ast_node=expression)
    elif expression.type == "dictionary_comprehension":
        assert expression.children[0].type == "{"
        assert expression.children[-1].type == "}"

        # Dictionary comprehensions have their own internal scope
        comprehension_scope = Scope(scope)

        # Process the for and if clauses first so that variables defined in the for clauses are added to the scope first and thus reference the body correctly.
        clauses = convert_comprehension_clauses(expression.children[2:-1], comprehension_scope)

        pair_node = expression.child_by_field_name("body")
        assert pair_node.type == "pair"

        body_ops = []
        # The key argument expression (left in the pair) is evaluated first.
        key = expand_subexpression(pair_node.child_by_field_name("key"), comprehension_scope, body_ops)
        value = expand_subexpression(pair_node.child_by_field_name("value"), comprehension_scope, body_ops)
        # We'll need to save these variables because both key and value are needed in the conversion.
        
        return ProxyDictionaryComprehension(
            body_ops,
            (key, value),
            clauses,
            [comprehension_scope.create_temporary()], # only need one temporary here because it is re-used in the store op
            ast_node=expression
        )
    elif expression.type == "set_comprehension":
        assert expression.children[0].type == "{"
        assert expression.children[-1].type == "}"

        # Dictionary comprehensions have their own internal scope
        comprehension_scope = Scope(scope)

        # Process the for and if clauses first so that variables defined in the for clauses are added to the scope first and thus reference the body correctly.
        clauses = convert_comprehension_clauses(expression.children[2:-1], comprehension_scope)

        body_ops = []
        body_result = expand_subexpression(expression.child_by_field_name("body"), comprehension_scope, body_ops)
        temporary_variables = [comprehension_scope.create_temporary(), comprehension_scope.create_temporary()]

        return ProxySetComprehension(body_ops, body_result, clauses, temporary_variables, ast_node=expression)
    elif expression.type == "generator_expression":
        assert expression.children[0].type == "("
        assert expression.children[-1].type == ")"
        # Collect all of the variables used in the generator expression.
        variables_read = []
        
        class ReadRecordingScope(Scope):
            def variable_read(self, variable_name: str) -> Variable:
                variable = super().variable_read(variable_name)
                scope = self._find_scope(variable_name)
                if scope != self:
                    variables_read.append(variable)
                return variable
            
        generator_scope = ReadRecordingScope(scope)

        # We don't actually care what operators were generated; we are only concerned with the variables from outside the generator expression that were used inside.
        # Those variables are the arguments to the generator expression operator.
        convert_comprehension_clauses(expression.children[2:-1], generator_scope)
        expand_subexpression(expression.child_by_field_name("body"), generator_scope, [])
        
        # TODO: Consider situation with same variables, different order.
        return ProxyVarOperator(GENERATOR_OP, variables_read, expression) # The arguments to a generator expression are the variables it reads from the scopes outside of its own.
    elif expression.type == "list_splat":
        assert expression.children[0].type == "*"
        assert len(expression.children) == 2
        unpacked_list = expand_subexpression(expression.children[1], scope, expression_ops)
        return ProxyVarOperator(ARGUMENT_UNPACK, [unpacked_list], expression)
    elif expression.type == "dictionary_splat":
        assert expression.children[0].type == "**"
        assert len(expression.children) == 2
        unpacked_dict = expand_subexpression(expression.children[1], scope, expression_ops)
        return ProxyVarOperator(KEYWORD_ARGUMENT_UNPACK, [unpacked_dict], expression)
    elif expression.type == "yield":
        assert expression.children[0].type == "yield"
        if expression.children[1].type == "from":
            raise NotImplementedError("'yield from' statement not implemented.")
        else:
            assert len(expression.children) == 2
            item = expand_subexpression(expression.children[1], scope, expression_ops)
            return ProxyVarOperator(YIELD_OP, [item], expression)
    elif expression.type == "concatenated_string":
        assert len(expression.children) == 2, f"Support for {len(expression.children)} whitespace concatenated strings with is not implemented."
        return ProxyVarOperator("+", [expand_subexpression(expression.children[0], scope, expression_ops), 
                                      expand_subexpression(expression.children[1], scope, expression_ops)], 
                                expression)
    else:
        raise NotImplementedError(f"No code to handle expression of type {expression.type}.")

def convert_comprehension_clauses(clause_nodes: List[Node], comprehension_scope: Scope) -> List[Union[ProxyComprehension.ForClause, ProxyComprehension.IfClause]]:
    """Convert the for and if clauses of list and dictionary comprehensions and place them in proxy wrappers.
    """
    assert clause_nodes[0].type == "for_in_clause"
    clauses = []
    for clause_node in clause_nodes:
        if clause_node.type == "for_in_clause":
            pre_iterator_block_ops: List[VarOperator] = []
            iterand_or_iterable = expand_subexpression(clause_node.child_by_field_name("right"), comprehension_scope, pre_iterator_block_ops)
            
            iterand_node = clause_node.child_by_field_name("left") # the "i" in "for i in range(10)"
            if iterand_node.type == "identifier":
                iterand = comprehension_scope.variable_written(iterand_node.text.decode("utf8"))
            else:
                assert iterand_node.type == "pattern_list" or iterand_node.type == "tuple_pattern"
                iterand = comprehension_scope.create_temporary()
            
            iterator_block_ops = [
                FunctionVarOperator(ITERATOR_FN_NAME, iterand, [iterand_or_iterable], ast_node=clause_node), 
                VarOperator(LOOP_OP, None, [iterand], ast_node=clause_node)
            ]

            for_body_operators = []
            if iterand_node.type == "pattern_list" or iterand_node.type == "tuple_pattern":
                convert_pattern_list_or_tuple_pattern(iterand_node, comprehension_scope, for_body_operators, iterand)
            
            clauses.append(ProxyComprehension.ForClause(
                pre_iterator_block_ops,
                iterator_block_ops,
                for_body_operators
            ))
        else:
            assert clause_node.type == "if_clause"
            assert clause_node.children[0].type == "if"
            comparison_ops: List[VarOperator] = []
            comparison_result = expand_subexpression(clause_node.children[1], comprehension_scope, comparison_ops)
            comparison_ops.append(VarOperator(IF_OP, None, [comparison_result], clause_node))
            clauses.append(ProxyComprehension.IfClause(
                comparison_ops
            ))
    
    return clauses

    

def assigns_to_pattern_list_or_tuple_pattern(statement: Node) -> bool:
    """Statements which assign to pattern lists (e.g. a, b = obj) cannot be handled by bind_expression, 
    the function that normally handles expession_statements. This function checks if an expression_statement
    assigns to a pattern_list.
    """
    assert statement.type == "expression_statement"
    if statement.children[0].type == "assignment":
        left = statement.children[0].child_by_field_name("left").type
        return left == "pattern_list" or left == "tuple_pattern"
    return False

def convert_pattern_list_or_tuple_pattern(pattern: Node, scope: Scope, expression_ops: List[VarOperator], rhs: Variable):
    """Converts a pattern list (e.g. a, b = obj) into a sequence of individual assignments to variables. Also handles the
    related tuple pattern (e.g. (a, b) = obj).

    Implicitly, pattern_lists uses iterators; this conversion does so as well.
    """
    assert pattern.type == "pattern_list" or pattern.type == "tuple_pattern"

    def add_pattern_var_init(name_node: Node):
        if name_node.type == "identifier":
            expression_ops.append(
                FunctionVarOperator(ITERATOR_FN_NAME,
                                    scope.variable_written(name_node.text.decode("utf8")), 
                                    [rhs], ast_node=pattern))
        else:
            # There can be lval expressions in a pattern list, like l[0], l[1] = (1, 2)
            lval_result = expand_subexpression(name_node, scope, expression_ops)
            next_result = scope.create_temporary()
            expression_ops.append(FunctionVarOperator(ITERATOR_FN_NAME, next_result, [rhs], ast_node=pattern))
            expression_ops.append(VarOperator(STORE_OP, lval_result, [lval_result, next_result], ast_node=name_node))

    def process_tuple_pattern(tuple_pattern: Node):
        assert tuple_pattern.children[0].type == "("
        assert tuple_pattern.children[-1].type == ")"
        for child in tuple_pattern.children[1:-1]:
            if child.type == ",":
                continue
            elif child.type == "tuple_pattern":
                process_tuple_pattern(child)
            else:
                add_pattern_var_init(child)
                

    if pattern.type == "tuple_pattern":
        process_tuple_pattern(pattern)
    else:
        for child in pattern.children:
            if child.type == ",":
                continue
            elif child.type == "tuple_pattern":
                process_tuple_pattern(child)
            else:
                add_pattern_var_init(child)

def convert_exception_tuple(tuple_node: Node, operators: list[VarOperator], scope: Scope) -> Variable:
    """Converts a tuple of identifiers into a TUPLE_INITIALIZER_OP VarOperator. 
    
    This function is intended to be used to process the tuple in the
    except (Exception1, Exception2):
    and
    except (Exception1, Exception2) as e:
    patterns.
    """
    assert tuple_node.type == "tuple"
    assert tuple_node.children[0].type == "("
    assert tuple_node.children[-1].type == ")"

    exception_names: list[ExceptionName] = []
    for child in tuple_node.children[1:-1]:
        if child.type == ",":
            continue
        assert child.type == "identifier"
        exception_names.append(ExceptionName(child.text.decode("utf8")))

    temporary_variable = scope.create_temporary()
    operators.append(VarOperator(TUPLE_INITIALIZER_OP, temporary_variable, exception_names, ast_node=tuple_node))
    return temporary_variable

### Convert statements

def convert_block(block: Node, scope: Scope) -> List[Tuple[BasicBlock, Optional[BlockSuccessorAssignment]]]:
    """Convert a python block into a sequence of codealign IR basic blocks.
    """
    assert block.type == "block"
 
    # A None block successor assignment means the successors have already been assigned and there's nothing left to do.
    basic_blocks: List[Tuple[BasicBlock, Optional[BlockSuccessorAssignment]]] = []
    current_block = BasicBlock([], [], [])

    # For a full list of statement types, see https://github.com/tree-sitter/tree-sitter-python/blob/master/src/node-types.json
    for statement in block.children:
        if statement.type == "comment":
            continue
        elif statement.type == "expression_statement":
            # This might be an expression_list interpreted as a statement.
            for child in statement.children:
                if child.type == ",":
                    continue
                # Pattern lists (e.g. a, b = obj) violate the assumption of bind_expression that expressions
                # resolve to a single variable.
                if assigns_to_pattern_list_or_tuple_pattern(statement):
                    assert len(statement.children) == 1
                    # assigns_to_pattern_list ensures that the child of statement is an assignment.
                    assignment = statement.children[0]
                    rhs_var = expand_subexpression(assignment.child_by_field_name("right"), scope, current_block.operators)
                    convert_pattern_list_or_tuple_pattern(assignment.child_by_field_name("left"), scope, current_block.operators, rhs_var)
                else:
                    bind_expression(statement.children[0], scope, current_block.operators)
        elif statement.type == "return_statement":
            if len(statement.children) == 1:
                current_block.operators.append(VarOperator(RETURN_OP, None, [], ast_node=statement))
            else:
                assert len(statement.children) == 2
                assert statement.children[0].type == "return"
                return_val = expand_subexpression(statement.children[1], scope, current_block.operators)
                current_block.operators.append(VarOperator(RETURN_OP, None, [return_val], ast_node=statement))
            basic_blocks.append((current_block, None)) # A block ending in a return statement has no successors.
            return basic_blocks # Everything after this statement in this block is irrelevant.
        elif statement.type == "raise_statement":
            assert statement.children[0].type == "raise"
            if len(statement.children) == 1: # This is a raise statement without an argument.
                raise_arguments = []
            else: # argument is the exception to raise.
                raise_arguments = [expand_subexpression(statement.children[1], scope, current_block.operators)]
            current_block.operators.append(VarOperator(RAISE_OP, None, raise_arguments, ast_node=statement))
            basic_blocks.append((current_block, None)) # A block ending in a return statement has no successors.
            return basic_blocks # Everything after this statement in this block is irrelevant.
        elif statement.type == "break_statement":
            current_block.operators.append(VarOperator(BREAK_OP, None, [], ast_node=statement))
            basic_blocks.append((current_block, Break()))
            return basic_blocks # Everything after this statement in this block is irrelevant.
        elif statement.type == "continue_statement": # Uncomment after testing.
             current_block.operators.append(VarOperator(CONTINUE_OP, None, [], ast_node=statement))
             basic_blocks.append((current_block, Continue()))
             return basic_blocks
        elif statement.type == "if_statement":
            assert statement.children[0].type == "if"
            assert statement.children[2].type == ":"
            current_if_index = 3
            # Comments can also be the direct children of if statements instead of being factored into the "block" statement of the if body.
            while statement.children[current_if_index].type == "comment":
                current_if_index += 1
            assert statement.children[current_if_index].type == "block"
            condition_result = expand_subexpression(statement.children[1], scope, current_block.operators)
            current_block.operators.append(VarOperator(IF_OP, None, [condition_result], ast_node=statement))

            # The body_blocks list will track the basic blocks generated and assign successors later.
            body_blocks: List[Tuple[BasicBlock, BlockSuccessorAssignment]] = convert_block(
                statement.children[current_if_index], scope # If statement bodies share the same scope as outside the if in python, so we can simpy pass scope here
            )

            current_block.add_successor(body_blocks[0][0]) # One of an if condition's successors is the first block in the body of the "true" consequence.
            basic_blocks.append((current_block, None))

            predecessor_condition_block = current_block # Except for the first one, predecessor condition blocks are
            for i, clause in enumerate(statement.children[current_if_index + 1:]):
                if clause.type == "elif_clause":
                    # clause children: elif, condition, :, block
                    elif_condition_block = BasicBlock([], [], [])
                    predecessor_condition_block.add_successor(elif_condition_block) # add successor handles the symmetric adding of predecessor links.

                    # Process the elif condition.
                    elif_condition_result = expand_subexpression(clause.child_by_field_name("condition"), scope, elif_condition_block.operators)
                    elif_condition_block.operators.append(VarOperator(IF_OP, None, [elif_condition_result], ast_node=clause))

                    # Add the blocks from the elif condition and body to have their successors assigned later.
                    elif_blocks = convert_block(clause.child_by_field_name("consequence"), scope)
                    elif_condition_block.add_successor(elif_blocks[0][0]) # One successor of an elif condition block is the first block in its body.
                    body_blocks.append((elif_condition_block, None)) # Have already assigned one of this block's successors, the other will be assigned in or immediately after this for loop.
                    body_blocks.extend(elif_blocks)

                    # The successor of an arbitrary elif block is the previous elif block (except for the first; it's the original if condition block).
                    predecessor_condition_block = elif_condition_block
                else:
                    assert clause.type == "else_clause"
                    # Why i + 5: we're iterating over statement.children[current_if_index + 1:], so the last index of statement.children is i + current_if_index + 1. The length is thus i + current_if_index + 2.
                    assert i + current_if_index + 2 == len(statement.children), "If an else block exists, it must be the last block in an if statement."
                    else_blocks = convert_block(clause.child_by_field_name("body"), scope)
                    predecessor_condition_block.add_successor(else_blocks[0][0])
                    body_blocks.extend(else_blocks)
                    predecessor_condition_block = None
                
            current_block = BasicBlock([], [], []) # This is the basic block AFTER the if statement. Created here for successor assignment purposes.

            # This is the last block. We have an if, elif... elif with no else. Thus, a false condition will cause control to flow to the next block after the if statement.
            if predecessor_condition_block is not None: # assigned None only if an else_clause is encountered.
                # Not added to body_blocks because body_blocks already contains predecessor_condition_block or predecessor_condition_block is the original if condition.
                # The original if condition is also already added to basic_blocks. We just need to do the last successor assignment.
                predecessor_condition_block.add_successor(current_block) # Now the block after the if statement.
            
            for ifblock, successor_assignment in body_blocks:
                if isinstance(successor_assignment, Next):
                    ifblock.add_successor(current_block)
                    basic_blocks.append((ifblock, None))
                else:
                    basic_blocks.append((ifblock, successor_assignment)) # propagate other successor assignments, like Break, outside this scope.
        elif statement.type == "for_statement":
            # children of for statement: for, left, in, right, :, body, (else_clause). 
            assert statement.children[0].type == "for"
            assert statement.children[2].type == "in"
            assert statement.children[4].type == ":"

            iterator_block = BasicBlock([], [], [])
            current_block.add_successor(iterator_block)
            basic_blocks.append((current_block, None))
            basic_blocks.append((iterator_block, None))
            
            # Hand-waving StopIteration for simplicity. Also, codealign.ir doesn't have a great way to express exception catching.
            # In theory, this will mean that a while loop written to simulate a for loop (e.g. try: while True:, item = next(x); # process item; catch StopIteration) won't be
            # equivalent to a normal for loop. Given how rare manually-written simulated for-loops are, the increased complexity is not worth the pure correctness of handling
            # StopIteration.
            # This doesn't present too many issues for generating the IR, though. We can define codealign.ir semantics so that this can work.
            # All loops in codealign.ir are controlled by the LOOP_OP opcode. In codealign.ir, the next() function returns a boxed value or a stop-iteration value; the value is
            # auto-unboxed when used in the loop body.
            pattern_list_ops = []
            iterator_or_iterable = expand_subexpression(statement.children[3], scope, current_block.operators)
            if statement.children[1].type == "identifier":
                iterand = scope.variable_written(statement.children[1].text.decode("utf8"))
                iterator_block.operators.append(FunctionVarOperator(ITERATOR_FN_NAME, iterand, [iterator_or_iterable], ast_node=statement))
            else:
                assert statement.children[1].type == "pattern_list" or statement.children[1].type == "tuple_pattern"
                iterand = scope.create_temporary()
                iterator_block.operators.append(FunctionVarOperator(ITERATOR_FN_NAME, iterand, [iterator_or_iterable], ast_node=statement))
                convert_pattern_list_or_tuple_pattern(statement.children[1], scope, pattern_list_ops, iterand)
            
            iterator_block.operators.append(VarOperator(LOOP_OP, None, [iterand], ast_node=statement))

            # Comments can also be the direct children of for loops instead of being factored into the "block" statement of the loop body.
            body_start_idx = 5
            while statement.children[body_start_idx].type == "comment":
                body_start_idx += 1
            body_blocks = convert_block(statement.children[body_start_idx], scope)
            first_loop_block = body_blocks[0][0]
            iterator_block.add_successor(first_loop_block)
            if len(pattern_list_ops) > 0:
                first_loop_block.operators = pattern_list_ops + first_loop_block.operators
            

            # For loops in Python can have else blocks. These are executed after the loop if a break statement
            # was not used to exit the loop.
            start_of_else: Optional[BasicBlock] = None
            if len(statement.children) > body_start_idx + 1: # len = idx + 1
                assert len(statement.children) == body_start_idx + 2
                assert statement.children[body_start_idx + 1].type == "else_clause"
                else_blocks = convert_block(statement.children[body_start_idx + 1].child_by_field_name("body"), scope)
                start_of_else = else_blocks[0][0]

                # current_block is now the block AFTER the for loop.
                if not any(map(lambda x: isinstance(x[1], Break), body_blocks)): # If there wasn't a break statement in the loop body...
                    current_block = else_blocks[-1][0] # ...then there's no need for the last else block to be a separate basic block from current_block because control flows always from the former to the latter.
                    else_blocks = else_blocks[:-1] # Remove the last block from else_blocks to prevent it from being added twice.
                else:
                    current_block = BasicBlock([], [], []) 

                iterator_block.add_successor(start_of_else)
            else: # There is no else clause.
                current_block = BasicBlock([], [], []) # the block AFTER the for loop.
                iterator_block.add_successor(current_block)

            # current_block should now be re-defined by this point to be to block after the loop.
            # This is important for successor assignment.
            
            for for_block, successor in body_blocks:
                if isinstance(successor, Next):
                    for_block.add_successor(iterator_block)
                    basic_blocks.append((for_block, None))
                elif isinstance(successor, Break):
                    for_block.add_successor(current_block)
                    basic_blocks.append((for_block, None))
                elif isinstance(successor, Continue):
                    for_block.add_successor(iterator_block)
                    basic_blocks.append((for_block, None))
                else:
                    basic_blocks.append((for_block, successor))

            if start_of_else is not None:
                for else_block, successor in else_blocks:
                    if isinstance(successor, Next):
                        else_block.add_successor(current_block)
                        basic_blocks.append((else_block, None))
                    else:
                        basic_blocks.append((else_block, successor))
        
        elif statement.type == "while_statement":
            assert statement.children[0].type == "while"
            assert statement.children[2].type == ":"

            loop_condition_block = BasicBlock([], [], [])
            current_block.add_successor(loop_condition_block)
            loop_condition_block.operators.append(VarOperator(
                LOOP_OP, None,
                [expand_subexpression(statement.children[1], scope, loop_condition_block.operators)],
                ast_node=statement
            ))
            basic_blocks.append((current_block, None))
            basic_blocks.append((loop_condition_block, None))

            body_blocks = convert_block(statement.children[3], scope)
            loop_condition_block.add_successor(body_blocks[0][0])
            
            # while loops in python can have else blocks. These are executed after the loop finishes normally
            # without encountering a break statement.
            start_of_else: Optional[BasicBlock] = None
            if len(statement.children) > 4:
                assert len(statement.children) == 5
                assert statement.children[4].type == "else_clause"
                else_blocks = convert_block(statement.children[4].child_by_field_name("body"), scope)
                start_of_else = else_blocks[0][0]

                # current_block is now the block AFTER the for loop.
                if not any(map(lambda x: isinstance(x[1], Break), body_blocks)): # If there wasn't a break statement in the loop body...
                    current_block = else_blocks[-1][0] # ...then there's no need for the last else block to be a separate basic block from current_block because control flows always from the former to the latter.
                    else_blocks = else_blocks[:-1] # Remove the last block from else_blocks to prevent it from being added twice.
                else:
                    current_block = BasicBlock([], [], []) 

                loop_condition_block.add_successor(start_of_else)
            else: # There is no else clause.
                current_block = BasicBlock([], [], []) # the block AFTER the for loop.
                loop_condition_block.add_successor(current_block)
            
            # current_block should now be re-defined by this point to be to block after the loop.
            # This is important for successor assignment.

            for while_block, successor in body_blocks:
                if isinstance(successor, Next):
                    while_block.add_successor(loop_condition_block)
                    basic_blocks.append((while_block, None))
                elif isinstance(successor, Break):
                    while_block.add_successor(current_block)
                    basic_blocks.append((while_block, None))
                elif isinstance(successor, Continue):
                    while_block.add_successor(loop_condition_block)
                    basic_blocks.append((while_block, None))
                else:
                    basic_blocks.append((while_block, successor))

            if start_of_else is not None:
                for else_block, successor in else_blocks:
                    if isinstance(successor, Next):
                        else_block.add_successor(current_block)
                        basic_blocks.append((else_block, None))
                    else:
                        basic_blocks.append((else_block, successor))
        elif statement.type == "try_statement":
            assert statement.children[0].type == "try"
            assert statement.children[1].type == ":"
            assert statement.children[2].type == "block"

            # Codealign IR control flow is done as if exceptions don't exist. (Because exceptions can theoretically happen anywhere
            # in a python program, a truly correct control-flow model factoring in exceptions would have each operator
            # in a basic block by itself with one successor being an exit block representing a thrown exception or to the
            # nearest except block if one exists). This makes parsing try-catch blocks somewhat tricky to implement
            # in codealign IR. We model the control flow between basic blocks given by the try, except, and finally statements
            # 
            # An option not taken here would be to insert try, catch, and finally operators at the top of those blocks to
            # indicate their function when appropriate. Try and finally operators, however, have no operands or return values,
            # which would make them all identical in the eyes of dataflow constraint generation.

            basic_blocks.append((current_block, None))

            # Initialize 
            new_blocks: List[Tuple[BasicBlock, Optional[BlockSuccessorAssignment]]] = convert_block(statement.children[2], scope)
            current_block.add_successor(new_blocks[0][0])

            # If desired, insert try operator here as the start of the first block in exception_handling_blocks.

            # We want two copies of this because the predecessor variable will be updated as
            # exception clauses are processed, while final_try_block is not updated. It is needed
            # for correct successor assignments after processing the except clauses.
            final_try_block: BasicBlock = new_blocks[-1][0]
            predecessor: BasicBlock = new_blocks[-1][0]

            finally_blocks: Optional[List[Tuple[BasicBlock, Optional[BlockSuccessorAssignment]]]] = None

            except_all: bool = False # is there a catch all except statement (e.g. except:) without any exception(s) specified.
            for clause in statement.children[2:]: # Process all except and finally clauses.
                if clause.type == "except_clause":
                    assert clause.children[0].type == "except"
                    if clause.children[1].type == "as_pattern" or clause.children[1].type == "identifier" or clause.children[1].type == "tuple": # except ValueError as e or except ValueError
                        # This type of except clause has an implicit conditional. You only enter these blocks if the
                        # exception type matches.

                        except_condition_block = BasicBlock([], [], [])
                        new_blocks.append((except_condition_block, None))

                        exception_info_node = clause.children[1]
                        if exception_info_node.type == "as_pattern": # except ValueError as e
                            # An attribute can be found here with an imported exception like codealign.lang.python.SyntaxError.
                            # We include the whole name in this case; we want to capture the identity of the exception without decomposing it.
                            # (An ExceptionName instance with a Variable value doesn't make sense as a constant).
                            # Another option would be to reduce the name to only the class name, without qualification; this could increase false
                            # positive alignments (two catch operators align because they have the same exception name where they're actually 
                            # from different packages) while the current option could increase false negatives.
                            if exception_info_node.children[0].type == "identifier" or exception_info_node.children[0].type == "attribute": # The exception name.
                                exception_name = ExceptionName(exception_info_node.children[0].text.decode("utf8"))
                            else:
                                # convert_exception_to_tuple checks that this is actually a tuple with an assertion.
                                exception_name = convert_exception_tuple(exception_info_node.children[0], except_condition_block.operators, scope)
                            
                            as_pattern_target_node = exception_info_node.child_by_field_name("alias")
                            assert len(as_pattern_target_node.children) == 1
                            assert as_pattern_target_node.children[0].type == "identifier"
                            exception_variable = scope.variable_written(as_pattern_target_node.children[0].text.decode("utf8"))
                        elif exception_info_node.type == "tuple":
                            exception_name = convert_exception_tuple(exception_info_node, except_condition_block.operators, scope)
                            exception_variable = scope.create_temporary() # Will be unused in the IR; inserted for regularity.
                        else:
                            assert exception_info_node.type == "identifier" # except ValueError
                            exception_name = ExceptionName(exception_info_node.text.decode("utf8"))
                            exception_variable = scope.create_temporary() # Will be unused in the IR; inserted for regularity.

                        except_condition_block.operators.append(VarOperator(CATCH_OP, exception_variable, [exception_name], ast_node=clause))

                        # The if condition below can be true when the last BasicBlock generated by calling convert_block on the try block is empty.
                        # This condition is signaled with the empty_final_try_block flag.
                        predecessor.add_successor(except_condition_block)
                        predecessor = except_condition_block

                        assert clause.children[2].type == ":"
                        except_body_node = clause.children[3]
                    else:
                        assert clause.children[1].type == ":"
                        assert not except_all, "Cannot have multiple catch-all except clauses"
                        except_all = True
                        except_body_node = clause.children[2]
                    
                    except_body_blocks = convert_block(except_body_node, scope)
                    predecessor.add_successor(except_body_blocks[0][0])
                    new_blocks.extend(except_body_blocks)

                elif clause.type == "finally_clause":
                    assert clause.children[0].type == "finally"
                    assert clause.children[1].type == ":"
                    finally_blocks: List[Tuple[BasicBlock, Optional[BlockSuccessorAssignment]]] = convert_block(clause.children[2], scope)

                    if len(finally_blocks) == 1 and len(finally_blocks[0][0].operators) == 0:
                        # This is just finally: pass. An empty finally block complicates successor assignment in some cases
                        # (specifically with empty_multi_successor_blocks), and is semantically meaningless, so we simply ignore
                        # such blocks.
                        finally_blocks = None

            # The raise block makes explicit the implication that an exception not caught in any of the except clauses
            # is still raised. A raise block is unnecessary when there's an "except:" (except-all) clause that 
            # handles every exception type. The handling for the raise_block is distributed across several if statements
            # in the below code; these are marked by # RAISE BLOCK.
            # RAISE BLOCK
            if not except_all: # Create this here to ensure consistent BasicBlock ID ordering (makes debugging easier.) In particular, this must be before current_block.
                raise_block = BasicBlock([VarOperator(RAISE_OP, None, [])], [], [])

            current_block = BasicBlock([], [], []) # The block after the if statement or the finally block.

            if finally_blocks is None:
                try_except_successor = current_block
            else:
                try_except_successor = finally_blocks[0][0]

            # RAISE BLOCK
            if not except_all:
                if finally_blocks is None:
                    predecessor.add_successor(raise_block)
                else:
                    predecessor.add_successor(try_except_successor) # The raise block will be after the finally statements.
            
            for new_block, successor in new_blocks:
                if isinstance(successor, Next):
                    new_block.add_successor(try_except_successor)
                    basic_blocks.append((new_block, None))
                else:
                    basic_blocks.append((new_block, successor))

            # Normally, this would be invalid: how can an empty block have multiple successors
            # if it has no instructions (in particular, no conditional branch instructions)?
            # This is possible here as part of exception control-flow modeling. We handle this
            # case explicitly here instead of in clean_up_empty_blocks because the latter has
            # checks to ensure that empty blocks have no more than one successor.
            empty_multi_successor_blocks = [final_try_block] if len(final_try_block.operators) == 0 and len(final_try_block.successors) > 1 else []
            
            if finally_blocks is not None:
                for new_block, successor in finally_blocks:
                    if isinstance(successor, Next):
                        # RAISE BLOCK
                        if not except_all:
                            new_block.add_successor(raise_block)
                            if len(new_block.operators) == 0:
                                empty_multi_successor_blocks.append(new_block)
                        new_block.add_successor(current_block)
                        basic_blocks.append((new_block, None))
                    else:
                        basic_blocks.append((new_block, successor))

            
            # RAISE BLOCK
            if not except_all: # Do the append here to ensure that BasicBlocks are added in ID order (makes debugging easier.)
                basic_blocks.append((raise_block, None))

            # Remove empty blocks with multiple successors. (Empty blocks with one or fewer successors are removed later.)
            for emsb in empty_multi_successor_blocks:
                former_predecessors = emsb.predecessors
                former_successors = emsb.successors
                

                for predecessor in emsb.predecessors:
                    for successor in emsb.successors:
                        if successor not in predecessor.successors: # Can happen with try: finally blocks.
                            predecessor.add_successor(successor)
                
                for predecessor in former_predecessors:
                    predecessor.successors.remove(emsb)
                for successor in former_successors:
                    successor.predecessors.remove(emsb)
                
                basic_blocks.remove((emsb, None))


        elif statement.type == "with_statement":
            assert statement.children[0].type == "with"
            assert statement.children[1].type == "with_clause"
            assert statement.children[2].type == ":"
            assert statement.children[3].type == "block"

            # Because they are exception-related, with statements are difficult to implement in codealign IR.
            # Here, the approach is to simply have a "with" operator and call that, leaving everythinge else the same.
            # An alternative approach is to keep track of the object declared in the with statement (the "as" alias or,
            # if that does not exist, a temporary variable) and call the __enter__ and __exit__ methods on that variable.
            # This is complicated by the fact that __exit__ takes three arguments related to a potential exception thrown.
            # These could be None or just omitted, though neither of these are strictly accurate.

            with_clause = statement.children[1]
            assert len(with_clause.children) == 1
            with_item = with_clause.children[0]
            assert with_item.type == "with_item"
            assert len(with_item.children) == 1
            node = with_item.children[0]
            if node.type == "as_pattern":
                with_expression_value = expand_subexpression(node.children[0], scope, current_block.operators)
                alias = node.child_by_field_name("alias")
                assert alias.type == "as_pattern_target"
                assert len(alias.children) == 1
                assert alias.children[0].type == "identifier"
                result = alias.children[0].text.decode("utf8")
            else:
                with_expression_value = expand_subexpression(node, scope, current_block.operators)
                result = scope.create_temporary()

            current_block.operators.append(VarOperator(WITH_OP, result, [with_expression_value], ast_node=statement))

            with_blocks = convert_block(statement.children[3], scope)

            basic_blocks.append((current_block, None))
            current_block.add_successor(with_blocks[0][0])
            current_block = BasicBlock([], [], [])

            for block, successor in with_blocks:
                if isinstance(successor, Next):
                    block.add_successor(current_block)
                    basic_blocks.append((block, None))
                else:
                    basic_blocks.append((block, successor))
            
        elif statement.type == "assert_statement":
            assert statement.children[0].type == "assert"
            assert_condition = expand_subexpression(statement.children[1], scope, current_block.operators)
            operands = [assert_condition]
            if len(statement.children) > 2:
                assert len(statement.children) == 4
                assert statement.children[2].type == ","
                operands.append(expand_subexpression(statement.children[3], scope, current_block.operators))
            # A subtle issue with this implementation that is hand-waved away here is that assert statements
            # function somewhat like if statements. If the condition is false, the optional message expression
            # (which is usually just a string) is not executed.
            current_block.operators.append((FunctionVarOperator("assert", None, operands, ast_node=statement)))
        elif statement.type == "delete_statement":
            # Delete statements are handled differently depending on what is being deleted. Variable deletion
            # can be best represented by removing that variable from the scope. Menewhile, deleting an element
            # of a data structure, like a dictionary, modifies that data structure; it does not change the variables
            # which are available at this scope.
            assert statement.children[0].type == "del"
            deleted = clean_expression(statement.children[1]) # You can successfully del a variable if it's wrapped in parentheses.
            if deleted.type == "identifier":
                scope.delete_variable(deleted.text.decode("utf8"))
            else:
                deleted_expression = expand_subexpression(deleted, scope, current_block.operators)
                current_block.operators.append(VarOperator(DEL_OP, None, [deleted_expression], ast_node=statement))
        elif statement.type == "import_statement":
            assert statement.children[0].type == "import"
            module_node = statement.children[1]
            # From a codealign IR perspective, modules are variables, and their attributes can be accessed like those of objects. 
            if module_node.type == "dotted_name":
                module_name_base_text = module_node.children[0].text.decode("utf8") # Add only the first item in the dotted name to the scope. Subsequent uses of that dotted name will be interpreted as attribute accesses.
                module = scope.variable_written(module_name_base_text)
                current_block.operators.append(VarOperator(IMPORT_OP, module, [ModuleName(module_name_base_text)], ast_node=statement))
            else:
                assert module_node.type == "aliased_import"
                module_name_node = module_node.child_by_field_name("name")
                alias_node = module_node.child_by_field_name("alias")
                assert module_name_node.type == "dotted_name"
                assert alias_node.type == "identifier"
                current_block.operators.append(VarOperator(
                    IMPORT_OP,
                    scope.variable_written(alias_node.text.decode("utf8")),
                    [Field(module_name_node.children[0].text.decode("utf8"))],
                    ast_node=statement
                ))
        elif statement.type == "import_from_statement":
            assert statement.children[0].type == "from"
            assert statement.children[1].type == "dotted_name"
            assert statement.children[2].type == "import"
            module_node = statement.children[1]
            module_name_base_text = module_node.children[0].text.decode("utf8")

            # Use temporaries so as not to interfere with other variables in the scope
            temporary_variable = scope.create_temporary()
            current_block.operators.append(VarOperator(IMPORT_OP, temporary_variable, [ModuleName(module_name_base_text)], ast_node=statement))

            for module_name_component in module_node.children[1:]:
                if module_name_component.type == ".":
                    continue
                assert module_name_component.type == "identifier"
                new_temporary = scope.create_temporary()
                current_block.operators.append(VarOperator(MEMBER_ACCESS_OP, new_temporary, [temporary_variable, Field(module_name_component.text.decode("utf8"))], ast_node=statement))
                temporary_variable = new_temporary
            
            # As with a normal import statement, further dots are reduced to just standard attribute accesses.
            if statement.children[3].type == "dotted_name":
                imported_component_text = statement.children[3].children[0].text.decode("utf8")
                current_block.operators.append(VarOperator(MEMBER_ACCESS_OP, scope.variable_written(imported_component_text), [temporary_variable, Field(imported_component_text)], ast_node=statement))
            else:
                assert statement.children[3].type == "aliased_import"
                aliased_import = statement.children[3]
                assert aliased_import.children[0].type == "dotted_name"
                assert aliased_import.children[1].type == "as"
                assert aliased_import.children[2].type == "identifier"

                current_block.operators.append(VarOperator(
                    MEMBER_ACCESS_OP,
                    scope.variable_written(aliased_import.children[2].text.decode("utf8")), # the alias is the name of the variable
                    [temporary_variable, Field(aliased_import.children[0].text.decode("utf8"))],  # but the actual name of the imported module is the name of the field.
                    ast_node=statement
                ))
        elif statement.type == "global_statement":
            assert statement.children[0].type == "global"
            assert statement.children[1].type == "identifier"
            scope.declare_global(statement.children[1].text.decode("utf8"))
        elif statement.type == "function_definition":
            continue # Ignore function definitions because they are not operations.
        elif statement.type == "pass_statement":
            continue
        else:
            raise NotImplementedError(f"No code implemented to handle statements of type {statement.type}.")
    
    basic_blocks.append((current_block, Next()))
    return basic_blocks

def finish_control_flow_proxies(fn_blocks: List[BasicBlock]) -> List[BasicBlock]:
    """Expressions that result in control flow may be represented as ProxyControlFlowExpressions. This 
    method converts these to real control flow.
    """
    blocks = collections.deque(fn_blocks)
    output: List[BasicBlock] = []

    def split_block_at_proxy(block: BasicBlock, i: int) -> Tuple[BasicBlock, ProxyControlFlowExpression, BasicBlock]:
        proxy = block.operators[i]
        after_operators = block.operators[i + 1:]
        block.operators = block.operators[:i]
        new_block = BasicBlock(after_operators, [], []) # Insert to comprehension_blocks later so that this is listed after the main comprehension blocks in the function.

        # We split the current block into two pieces. block is the first half, and new_block
        # is the second half. new_block, then, must have all of block's successors, and block
        # will have no successors (until we assign them later).
        for successor in block.successors:
            successor.predecessors.remove(block)
            new_block.add_successor(successor)
        block.successors = []

        return block, proxy, new_block

    while len(blocks) > 0:
        block = blocks.popleft()
        for i in range(len(block.operators)):
            if isinstance(block.operators[i], ProxyComprehension):
                comprehension_blocks: List[BasicBlock] = [] # Contains the new basic blocks created in processing the comprehension.
                block, proxy, new_block = split_block_at_proxy(block, i)

                if isinstance(proxy, ProxyListComprehension):
                    block.operators.append(VarOperator(ARRAY_INITIALIZER_OP, proxy.result, [], proxy.ast_node))
                elif isinstance(proxy, ProxyDictionaryComprehension):
                    block.operators.append(VarOperator(DICTIONARY_INITIALIZER_OP, proxy.result, [], proxy.ast_node))
                else:
                    assert isinstance(proxy, ProxySetComprehension)
                    block.operators.append(FunctionVarOperator("set", proxy.result, [], ast_node=proxy.ast_node))

                current_block = block
                false_successor = new_block

                for clause in proxy.clauses:
                    if isinstance(clause, ProxyComprehension.ForClause):
                        # if there's already operators in this block we have to create another one
                        # so that these operators aren't "rerun" as a part of the loop condition each time.
                        if len(current_block.operators) > 0 or len(clause.pre_iterator_block_operators) > 0 or current_block == block:
                            current_block.operators.extend(clause.pre_iterator_block_operators)
                            inner = BasicBlock([], [], [])
                            comprehension_blocks.append(inner)
                            current_block.add_successor(inner)
                            current_block = inner
                        
                        inner = BasicBlock([], [], [])
                        comprehension_blocks.append(inner)
                        current_block.add_successor(inner)
                        current_block.add_successor(false_successor)
                        current_block.operators.extend(clause.iterator_block_operators)

                        false_successor = current_block

                        # Pattern lists are decomposed into a sequence of calls to next(). This is the block that contains those calls.
                        if len(proxy.body) > 0:
                            current_block = inner
                            inner = BasicBlock([], [], [])
                            comprehension_blocks.append(inner)
                            current_block.add_successor(inner)
                            inner.operators.extend(clause.body_operators)

                        current_block = inner
                    else:
                        assert isinstance(clause, ProxyComprehension.IfClause)
                        inner = BasicBlock([], [], [])
                        comprehension_blocks.append(inner)
                        current_block.add_successor(inner)
                        current_block.add_successor(false_successor)
                        current_block.operators.extend(clause.condition_block_operators)

                        current_block = inner
                
                if isinstance(proxy, ProxyListComprehension):
                    current_block.operators.append(VarOperator(MEMBER_ACCESS_OP, proxy.temporary_variables[0], [proxy.result, Field("append")], proxy.ast_node))
                    current_block.operators.extend(proxy.body)
                    current_block.operators.append(FunctionVarOperator(proxy.temporary_variables[0], proxy.temporary_variables[1], [proxy.body_result], ast_node=proxy.ast_node)) # the return value here is not necessary but is done for consistency.
                elif isinstance(proxy, ProxyDictionaryComprehension):
                    current_block.operators.extend(proxy.body)
                    current_block.operators.append(VarOperator(SUBSCRIPT_OP, proxy.temporary_variables[0], [proxy.result, proxy.body_result[0]], proxy.ast_node)) # proxy.result is the dictionary, proxy.body_result is the key in the key:value pair.
                    current_block.operators.append(VarOperator(STORE_OP, proxy.temporary_variables[0], [proxy.temporary_variables[0], proxy.body_result[1]], proxy.ast_node)) # proxy.temporary_variables[0] is the store address, proxy.body_result is value in the key:value pair.
                else:
                    assert isinstance(proxy, ProxySetComprehension)
                    current_block.operators.append(VarOperator(MEMBER_ACCESS_OP, proxy.temporary_variables[0], [proxy.result, Field("add")], proxy.ast_node))
                    current_block.operators.extend(proxy.body)
                    current_block.operators.append(FunctionVarOperator(proxy.temporary_variables[0], proxy.temporary_variables[1], [proxy.body_result], ast_node=proxy.ast_node))
                current_block.add_successor(false_successor)

                # Finally, add the latter half of the split block. It could contain more list comprehensions.
                comprehension_blocks.append(new_block)

                # Do this in reverse to preserve a better order for the blocks upon display.
                for addblock in reversed(comprehension_blocks):
                    blocks.appendleft(addblock) # Any of these blocks could contain more ProxyControlFlowExpressions.

                break # Do not continue looking through operators. In fact, because we split this basic block, there are no more, even though range() has iterations left.
            elif isinstance(block.operators[i], ProxyTernaryExpression):
                block, proxy, new_block = split_block_at_proxy(block, i)
                ternary_blocks: List[BasicBlock] = []
                assert isinstance(proxy, ProxyTernaryExpression) # redundant but here to let mypy make better suggestions in the IDE. Can remove later.

                block.operators.extend(proxy.condition_block)
                block.operators.append(VarOperator(IF_OP, None, [proxy.condition_result], ast_node=proxy.ast_node))

                def add_branch(operators: List[VarOperator], result: VarOperand):
                    """Add an intervining basic block between a true and false 
                    """
                    branch_block = BasicBlock([], [], [])
                    ternary_blocks.append(branch_block)
                    block.add_successor(branch_block)
                    branch_block.add_successor(new_block)
                    branch_block.operators.extend(operators)
                    # Unfortunately, if the branch is a non-leaf expression (such as x + y), this will
                    # result in an awkward assignment to a temporary varaible. Thus, a 
                    # ternary like z = x + y if c else ...
                    # will have the "true" branch converted to t0 = x + y; z = t0, which is not ideal.
                    # However, this should be corrected for in copy propagation.
                    branch_block.operators.append(VarOperator(COPY_OP, proxy.result, [result]))
                
                add_branch(proxy.true_block, proxy.true_result)
                add_branch(proxy.false_block, proxy.false_result)

                ternary_blocks.append(new_block)

                # Do this in reverse to preserve a better order for the blocks upon display.
                for b in reversed(ternary_blocks):
                    blocks.appendleft(b)

                break # Because we split this block up, there are no elements after position i, though the range() iterator will still have some iterations left.
        output.append(block)
    
    return output
    

def convert_parameters(parameter_node: Node, scope: Scope) -> List[Parameter]:
    assert parameter_node.type == "parameters"
    assert parameter_node.children[0].type == "("
    assert parameter_node.children[-1].type == ")"

    parameters = []
    for child in parameter_node.children[1:-1]:
        if child.type == "," or child.type == "comment":
            continue
        if child.type == "identifier":
            parameters.append(scope.create_parameter(child.text.decode("utf8")))
        elif child.type == "typed_parameter":
            # Typed parameters' name fields are None for some reason.
            parameters.append(scope.create_parameter(child.children[0].text.decode("utf8")))
        elif child.type == "list_splat_pattern":
            assert child.children[0].type == "*"
            parameters.append(scope.create_parameter(child.children[1].text.decode("utf8")))
        elif child.type == "dictionary_splat_pattern":
            assert child.children[0].type == "**"
            parameters.append(scope.create_parameter(child.children[1].text.decode("utf8")))
        elif child.type == "positional_separator":
            continue
        elif child.type == "keyword_separator":
            continue
        else:
            # TODO: Handle default parameters' initialization.
            assert child.type == "default_parameter" or child.type == "typed_default_parameter", f"Cannot process parameter of type {child.type}"
            parameters.append(scope.create_parameter(child.child_by_field_name("name").text.decode("utf8")))
    
    return parameters


def convert_function(definition: Node) -> Function:
    assert definition.type == "function_definition"

    # Tree-sitter can effectively recover from some errors, but other times it inserts an ERROR node.
    # This can cause problems for generating IR. Thus, we do an initial check to make sure that the
    # given AST does not have any ERROR nodes.
    error_check(definition)

    function_name = definition.child_by_field_name("name").text.decode("utf8")

    # The global registry begins empty for each function separately. This is because this software is designed
    # to operate on isolated functions out of context; doing it this way ensures consistent results regardless of
    # the amount of context provided.
    global_scope = Scope()
    function_scope = Scope(global_scope)
    parameters = convert_parameters(definition.child_by_field_name("parameters"), function_scope)

    function_body = convert_block(definition.child_by_field_name("body"), function_scope)
    basic_blocks = [b[0] for b in function_body] # Items in position b[1] are block succession assignments, which aren't relevant after this point.
    basic_blocks = finish_control_flow_proxies(basic_blocks)
    clean_up_empty_blocks(basic_blocks)
    return Function(function_name, basic_blocks, parameters)


def parse(code: bytes) -> List[Function]:
    """Parse python code using tree-sitter, and convert it into variable-oriented IR function form.
    """
    ast = parser.parse(code)
    cursor = ast.walk()
    assert cursor.node.type == "module"
    functions = []

    def find_functions(node: Node):
        """Find all functions not nested inside other functions in python code.
        """
        for child in node.children:
            if child.type == "function_definition":
                functions.append(convert_function(child))
            if child.type == "class_definition":
                functions.append(find_functions(child.child_by_field_name("body"))) # Classes can be nested.
            if child.type == "decorated_definition":
                i = 0
                while i < len(child.children):
                    if child.children[i].type == "function_definition":
                        functions.append(convert_function(child.children[i]))
                        break
                    i += 1
                else: # This else is attached to the while loop. It is executed if the break statement is not executed.
                    raise SemanticError("No function definition found with decorator!")
                assert i + 1 == len(child.children), "The function definition must be the last field of a decorated function node!"
    
    find_functions(cursor.node)
    return functions

