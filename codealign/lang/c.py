"""Interact with tree_sitter to convert C code into IR form.
"""

from abc import ABC
from typing import Tuple, Set
import itertools
from pathlib import Path

import tree_sitter_c
from tree_sitter import Language, Parser, Node

from ..ir import *

C_LANGUAGE = Language(tree_sitter_c.language())
parser = Parser(C_LANGUAGE)

class SemanticError(Exception):
    pass

class ParsingError(Exception):
    pass

class VariableRegistry:
    def __init__(self, parent_registry: 'VariableRegistry' = None):
        """Maps variable names to IR variable objects. If this scope is contained inside another scope,
        that scope can be accessed via parent_registry.

        :param parent_registry: The scope that this scope is found in. If this scope is the global scope, then

        """
        self.name2obj = {}
        self.parent_registry = parent_registry
        self.temporary_idx = 0

        # TODO: Add functionality to track function names and check if they conflict with the variable names.
    
    def variable_exists(self, variable_name: str):
        assert(type(variable_name) == str)

        if variable_name in self.name2obj:
            return True
        registry = self

        # The variable is not found at the innermost scope. However, it may exist in an outer scope.
        while registry.parent_registry is not None:
            registry = registry.parent_registry
            if variable_name in registry.name2obj:
                return True
        
        return False
    
    def check_variable(self, variable_name: str, declared: bool) -> Variable:
        """Check if this variable was defined in this scope or an outer scope. If so, return it; if not,
        add it to the variable registry.

        :param variable_name: the name of the variable to check
        :param declared: whether this variable is being declared at this point or not.
        :returns: the variable object corresponding to this variable.
        """
        assert(type(variable_name) == str)

        if variable_name in self.name2obj:
            if declared:
                raise SemanticError(f"Variable {variable_name} was already declared in this scope.")
            return self.name2obj[variable_name]
        
        # If this variable was declared (in this scope), we add it to the current scope.
        if declared:
            newvar = Variable(variable_name)
            self.name2obj[variable_name] = newvar
            return newvar
        
        registry = self

        # The variable is not found at the innermost scope. However, it may exist in an outer scope.
        while registry.parent_registry is not None:
            registry = registry.parent_registry
            if variable_name in registry.name2obj:
                return registry.name2obj[variable_name]
        # This variable name is not defined at any scope.

        # If this variable was not declared, we assume that it was a global variable.
        # We use this approach because this tool is designed to process single functions
        # individually, apart from the rest of the codebase. Thus, we have no way of knowing
        # which identifiers are actually global variables and which are just undeclared identifiers.
        else:
            newvar = GlobalVariable(variable_name)
            registry.name2obj[variable_name] = newvar
            return newvar
    
    def add_parameter(self, variable_name: str) -> Parameter:
        """Add a new parameter variable to this scope.

        :param variable_name: the name of the new parameter variable.
        """
        assert(type(variable_name) == str)
        p = Parameter(variable_name)
        self.name2obj[variable_name] = p
        return p

    def create_temporary(self):
        """Create a temporary variable with a unique name that does not exist in this scope or 
        any enclosing scope.
        """
        temporary_variable_name = f"t{self.temporary_idx}"
        while self.variable_exists(temporary_variable_name):
            self.temporary_idx += 1
            temporary_variable_name = f"t{self.temporary_idx}"
        variable = Variable(temporary_variable_name, is_temporary=True)
        self.temporary_idx += 1
        self.name2obj[temporary_variable_name] = variable
        return variable

    def __repr__(self):
        """Return a string describing the contents of this variable registry and all parent registries.
        """
        variables = [repr(v) for _, v in self.name2obj.items()]
        outstr = "VariableRegistry(" + ", ".join(variables) + ")"

        if self.parent_registry is not None:
            outstr += " ->\n  " + repr(self.parent_registry)

        return outstr
                

def print_types_recursively(node: Node):
    print(node.type)
    for child in node.children:
        print_types_recursively(child)


def print_immediate_children(root: Node):
    for i, child in enumerate(root.children):
        print(child.type, end=": ")
        print(root.field_name_for_child(i))


def check_expression_leaf(expression: Node, variable_registry: VariableRegistry) -> Optional[Node]:
    if expression.type == "identifier":
        return variable_registry.check_variable(expression.text.decode("utf8"), declared=False)
    if expression.type == "number_literal":
        return NumberConstant(expression.text.decode("utf8"))
    if expression.type == "string_literal": # Note: children of string literal are " and "
        return StringLiteral(expression.text.decode("utf8")) # the 'text' field includes the ""
    if expression.type == "char_literal":
        return CharLiteral(expression.text.decode("utf8"))
    # true, TRUE, false, FALSE, and NULL are not keywords in C but tree-sitter recognizes them with their own node types anyway.
    if expression.type == "null":
        return NumberConstant("0")
    if expression.type == "true":
        return NumberConstant("1")
    if expression.type == "false":
        return NumberConstant("0")
    if expression.type == "concatenated_string":
        assert all(child.type == "string_literal" for child in expression.children)
        text = [child.text.decode("utf8") for child in expression.children]
        assert all(t[0] == '"' and t[-1] == '"' for t in text)
        return StringLiteral('"' + ''.join(t[1:-1] for t in text) + '"')
    
    return None

def clean_expression(expression: Node) -> Node:
    while expression.type == "parenthesized_expression":
        assert(expression.children[0].type == "(")
        assert(expression.children[-1].type ==")")
        assert(len(expression.children) == 3)
        expression = expression.children[1]
    return expression

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
    assert operator.type in ASSIGNMENT_SUBOPS, f"{operator.type} not a valid C assignment operator."
    return ASSIGNMENT_SUBOPS[operator.type]

# C expressions are recursively defined entities. The IR requres expressions be represented as a sequence
# of single operations, with the result of each stored in a variable. A set of three functions, which form
# a three-way recursive system, help perform this conversion. The recursion can be started from any of the
# three functions, depending on what is desired (see the end of this comment for details).
#
# The relationships between the functions are as follows, where -> indicates a call.
#  ... -> bind_expression -> convert_operator -> expand_subexpression -> ...
#
########## convert_operator ##########
# convert_operator converts a tree-sitter AST node into partial components for an operator: it extracts the 
# opcode and the arguments but does not create a VarOperator IR object. Rather, it returns these components.
# This is because an IR VarOperator requires one additional piece of information: where to store the result
# of the computation. Determining this is the responsibility of the bind_expression function.
#
########## bind_expression ##########
# The role of the bind_expression function is to determine in what variable the results of a given operator
# should be stored. Determining what the operator and its arguments actually are is the responsibility of 
# convert_operator; bind_expression calls convert_operator for this purpose. bind_expression stores 
# the result of an operator in a variable from the original program if possible (e.g. in "int a = b + c;",
# the result will be stored in "a"). Sometimes, however, the operator's result is, in the original program,
# used as part of a larger expression (as "4 * b" is in "int a = 4 * b + 1;"). In this case, bind_expression
# will assign the operation's result to a temporary variable.
#
########## expand_subexpression ##########
# expand_subexpression is a wrapper around bind_expression that helps prevent unnecessary "copy" operations.
# A "copy" operation stores the value of a variable or constant into a variable, as in "x = 3;" or "y = x;"
# If expand_subexpression encounters a "leaf" node, (a variable or constant), it simply returns it. Otherwise,
# it forwards the expression to bind_expression. bind_expression always binds the expression it recieves to 
# a variable, creating a fresh temporary if necessary. This is undesirable when the expression is simply 
# a variable or constant itself used as an argument to another operator. For example, if calls were routed
# directly to bind_expression, "x = y + 2;" would be converted into
#   t0 = y
#   t1 = 2
#   x = + t0 t1
# which has two redundant copy operators.
#
# 
#
########## initiating recursion ##########
# There are situations where calling each of these functions to initiate expression parsing makes sense.
# Call bind_expression if
# - The top-level expression can be converted to an operator no matter what. (This is handy for genuine 
#   copy operations present in the original code like x = y;).
# Call expand_subexpression if
# - You want a value (variable or constant) that is the result of the entire expression.
# Call convert_operator if
# - You want to bind the operation and arguments to a variable yourself.

def bind_expression(expression: Node, variable_registry: VariableRegistry, expression_ops: List[Operator]) -> Variable:
    expression = clean_expression(expression)
    if expression.type == "assignment_expression":
        # expression.children[0]: (left) - the lhs of the assignment
        # expression.children[1]: (operator) - either = or +=
        # expression.children[2]: (right) - the rhs of the assignment

        ## Process the LHS
        lhs = expression.child_by_field_name("left")
        if (lhs.type == "identifier"):
            result_var = variable_registry.check_variable(lhs.text.decode("utf8"), declared=False)
            store_required = False
        else:
            assert lhs.type != "assignment_expression", "Assignment expressions should be on the rhs"
            # Handle the case where the LHS is an expression, as in 'point->x'.
            result_var = bind_expression(lhs, variable_registry, expression_ops)
            store_required = True
        
        ## Process the RHS
        rhs = expression.child_by_field_name("right")
        assignment_operator = expression.child_by_field_name("operator")
        if store_required:
            # Here, we hand-wave away a subtle type mismatch. Consider the expression pt->x = pt->y = var;.
            # This will decompose into:
            #   t0 = pt->x
            #   t1 = pt->y
            #   store t1 var
            #   store t0 t1
            # Here, t0 and t1 represent memory addresses: where to store the result. However, var represents 
            # the value of the variable var. In the first store instance, the second argument to "store" 
            # takes a value. Meanwhile, the second argument of the second store instance takes a memory address
            # rather than a value. We can ignore this subtle distinction here because it doesn't matter for
            # generating dataflow constraints for alignment. 
            # 
            # This problem could be solved in several ways:
            # 1. By having two different types of store operations, one for each case.
            # 2. The above presentation is simplified for clarity; our store operators actually have a return value:
            # t0 = store t0 t1, for example. Store operators could return a different temporary variable, representing
            # a value.
            value_to_store = expand_subexpression(rhs, variable_registry, expression_ops)
            if assignment_operator.type != "=": # could be +=, -=, etc.
                subopcode = get_assignment_subopcode(assignment_operator)
                # This also commits the subtle type mismatch described above: result_var is treated as both 
                # an address and a value. (It is treated as a value in this operand and in the lhs processing
                # code but as an address in the STORE_OP).
                temporary_variable = variable_registry.create_temporary()
                expression_ops.append(VarOperator(subopcode, temporary_variable, [result_var, value_to_store], ast_node=expression))
                value_to_store = temporary_variable # so the correct value_to_store is included in the arguments to the store operator being constructed.
            opcode = STORE_OP # arguments (address, value to store)
            operands = [result_var, value_to_store]
            opname = None
            ast_node = expression
        else:
            if assignment_operator.type == "=":
                if rhs.type == "assignment_expression" or rhs.type == "update_expression":
                    # Here, we have a nested assignment expression, e.g. a = b = 1;
                    operand = bind_expression(rhs, variable_registry, expression_ops)
                    # If the right hand side is an assignment expression, then it will bind the result
                    # to some variable (returned as operand just above). In the case of a = b = 1, that variable
                    # will be "b". The only thing left to do is copy the result from that variable into the one on 
                    # the lhs of this assignment (in the example, "a").
                    opcode = COPY_OP
                    operands = [operand]
                    opname = None
                    ast_node = None
                else: # The rhs is not an assignment expression.
                    # This is the "typical" case for an assignment expression: the lhs is simply a variable, and the 
                    # right hand side is a nonassignment expression. The assignment expression uses a plain =.
                    # Examples include x = x + 1; and a = foo(b, c); 
                    opcode, operands, opname, ast_node = convert_operator(rhs, variable_registry, expression_ops)
                    # opname is None unless the opcode is a function call.
            else: # the assignment is a +=, -=, etc.
                # Get a single variable or constant representing the rhs of the expression, then build the appropriate binary
                # operator operating on the lhs variable and the new rhs variable.
                rhs_result = expand_subexpression(rhs, variable_registry, expression_ops)
                # set up variables so that this operator can be built correctly.
                opcode = get_assignment_subopcode(assignment_operator)
                operands = [result_var, rhs_result]
                opname = None
                ast_node = expression
    elif expression.type == "update_expression":
        # ++ and --
        # The children of expression are 'argument' and 'operator'; the order depends on if prefix or postifix form is used.
        operand = expand_subexpression(expression.child_by_field_name("argument"), variable_registry, expression_ops)
        opcode = expression.child_by_field_name("operator").text.decode('utf8')
        assert opcode == "++" or opcode == "--"
        opcode = "+" if opcode == "++" else "-"

        # Someone could write something like 2++. This does not compile in C and is generally nonsensical.
        if not isinstance(operand, Variable):
            raise SemanticError(f"Cannot apply update operator {opcode} to expression \"{operand}\" of type \"{type(operand)}\"")
        
        if expression.field_name_for_child(0) == "operator": # is prefix (++i). Easy case.
            operands = [operand, NumberConstant("1")]
            opname = None
            result_var = operand
            ast_node = expression
        else: # is postfix (i++).
            # To handle postfix update expressions (e.g. i++), we decompose them into two operators: a copy operator
            # and a binary arithmetic operator (+ or -). Importantly, the return value (the variable representing the
            # overall expression) is the temporary variable, not variable being updated. This is because if i++ is used
            # in an expression, that use should reflect the unincremented version of value of i, as is the semantics of
            # the postfix update expression. The copy operator will eventually be eliminated as part of copy propagation
            # when converting to SSA.

            # Note that we can't use the operator-building code below because the final operator we want to create is the
            # increment/decrement operation, but we want to return the value from the copy operation that reflects the value
            # of operand before the increment/decrement.
            result_var = variable_registry.create_temporary()
            expression_ops.append(VarOperator(COPY_OP, result_var, [operand], None)) # must come first to reflect the value of operand before the update
            expression_ops.append(VarOperator(opcode, operand, [operand, NumberConstant("1")], expression))
            return result_var
    else: # Is not an assignment expression. Bind to a temporary variable
        opcode, operands, opname, ast_node = convert_operator(expression, variable_registry, expression_ops)
        result_var = variable_registry.create_temporary()
    
    # Reaching this point requires: result_var, opcode, operands, opname, and ast_node from the above if/else
    if opcode == FUNCTION_CALL_OP:
        assert(opname is not None)
        expression_ops.append(FunctionVarOperator(opname, result_var, operands, ast_node=ast_node))
    else:
        assert(opname is None)
        expression_ops.append(VarOperator(opcode, result_var, operands, ast_node=ast_node))

    return result_var

def expand_subexpression(expression: Node, variable_registry: VariableRegistry, expression_ops: List[VarOperator]) -> VarOperand:
    expression = clean_expression(expression)
    operand = check_expression_leaf(expression, variable_registry)
    if operand is None:
        operand = bind_expression(expression, variable_registry, expression_ops)
    return operand

def convert_operator(expression: Node, variable_registry: VariableRegistry, subexpression_ops: List[VarOperator]) -> Tuple[str, List[VarOperand], Optional[Union[str, Variable]], Optional[Node]]:
    assert expression.type != "assignment_expression" # Should be handled by bind_expression
    assert expression.type != "update_expression" # Should be handled by bind_expression
    expression = clean_expression(expression)

    leaf = check_expression_leaf(expression, variable_registry)

    if leaf:
        # A copy instruction, e.g. int x = y; or int x = 3;
        # This should ONLY be reached if there's a direct copy statement like this that occurs in the code.
        # We do not want copy instructions to occur as part of a subexpression, e.g. x = y + z; should not
        # be converted into x = t1; t1 = y + z;
        return (COPY_OP, [leaf], None, None)
    elif expression.type == "unary_expression":
        # expression.children[0]: (operator) - the operation being performed (e.g. !)
        # expression.children[1]: (argument) - the operand

        operand = expand_subexpression(expression.child_by_field_name("argument"), variable_registry, subexpression_ops)
        
        opcode = expression.child_by_field_name("operator").text.decode('utf8')
        return (opcode, [operand], None, expression)
    elif expression.type == "binary_expression":
        # expression.children[0]: (left) - the left operand
        # expression.children[1]: (operator) - the operation being performed (e.g. +, -)
        # expression.children[2]: (right) - the right operand
        left_operand = expand_subexpression(expression.child_by_field_name("left"), variable_registry, subexpression_ops)
        right_operand = expand_subexpression(expression.child_by_field_name("right"), variable_registry, subexpression_ops)
        opcode = expression.child_by_field_name("operator").text.decode('utf8')
        return (opcode, [left_operand, right_operand], None, expression)
    elif expression.type == "call_expression":
        # expression.children[0]: (function) - the name of the function.
        # expression.children[1]: (arguments; argument_list) - a list of arguments.

        # Get the name of the function
        name_node = expression.child_by_field_name("function")
        if name_node.type == "identifier":
            # If we call "expand_subexpression" with an identifier, it will interpret that identifier as a
            # variable, which is incorrect. Instead, we manually extract the function name here.
            function_name = name_node.text.decode("utf8") # function_name has type String

            # If the name of this function is an already defined variable. If it is, then that variable must
            # be a function call with a function pointer. If not, then we assume it is a standard call to a 
            # function with that name. (In theory, it could be a function call using a yet enencountered global variable,
            # but we don't have enough information to differentiate this case from the much more common normal 
            # function call).
            if variable_registry.variable_exists(function_name):
                function_name = variable_registry.check_variable(function_name, declared=False)
                # function_name now has type Variable
        else:
            # This could be the (*function_pointer)(args) syntax. If so, we extract the function pointer variable name.
            if name_node.type == "parenthesized_expression" and  name_node.children[1].type == "pointer_expression":
                assert name_node.children[1].children[0].type == "*"
                name_node = name_node.children[1].children[1]
                # It is possible for name_node to be an identifier here. However, if the function is in (*function_pointer)(args)
                # syntax, we assume that this function is a function pointer variable (possibly a global one) because 
                # this syntax is predominantly used for function pointers, unlike the normal function-call syntax.
            
            # Regardless of which syntax is used, the resulting expression could arbitrarily complex. We therefore must call
            # expand_subexpression to handle it.
            function_name = expand_subexpression(name_node, variable_registry, subexpression_ops)

        # Process the arguments of the function
        arguments = expression.child_by_field_name("arguments")
        assert(arguments.children[0].type == "(")
        assert(arguments.children[-1].type == ")")
        arguments = arguments.children[1:-1]

        operands = [] # the operands in this expression
        for argument in arguments:
            if argument.type == ",":
                continue

            operands.append(expand_subexpression(argument, variable_registry, subexpression_ops))
        
        return (FUNCTION_CALL_OP, operands, function_name, expression)
    elif expression.type == "pointer_expression":
        # expression.children[0]: (operator)
        # expression.children[1]: (argument) - the thing being dereferenced.
        operand = expand_subexpression(expression.child_by_field_name("argument"), variable_registry, subexpression_ops)
        operator = expression.child_by_field_name("operator").text.decode("utf8")
        # rename * operator to disambiguate it from multiplication
        if operator == "*":
            operator = POINTER_DEREFERENCE_OP
        return (operator, [operand], None, expression)
    elif expression.type == "conditional_expression":
        # expression.children[0]: (condition) - the conditional part of the ternary
        # expression.children[1]: ?
        # expression.children[2]: (consequence) - the 'true' part of the ternary
        # expression.children[3]: :
        # expression.children[4]: (alternative) - the 'false' part of the ternary

        condition = expand_subexpression(expression.child_by_field_name("condition"), variable_registry, subexpression_ops)
        consequence = expand_subexpression(expression.child_by_field_name("consequence"), variable_registry, subexpression_ops)
        alternative = expand_subexpression(expression.child_by_field_name("alternative"), variable_registry, subexpression_ops)

        return (TERNARY_OP, [condition, consequence, alternative], None, expression)
    elif expression.type == "field_expression":
        # expression.children[0]: (argument) - an expression that resolves to the struct
        # expression.children[1]: (operator) ->
        # expression.children[2]: (field) - the field being accessed.

        argument = expand_subexpression(expression.child_by_field_name("argument"), variable_registry, subexpression_ops)
        operator = expression.child_by_field_name("operator").text.decode("utf8")
        field = Field(expression.child_by_field_name("field").text.decode("utf8"))

        return (operator, [argument, field], None, expression)
    elif expression.type == "cast_expression":
        # expression.children[0]: (
        # expression.children[1]: (type) - the type being casted to
        # expression.children[2]: )
        # expression.children[3]: (value) - the expression to cast

        # Note that they type can sometimes be further broken down but we don't need to here.
        cast_type = TypeName(expression.child_by_field_name("type").text.decode("utf8"))
        value = expand_subexpression(expression.child_by_field_name("value"), variable_registry, subexpression_ops)

        return (CAST_OP, [cast_type, value], None, expression)
    elif expression.type == "subscript_expression":
        # expression.children[0]: (argument) - the name of the array, or an expression that resolves to an array.
        # expression.children[1]: [
        # expression.children[2]: (index) - an expression that resolves to an array index.
        # expression.children[3]: ]

        array = expand_subexpression(expression.child_by_field_name("argument"), variable_registry, subexpression_ops)
        index = expand_subexpression(expression.child_by_field_name("index"), variable_registry, subexpression_ops)

        return (SUBSCRIPT_OP, [array, index], None, expression)
    elif expression.type == "sizeof_expression":
        # sizeof can take a type or an expression as an argument. We handle each case separately.

        # if this sizeof is describing a type:
        # expressions.children[0]: sizeof
        # expressions.children[1]: (
        # expressions.children[2]: (type) - the type for which the size is being measured.
        # expressions.children[3]: )
        type_descriptior = expression.child_by_field_name("type")
        if type_descriptior is not None: # It is sizeof(type)
            arg_type = TypeName(type_descriptior.text.decode("utf8"))
            return (SIZEOF_OP, [arg_type], None, expression)
        else:
            # expression.children[0]: sizeof
            # expression.children[1]: expression
            argument = expand_subexpression(expression.children[1], variable_registry, subexpression_ops)
            return (SIZEOF_OP, [argument], None, expression)
    elif expression.type == "comma_expression":
        # expressions.children[0]: (left) - the first expression evaluated in the comma expression
        # expressions.children[1]: ,
        # expressions.children[2]: (right) - the second expression evaluated in the comma expression

        # If irrelevant values need not be bound to temporaries, then this can be refactored.
        _ = bind_expression(expression.child_by_field_name("left"), variable_registry, subexpression_ops) # subexpression_ops includes the left expression
        # We want this order because the left expression in a comma expression is evaluated first
        right = expression.child_by_field_name("right")
        if right.type == "assignment_expression" or right.type == "update_expression":
            # This is a corner case where the right expression in a comma expression is itself an assignment 
            # expression, i.e. x = (expr(), y = expr2());. In this case, x will copy y.
            # This reduces the comma expression to a copy operation, copying the result of the right expression.
            right_result = bind_expression(right, variable_registry, subexpression_ops)
            return (COPY_OP, [right_result], None, None)
        else:
            return convert_operator(right, variable_registry, subexpression_ops)
    elif expression.type == "initializer_list":
        expression.children[0].type == "{"
        expression.children[-1].type == "}"
        elements = []
        for element in expression.children[1:-1]:
            if element.type == ",":
                continue
            elements.append(expand_subexpression(element, variable_registry, subexpression_ops))
        return (ARRAY_INITIALIZER_OP, elements, None, expression)
    elif expression.type == "compound_literal_expression":
        # Form is (type) initializer_list. We only care about the initializer_list
        value = expression.child_by_field_name("value")
        assert value.children[0].type == "{" and value.children[-1].type == "}", f"Missing brackets on literal expression: {expression.text.decode('utf8')}"
        elements: list[VarOperand] = []
        for member in itertools.islice(value.children, 1, None, 2): # 2 to skip the commas.
            assert member.type == "initializer_pair", f"Expected initializer pair in compound literal {expression.text.decode('utf8')} but found {member.type}"
            # also has a field designator and value, but we only care about the value here.
            elements.append(expand_subexpression(member.child_by_field_name("value"), variable_registry, subexpression_ops))
        return (TUPLE_INITIALIZER_OP, elements, None, expression)
    elif expression.type == "ERROR":
        raise ParsingError(expression.text.decode("utf8")) 
    else:
        raise NotImplementedError(f"No code yet implemented to handle expressions of type '{expression.type}'")

def variable_name_from_declarator(declarator: Node) -> str:
    assert declarator.type == "identifier" or declarator.type == "pointer_declarator" or \
           declarator.type == "init_declarator" or declarator.type == "array_declarator" or \
           declarator.type == "function_declarator", f"Unexpected declarator type: {declarator.type}: {declarator.text.decode()}"
    
    if declarator.type == "init_declarator":
        # declarator.children[0] (declarator) - the name of the variable being declared, or a declarator for it
        # declarator.children[1] =
        # declarator.children[2] (value) - the expression used to initialize the variable.
        declarator = declarator.child_by_field_name("declarator")

    # Pointer declarators can be nested arbitrarily deep (e.g. int ****** x).
    while declarator.type == "pointer_declarator":
        # param_declarator.children[0] (None) is an *
        # param_declarator.children[1] (declarator) is another declarator - possibly a pointer.
        declarator = declarator.child_by_field_name("declarator")
    
    while declarator.type == "array_declarator":
        # declarator.children[0]: (declarator) - another declarator
        # declarator.children[1]: [
        # declarator.children[2]: (size; optional) - the array size
        # declarator.children[3]: ]
        declarator = declarator.child_by_field_name("declarator")
    
    if declarator.type == "function_declarator":
        # declarator.children[0]: (declarator)
        # declarator.children[1]: (parameters)
        declarator = declarator.child_by_field_name("declarator")

    if declarator.type == "parenthesized_declarator":
        assert declarator.children[0].type == "("
        assert declarator.children[2].type == ")"
        return variable_name_from_declarator(declarator.children[1]) # declarator.children[1] is unnamed.


    assert declarator.type == "identifier"
    return declarator.text.decode('utf8')


#
# These classes define how control flows after exiting a nested compound statement block (e.g. the body of a loop).
# By default, flow continues to the next block (indicated by the Next class). However, it may instead jump to an
# arbitrary label with goto (indicated by Label). In a for loop, the the update statement occurs in a block after the
# body. Normal flow (Next) goes to the update statement (e.g. i++), but a break skips it. Similarly, a continue
# statement directs control to the conditional loop-test block, not the update block.
#
class BlockSuccessorAssignment(ABC):
    pass

class Next(BlockSuccessorAssignment):
    pass

class Break(BlockSuccessorAssignment):
    pass

class Continue(BlockSuccessorAssignment):
    pass

class Label(BlockSuccessorAssignment):
    def __init__(self, label: str):
        self.label = label


def convert_declaration(declaration: Node, variable_registry: VariableRegistry, initialize: bool = True) -> List[VarOperator]:
    # statement.children[0]: (type) - the type of the parameter
    # statement.children[1]: (declarator) - the declarator
    # statement.children[2]: ; or ,
    # then repeated unnamed declarators and commas until the terminating ;.
    # The above can be offset by 1 if there is a type qualifier in position 1.
    # In a multi-initialization context, other declarators are unnamed, or the child_by_field_name defaults to the first.
    if declaration.children[1] == declaration.child_by_field_name("declarator"):
        declarators_start = 1
    else: 
        assert declaration.children[1].type == "type_qualifier" or declaration.children[1] == declaration.child_by_field_name("type"), "Unknown declaration pattern"
        assert declaration.children[2] == declaration.child_by_field_name("declarator"), "Unknown declaration pattern"
        declarators_start = 2
    # if the declarator is an init_declarator,
    # declarator.children[0] (declarator) - the name of the variable being declared, or a declarator for it
    # declarator.children[1] =
    # declarator.children[2] (value) - the expression used to initialize the variable.

    expression_ops = []
    for declarator in itertools.islice(declaration.children, declarators_start, None, 2): # step of 2 to skip the commas
        assert declarator.type != "," and declarator.type != ";"
        if declarator.type == "function_declarator":
            continue

        variable_name = variable_name_from_declarator(declarator)
        new_variable = variable_registry.check_variable(variable_name, declared=True)

        value = declarator.child_by_field_name("value") # if the declarator is an init declarator, this will not be None.
        if initialize and value is not None:
            partial_op_info = convert_operator(value, variable_registry, expression_ops)
            opcode, operands, opname, ast_node = partial_op_info # opname is None unless opcode is a function call.
            
            if opcode == FUNCTION_CALL_OP:
                expression_ops.append(FunctionVarOperator(opname, new_variable, operands, ast_node=ast_node))
            else:
                expression_ops.append(VarOperator(opcode, new_variable, operands, ast_node))
    
    return expression_ops # The variable was declared, but nothing was assigned to it. No computation was done.

def convert_compound_statement(body: Union[Node, List[Node]], variable_registry: VariableRegistry) -> List[Tuple[BasicBlock, List[BlockSuccessorAssignment]]]:
    if isinstance(body, list):
        statements = body
    elif body.type == "compound_statement":
        assert body.children[0].type == "{"
        assert body.children[-1].type == "}"
        statements = body.children[1:-1]
    else:
        statements = [body] # body will be an individual statement; we must wrap it in a list to use the code below.

    blocks = []
    current_block = BasicBlock([], [], [])
    for statement in statements:
        if statement.type == "declaration":
            current_block.operators.extend(convert_declaration(statement, variable_registry))
        elif statement.type == "expression_statement":
            # statement.children[0]: the expression
            # statement.children[1]: ;

            # Ignore empty statements. (i.e. just a semicolon)
            if len(statement.children) != 2:
                assert len(statement.children) == 1
                assert statement.children[0].type == ";"
            else:
                # The first return value of bind_expression is the exposed variable that contains the result
                # of the expression. At this point, we don't care about it, though it's useful in generating
                # operators from nested expressions.
                operators = []
                _ = bind_expression(statement.children[0], variable_registry, operators)
                current_block.operators.extend(operators)
        elif statement.type == "return_statement":
            if len(statement.children) <= 2:
                current_block.operators.append(VarOperator(RETURN_OP, None, [], statement))
            elif len(statement.children) == 3:
                assert statement.children[0].type == "return"
                assert statement.children[2].type == ";"
                return_value = expand_subexpression(statement.children[1], variable_registry, current_block.operators)
                current_block.operators.append(VarOperator(RETURN_OP, None, [return_value], statement))
            else:
                raise AssertionError(f"Invalid number of fields for return_statement node: {len(statement.children)}")
            
            blocks.append((current_block, None)) # A block ending in a return statement has no successors.
            return blocks # Everything after this statement in this block is irrelevant.
        elif statement.type == "break_statement":
            current_block.operators.append(VarOperator(BREAK_OP, None, [], statement))
            blocks.append((current_block, Break()))
            return blocks # Everything after this statement in this block is irrelevant.
        elif statement.type == "continue_statement":
            current_block.operators.append(VarOperator(CONTINUE_OP, None, [], statement))
            blocks.append((current_block, Continue()))
            return blocks
        elif statement.type == "if_statement":
            # statement.children[0]: if
            # statement.children[1]: (condition) - the conditional test
            # statement.children[2]: (consequence) - the if statement body; entered if true.
            # may have
            # statement.children[3]: (alternative) - the body of the else branch.
            condition_result = expand_subexpression(statement.child_by_field_name("condition"), variable_registry, current_block.operators)
            current_block.operators.append(VarOperator(IF_OP, None, [condition_result], statement))

            # Computation that occurs inside the conditional statement can be added to the end of the current block.
            # Control flow ends the current basic block.
            # We add "None" here even though this block has successors because the code immediately below
            # ensures that its successors are assigned.
            blocks.append((current_block, None)) 
            if_start_block = current_block # Need to keep a reference to this around so we can assign the first body block to it.a
            current_block = BasicBlock([], [], []) # New basic block after control flow is complete.

            # Convert the if-statement body. Create a new variable registry for the scope inside the if statement.
            body_blocks = convert_compound_statement(statement.child_by_field_name("consequence"), VariableRegistry(variable_registry))
            assert len(body_blocks) > 0, "Compound statement must have at least one corresponding basic block"
            if_start_block.add_successor(body_blocks[0][0])
            alternative_node = statement.child_by_field_name("alternative")
            if alternative_node is not None:
                # alternative_node.children[0]: else
                # alternative_node.children[1]: compound_statement or expression_statement: the else-clause body
                alternative_blocks = convert_compound_statement(alternative_node.children[1], VariableRegistry(variable_registry))
                assert len(alternative_blocks) > 0, "Compound statement must have at least one corresponding basic block"
                if_start_block.add_successor(alternative_blocks[0][0])
                body_blocks.extend(alternative_blocks)
            else:
                if_start_block.add_successor(current_block)
            

            for ifblock, successor_assignment in body_blocks:
                if isinstance(successor_assignment, Next):
                    ifblock.add_successor(current_block)
                    blocks.append((ifblock, None))
                else:
                    blocks.append((ifblock, successor_assignment)) # propagate other successor assignments, like Break, outside this scope.
            
        elif statement.type == "for_statement":
            # The relevant children (i.e. non-syntax children) are
            # initializer, condition, update, body

            loop_registry = VariableRegistry(variable_registry)

            initializer = statement.child_by_field_name("initializer")
            if initializer is not None:
                if initializer.type == "declaration":
                    initializer_result = convert_declaration(initializer, loop_registry)
                else:
                    initializer_result = []
                    _ = bind_expression(initializer, loop_registry, initializer_result)
                current_block.operators.extend(initializer_result)
            pre_loop_block = current_block # Keep a reference to the start of the loop arounds to assign successors later.
            blocks.append((pre_loop_block, None))
            current_block = BasicBlock([], [], []) # This is the block after the loop

            condition_operators = []
            condition = statement.child_by_field_name("condition")
            if condition is not None:
                condition_result = expand_subexpression(condition, loop_registry, condition_operators)
                condition_operators.append(VarOperator(LOOP_OP, None, [condition_result], statement))
            else: # an empty loop condition
                condition_operators.append(VarOperator(LOOP_OP, None, [NumberConstant("1")], statement))
            condition_block = BasicBlock(condition_operators, [], [])
            blocks.append((condition_block, None))
            pre_loop_block.add_successor(condition_block)
            condition_block.add_successor(current_block)
            
            update = statement.child_by_field_name("update")
            if update is not None:
                update_operators = []
                _ = bind_expression(update, loop_registry, update_operators)
                update_block = BasicBlock(update_operators, [], [])
            else:
                update_block = BasicBlock([], [], [])
            blocks.append((update_block, None))
            update_block.add_successor(condition_block)

            body_blocks = convert_compound_statement(statement.child_by_field_name("body"), loop_registry)
            condition_block.add_successor(body_blocks[0][0])

            for loopblock, successor_assignment in body_blocks:
                if isinstance(successor_assignment, Next):
                    loopblock.add_successor(update_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, Continue):
                    loopblock.add_successor(update_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, Break):
                    loopblock.add_successor(current_block) # current_block is the block after the loop
                    blocks.append((loopblock, None))
                else:
                    blocks.append((loopblock, successor_assignment)) # propagate other successor assignments out of this block.
        elif statement.type == "while_statement":
            # statement.children[0]: while
            # statement.children[1]: condition
            # statement.children[2]: body
            pre_loop_block = current_block
            blocks.append((current_block, None))
            current_block = BasicBlock([], [], []) # The block after the while loop

            condition_ops = []
            condition_result = expand_subexpression(statement.child_by_field_name("condition"), variable_registry, condition_ops)
            condition_ops.append(VarOperator(LOOP_OP, None, [condition_result], statement))
            condition_block = BasicBlock(condition_ops, [], [])
            blocks.append((condition_block, None))
            condition_block.add_successor(current_block)
            pre_loop_block.add_successor(condition_block)

            loop_registry = VariableRegistry(variable_registry)
            body_blocks = convert_compound_statement(statement.child_by_field_name("body"), loop_registry)
            condition_block.add_successor(body_blocks[0][0]) # The first block in the compound statement is the first block of the loop.
            
            for loopblock, successor_assignment in body_blocks:
                if isinstance(successor_assignment, Next):
                    loopblock.add_successor(condition_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, Continue):
                    loopblock.add_successor(condition_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, Break):
                    loopblock.add_successor(current_block) # break out of the loop; go to the block after the loop
                    blocks.append((loopblock, None))
                else:
                    blocks.append((loopblock, successor_assignment)) # propagate other successor assignments out of this block.
        elif statement.type == "do_statement":
            # statement.children[0]: do
            # statement.children[1]: (body) - the body of the loop
            # statement.children[2]: while
            # statement.children[3]: (condition) - the loop test
            # statement.children[4]: ;
            pre_loop_block = current_block # Keep a reference to the start of the loop arounds to assign successors later.
            blocks.append((pre_loop_block, None))
            current_block = BasicBlock([], [], []) # This is the block after the loop

            condition_ops = []
            condition_result = expand_subexpression(statement.child_by_field_name("condition"), variable_registry, condition_ops)
            condition_ops.append(VarOperator(LOOP_OP, None, [condition_result], statement))
            condition_block = BasicBlock(condition_ops, [], [])
            blocks.append((condition_block, None))
            condition_block.add_successor(current_block) # after the condition tests false, the loop exits to the next block after the loop.

            loop_registry = VariableRegistry(variable_registry)
            body_blocks = convert_compound_statement(statement.child_by_field_name("body"), loop_registry)
            pre_loop_block.add_successor(body_blocks[0][0])
            condition_block.add_successor(body_blocks[0][0])

            for loopblock, successor_assignment in body_blocks:
                if isinstance(successor_assignment, Next):
                    loopblock.add_successor(condition_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, Continue):
                    loopblock.add_successor(condition_block)
                    blocks.append((loopblock, None))
                elif isinstance(successor_assignment, Break):
                    loopblock.add_successor(current_block)
                    blocks.append((loopblock, None))
                else:
                    blocks.append((loopblock, successor_assignment)) # after the condition tests false, the loop exits to the next block after the loop.
        elif statement.type == "switch_statement":
            # condition_result is 'c' in switch (c) {...}
            condition_variable = expand_subexpression(statement.child_by_field_name("condition"), variable_registry, current_block.operators)

            blocks.append((current_block, None))
            current_if_block = current_block # Represents the block to which if statements will be added.
            prior_if_block: Optional[BasicBlock] = None # The if block representing the case condition before the current case condition.
            fallthrough_block: Optional[BasicBlock] = None # represents the previous block if there is no break statement at the end of that block.
            default_block: Optional[BasicBlock] = None # The first basic block from the default statement.

            # Contains the basic blocks from the individual case statements.
            switch_blocks: List[Tuple[BasicBlock, BlockSuccessorAssignment]] = []

            # For variables declared inside of the switch statement
            switch_registry = VariableRegistry(variable_registry)

            case_body = statement.child_by_field_name("body").children
            assert case_body[0].type == "{" and case_body[-1].type == "}"
            for substatement in case_body[1:-1]:
                if substatement.type == "case_statement":
                    if substatement.children[0].type == "case":
                        comparison_value = check_expression_leaf(clean_expression(substatement.child_by_field_name("value")), variable_registry)
                        if not isinstance(comparison_value, NumberConstant) and not isinstance(comparison_value, CharLiteral):
                            raise SemanticError("Case expression must be an integral constant expression.")
                        assert substatement.children[2].type == ":"

                        if prior_if_block is not None:
                            current_if_block = BasicBlock([], [], [])
                            switch_blocks.append((current_if_block, None))
                            prior_if_block.add_successor(current_if_block)

                        case_comparison_result = switch_registry.create_temporary()
                        current_if_block.operators.append(VarOperator("==", case_comparison_result, [condition_variable, comparison_value], substatement))
                        current_if_block.operators.append(VarOperator(IF_OP, None, [case_comparison_result], substatement))

                        case_blocks = convert_compound_statement(substatement.children[3:], switch_registry)

                        # If the previous block is falling through, assign it as a successor.
                        if fallthrough_block is not None:
                            fallthrough_block.add_successor(case_blocks[0][0])
                        
                        current_if_block.add_successor(case_blocks[0][0])

                        if isinstance(case_blocks[-1][1], Next):
                            # if this block doesn't end in a break or continue statement, fall through to the next case statement or default.
                            fallthrough_block: BasicBlock = case_blocks[-1][0]
                        else:
                            assert case_blocks[-1][1] is not None or case_blocks[-1][0].operators[-1].op == RETURN_OP
                            # We needn't do anything here to handle break or continue statements; those will be handled as part of the 
                            # block successor assignment loop at the end of switch statement processing.
                            fallthrough_block = None

                        prior_if_block = current_if_block
                        current_if_block = None # Not strictly necessary; added for clarity.

                        switch_blocks.extend(case_blocks)
                    else:
                        assert substatement.children[0].type == "default" and substatement.children[1].type == ":"
                        assert default_block is None, "Cannot have more than one default statement in a switch statement."

                        default_blocks = convert_compound_statement(substatement.children[2:], switch_registry)
                        default_block = default_blocks[0][0]

                        # The previous case statement fell through into the default block.
                        if fallthrough_block is not None:
                            fallthrough_block.add_successor(default_block)

                        if isinstance(default_blocks[-1][1], Next):
                            fallthrough_block: BasicBlock = default_blocks[-1][0]
                        else:
                            assert default_blocks[-1][1] is not None or default_blocks[-1][0].operators[-1].op == RETURN_OP
                            fallthrough_block = None
                        
                        switch_blocks.extend(default_blocks)
                elif substatement.type == "declaration":
                    convert_declaration(substatement, switch_registry, initialize=False)
                # Anything other than a case/default statement or a declaration is ignored.
            
            # Represents the block after the switch statement. Only has to be a new block if the
            # switch statement is non-empty.
            if prior_if_block is not None or default_block is not None:
                current_block = BasicBlock([], [], [])

            # When all cases are exhausted, control should flow to the default block if it exists.
            # Otherwise, control should flow to the block after the switch statement.
            if default_block is not None:
                if prior_if_block is not None: # Can be None if there are no case statements in the switch.
                    prior_if_block.add_successor(default_block)
                else:
                    current_if_block.add_successor(default_block)
            elif prior_if_block is not None: # Can be None if there are no case statements in the switch.
                prior_if_block.add_successor(current_block)
            # else: The situation where there's no default or case statements requires no successor assignment.

            # A break statement will have a link from the last case to the post-switch block
            # as well, but that's handled by the block successor assignment loop below.
            if fallthrough_block is not None:
                fallthrough_block.add_successor(current_block)

            for switch_block, successor_assignment in switch_blocks:
                if isinstance(successor_assignment, Break):
                    switch_block.add_successor(current_block)
                    blocks.append((switch_block, None))
                elif isinstance(successor_assignment, Next):
                    # "Next" successor assignment are dealt with above through fallthrough_block links.
                    blocks.append((switch_block, None))
                else:
                    blocks.append((switch_block, successor_assignment))
        elif statement.type == "compound_statement":
            # This is just a plain compound statement, not associated with a loop 
            # if statement, or other type of statement.
            
            blocks.append((current_block, None))
            # Model the separate scope introduced by a compound statement with a new VariableRegistry object.
            nested_blocks = convert_compound_statement(statement, VariableRegistry(variable_registry))
            current_block.add_successor(nested_blocks[0][0])
            current_block = BasicBlock([], [], []) # The block after the inner compound statement ends.
            for nested_block, successor_assignment in nested_blocks:
                if isinstance(successor_assignment, Next):
                    nested_block.add_successor(current_block)
                    blocks.append((nested_block, None))
                else:
                    blocks.append((nested_block, successor_assignment))
            if nested_blocks[-1][1] is None:
                assert nested_blocks[-1][0].operators[-1].op == RETURN_OP
                # A return statement in a nested scope terminates the outer scope as well.
                return blocks # propagate the return's termination of this path to this scope.
        elif statement.type == "comment":
            pass
        elif statement.type == "struct_specifier":
            pass
        elif statement.type == ";":
            pass
        elif statement.type == "ERROR":
            raise ParsingError(statement.text.decode("utf8"))
        # TODO: Implement processing for other types of statements.
        else:
            raise NotImplementedError(f"No code for handling statements of type {statement.type}")
    
    blocks.append((current_block, Next()))

    # for block, _ in blocks:
    #     print(block, end="\n\n")
    
    # print(variable_registry)

    return blocks

def process_function_declaration(declarator: Node, variable_registry: VariableRegistry) -> Tuple[str, List[Parameter]]:
    """Get the parameters to the function from the declarator. Add them to the VariableRegistry.
    Return the function name.

    :param declarator: The declarator of a function.
    :returns: The function name
    """
    while declarator.type == "pointer_declarator":
        # param_declarator.children[0] (None) is an *
        # param_declarator.children[1] (declarator) is another declarator - possibly a pointer, possibly an identifier.
        declarator = declarator.child_by_field_name("declarator")
    
    assert(declarator.type == "function_declarator")
    # declarator.children[0]: (declarator) - is the name of the function.
    # declarator.children[1]: (parameters) - the parameter list.
    parameter_node_list = declarator.child_by_field_name("parameters")

    parameters = []

    # The children of the parameter node list include the parameters but also parentheses and commas.
    # We filter out and record just the parameters themselves.
    for param_node in parameter_node_list.children:
        if param_node.type == "parameter_declaration":
            # param_node.children[0] (type) is the type of the parameter.
            # param_node.children[1] (declarator) is 
            #   - the name of the parameter if the parameter is a pass-by-value parameter.
            #   - a pointer_declarator if the parameter is a bass-by-reference parameter.
            param_declarator = param_node.child_by_field_name("declarator")

            if param_declarator is not None:
                parameter = variable_registry.add_parameter(variable_name_from_declarator(param_declarator))
                parameters.append(parameter)
    
    name = declarator.child_by_field_name("declarator")
    assert name.type == "identifier"
    return (name.text.decode("utf8"), parameters)

def error_check(node: Node):
    """Determine if there is an error node in this AST. If there is, raise a ParsingError.
    """
    if node.type == "ERROR":
        raise ParsingError(node.text.decode("utf8"))
    
    for child in node.children:
        error_check(child)

def clean_up_empty_blocks(blocks: List[BasicBlock]):
    remove_blocks: Set[BasicBlock] = set()
    # Ignore the entry block; it can be validly empty if the first statement in the function is a loop.
    for block in itertools.islice(blocks, 1, None):
        if len(block.operators) == 0:
            # Having more would require a branch, which would mean that the basic block is not empty.
            assert len(block.successors) <= 1, "An empty basic block should have one or fewer successors."
            successor = block.successors[0] if len(block.successors) == 1 else None

            # Only remove empty blocks if doing so would not decrease the number of successors of any predecessor block that
            # ends with a branch (i.e. with two or more successors.) Doing so affects the control flow graph, which may lead
            # to incorrect control dependence relationships.
            if any(len(predecessor.successors) > 1 and (successor is None or successor in predecessor.successors) 
                   for predecessor in block.predecessors):
                continue

            for predecessor in block.predecessors:
                # Find this block in the predecessor's successor list.
                this_block_index = None
                for i, pred_suc in enumerate(predecessor.successors):
                    if pred_suc == block:
                        assert this_block_index is None, "A basic block cannot have the same successor multiple times!"
                        this_block_index = i
                
                if successor is None or successor in predecessor.successors:
                    # sucessor in predecessor.successors can happen if block's predecessor also has block's successor
                    # as a successor. This happens in an if statement with an empty body.
                    del predecessor.successors[this_block_index]
                else:
                    predecessor.successors[this_block_index] = successor

            if successor is not None:
                # Also ensure that the empty block's successor's predecessor list is updated.
                successor.predecessors.remove(block) # Remove the empty block...
                # ...and replace it with all of the predecessors of the empty block
                for empty_predecessor in block.predecessors:
                    if empty_predecessor not in successor.predecessors:
                        successor.predecessors.append(empty_predecessor)
            
            # Remove these later so as not to damage the iterator.
            assert block not in remove_blocks, "A basic block should not be listed multiple times in a function's list of basic blocks!"
            remove_blocks.add(block)
    
    for block in remove_blocks:
        blocks.remove(block)

def remove_unreachable_blocks(fn: Function):
    """Remove basic blocks that are unreachable from the entry block. The modification
    is performed in-place.

    :param function: The function for which unreachable blocks should be removed.
    """
    reachable: set[BasicBlock] = set()

    def search(bb: BasicBlock):
        if bb in reachable:
            return
        reachable.add(bb)
        for successor in bb.successors:
            search(successor)
    search(fn.entry_block)

    # Only rebuild the basic_block list if we have to
    if len(reachable) < len(fn.basic_blocks):
        new_blocks = []
        for block in fn.basic_blocks:
            if block in reachable:
                new_blocks.append(block)
            else:
                # unreachable blocks may have successors that are reachable.
                for successor in block.successors:
                    successor.predecessors.remove(block)
        fn.basic_blocks = new_blocks

def function_ast2varform(definition: Node):
    """Converts a tree-sitter AST for a function definition into codealign IR variable form.

    :param definition: The root node of the function definition. Should be of type 'function_definition'.
    """
    assert(definition.type == "function_definition")

    # Tree-sitter can effectively recover from some errors, but other times it inserts an ERROR node.
    # This can cause problems for generating IR. Thus, we do an initial check to make sure that the
    # given AST does not have any ERROR nodes.
    error_check(definition)

    # The global registry begins uninitialized. This is because this software is designed to process one
    # function at a time, not necessarily the whole program. Thus, we don't know what global variables
    # have been declared in the program. When we encounter a variable in the code that has not been 
    # declared in any relevant scope, we consider it to be a global variable.
    global_registry = VariableRegistry() # The global scope
    function_registry = VariableRegistry(global_registry) # The highest level scope in the function itself.

    # The fields of the definition node depend on how the function is declared.
    # However, it has at least a type, a declarator, and a body.
    declarator = definition.child_by_field_name("declarator")
    function_name, parameters = process_function_declaration(declarator, function_registry)

    # definition.children[2] (body) is the function body. We need to convert all statements in the body into operators.
    blocks_with_metadata = convert_compound_statement(definition.child_by_field_name("body"), function_registry)

    basic_blocks = [b[0] for b in blocks_with_metadata]
    clean_up_empty_blocks(basic_blocks)

    func = Function(function_name, basic_blocks, parameters)
    remove_unreachable_blocks(func)
    return func

    
def parse(code: bytes) -> List[Function]:
    """Parse C code using tree-sitter, and convert it into variable-oriented IR function form.
    """
    ast = parser.parse(code)
    cursor = ast.walk()
    assert(cursor.node.type == "translation_unit")

    functions = []
    for child in cursor.node.children:
        if child.type == "function_definition":
            functions.append(function_ast2varform(child))
    
    return functions
