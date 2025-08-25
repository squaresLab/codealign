"""Intermediate representations of C code useful for code alignment.
"""

from abc import ABC
from typing import Dict, List, Union, Iterator, TypeVar, Optional, Literal

from tree_sitter import Node

#
# Constants
#
class Constant(ABC):
    def __repr__(self):
        return self.value
    
    def __hash__(self):
        return hash(self.value)
    
    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value

class NumberConstant(Constant):
    def __init__(self, value: str):
        self.value = value

class IntegerConstant(Constant):
    def __init__(self, value: int):
        self.value = value

class FloatConstant(Constant):
    def __init__(self, value: float):
        self.value = value

class CharLiteral(Constant):
    def __init__(self, value: str):
        self.value = value

class StringLiteral(Constant):
    def __init__(self, value: str):
        self.value = value

class BoolLiteral(Constant):
    def __init__(self, value: Literal["true", "false"]):
        self.value = value

class NullLiteral(Constant):
    def __init__(self):
        self.value = "null"

class Ellipsis(Constant):
    def __init__(self):
        self.value = "..."

# Struct field
class Field(Constant):
    def __init__(self, value: str):
        self.value = value
    
    def __repr__(self):
        return self.value

# Type name for typecast
class TypeName(Constant):
    def __init__(self, value: str):
        self.value = value
    
    def __repr__(self):
        return self.value

class ExceptionName(TypeName):
    def __init__(self, value: str):
        self.value = value
    
    def __repr__(self):
        return self.value

# For situations where variables need to be tracked.
class Uninitialized(Constant):
    def __init__(self):
        super().__init__()
        self.value = "<uninitialized>"

    def __eq__(self, other):
        return isinstance(other, Uninitialized)
    
    def __hash__(self):
        return hash(self.value)

# A stub placeholder for lambdas.   
class Lambda(Constant):
    def __init__(self, value: int):
        self.value = value # number of arguments
    
    def __repr__(self):
        return f"lambda({self.value} args)"

class ModuleName(Constant):
    def __init__(self, value: str):
        self.value = value
    
    def __repr__(self):
        return self.value

#
# Variables
#
class Variable(ABC):
    def __init__(self, name: str, is_temporary: bool=False):
        self.name = name
        self.is_temporary = is_temporary
    
    def __eq__(self, other: 'Variable'):
        return id(self) == id(other)

    def __repr__(self):
        return self.name
    
    def __hash__(self):
        return id(self)

class Parameter(Variable):
    def __init__(self, name: str):
        super().__init__(name, is_temporary=False)

class GlobalVariable(Variable):
    def __init__(self, name: str):
        super().__init__(name, is_temporary=False)

VarOperand = Union[Constant, Variable]
SSAOperand = Union[Constant, 'SSAOperator', Parameter, GlobalVariable]
FUNCTION_CALL_OP = "function_call"
POINTER_DEREFERENCE_OP = "*_dereference" # To disambiguate from multiplication, we use this name
TERNARY_OP = "ternary"
COPY_OP = "copy"
STORE_OP = "store"
CAST_OP = "cast"
SUBSCRIPT_OP = "[]"
ARRAY_INITIALIZER_OP = "array_init"
DICTIONARY_INITIALIZER_OP = "dict_init"
SET_INITIALIZER_OP = "set_init"
TUPLE_INITIALIZER_OP = "tuple_init"
SIZEOF_OP = "sizeof"
RETURN_OP = "return"
IF_OP = "if"
LOOP_OP = "loop"
BREAK_OP = "break"
CONTINUE_OP = "continue"
PHI_OP = "phi"
MEMBER_ACCESS_OP = "."
MEMBERSHIP_OP = "in"
NOT_OP = "!"
SLICE_OP = "slice"
GENERATOR_OP = "generator"
RAISE_OP = "raise"
YIELD_OP = "yield"
ARGUMENT_UNPACK = "*_unpack_list" # This name disambiguates from other uses of *
KEYWORD_ARGUMENT_UNPACK = "**_unpack_dict"
DEL_OP = "del"
IMPORT_OP = "import"
CATCH_OP = "catch"
WITH_OP = "with"
AND_OP = "&&"
OR_OP = "||"

#
# Operators
#
class Operator(ABC):
    """An operator is the source-level equivalent of an instruction: a single unit of computation that cannot be 
    further broken down at the source level.

    :param op: a symbol identifying what unit of computation this operator object represents.
    """
    def __init__(self, op):
        self.op = op

class VarOperator(Operator):
    """A VarOperator's inputs are variables and constants. Its output is stored in a variable.

    :param op: a symbol identifying what unit of computation this operator object represents.
    :param result: the variable storing the result of the computation. Can be None if the result is not stored anywhere.
    :param operands: the input arguments to this operator.
    :param ast_node: the tree_sitter AST node from which this operator was derived, if any.

    Note that VarOperator is not designed to handle multi-operation expressions like d = a * b + c;. Intermediate results
    should be stored in temporary variables, as in t1 = a * b; d = t1 + c;.
    """
    def __init__(self, op, result: Optional[Variable], operands: List[VarOperand], ast_node: Optional[Node] = None):
        super().__init__(op)
        self.result = result
        self.operands = operands
        self.ast_node = ast_node
    
    def __repr__(self):
        op_names = [repr(op) for op in self.operands]
        if self.result is None:
            return f"{self.op} " + " ".join(op_names)
        return f"{self.result} = {self.op} " + " ".join(op_names)

class FunctionVarOperator(VarOperator):
    """A particular type of VarOperator: a function call. Unlike most operators, different function calls do not necessarily
    represent the same computation. They can be differentiated by name.

    :param name: the name of the function.
    :param result: the variable storing the result of the computation. Can be None if the result is not stored anywhere.
    :param operands: the arguments to the function.
    :param kwargs: the function's keyword arguments, if any.
    :param ast_node: the tree_sitter AST node from which this operator was derived, if any.
    """
    def __init__(self, name: Union[str, Variable], result: Optional[Variable], operands: List[VarOperand], kwargs: Optional[Dict[str, VarOperand]] = None, ast_node: Optional[Node] = None):
        super().__init__(FUNCTION_CALL_OP, result, operands)
        self.name = name
        self.kwargs = kwargs
        self.ast_node = ast_node
    
    def __repr__(self):
        op_names = [repr(op) for op in self.operands]
        name_repr = f"(*{self.name})" if isinstance(self.name, Variable) else self.name
        if self.kwargs is not None:
            op_names += [f"{k}={v}" for k, v in self.kwargs.items()]
        return f"{self.result} = {name_repr}(" + ", ".join(op_names) + ")"

class SSAOperator(Operator):
    """An operator in Single Static Assignment form.

    :param op: a symbol identifying what unit of computation this operator object represents.
    :param operands: the input arguments to this operator.
    :param out_repr: a symbol representing the result of this operator for display purposes.
    :param ast_node: the tree_sitter AST node from which this operator was derived, if any.
    :param var_operator: the VarOperator from which this operator was derived, if any.
    """
    def __init__(self, op, operands: List[SSAOperand], out_repr: Optional[str] = None, ast_node: Optional[Node] = None, var_operator: Optional[VarOperator] = None):
        super().__init__(op)
        self.operands = operands
        self.out_repr = out_repr
        self.ast_node = ast_node
        self.var_operator = var_operator
    
    def __repr__(self):
        op_names = []
        for op in self.operands:
            if isinstance(op, SSAOperator):
                op_names.append(str(id(op)) if op.out_repr is None else op.out_repr)
            else:
                op_names.append(repr(op))
        if self.out_repr is None:
            return f"{self.op} " + " ".join(op_names)
        else:
            return f"{self.out_repr} = {self.op} " + " ".join(op_names)
    
    def __hash__(self):
        return id(self)

class FunctionSSAOperator(SSAOperator):
    """A particular type of SSAOperator: a funciton call. Unlike most operators, different function calls do not necessarily
    represent the same computation. They can be differentiated by name.

    :param name: the name of the function. Can be an SSAOperator if this function call is on a value returned by another operator, or a Parameter or GlobalVariable for a passed in or global function variable.
    :param operands: the arguments to the function.
    :param kwargs: the keyword arguments to this function, if any.
    :param out_repr: a symbol representing the result of this operator for display purposes.
    :param ast_node: the tree_sitter AST node from which this operator was derived, if any.
    :param var_operator: the VarOperator from which this operator was derived, if any.
    """
    def __init__(self, name: Union[str, SSAOperator, Parameter, GlobalVariable], operands: List[SSAOperand], kwargs: Optional[Dict[str, VarOperand]] = None, out_repr: Optional[str] = None, ast_node: Optional[Node] = None, var_operator: Optional[VarOperator] = None):
        super().__init__(FUNCTION_CALL_OP, operands, out_repr)
        self.name = name
        self.kwargs = kwargs
        self.ast_node = ast_node
        self.var_operator = var_operator
    
    def __repr__(self):
        op_names = []
        for op in self.operands:
            if isinstance(op, SSAOperator):
                op_names.append(str(id(op)) if op.out_repr is None else op.out_repr)
            else:
                op_names.append(repr(op))
        if self.kwargs is not None:
            for k, v in self.kwargs.items():
                if isinstance(v, SSAOperator):
                    op_names.append(f"{k}={str(id(v)) if v.out_repr is None else v.out_repr}")
                else:
                    op_names.append(f"{k}={v}")
        out_repr = id(self) if self.out_repr is None else self.out_repr
        name_repr = (id(self.name) if self.name.out_repr is None else self.name.out_repr) if isinstance(self.name, SSAOperator) else self.name
        return f"{out_repr} = {name_repr}(" + ", ".join(op_names) + ")"


OpT = TypeVar('OpT', bound=Operator)

#
# Basic Blocks
#
class BasicBlock:
    id_counter = 0

    def __init__(self, operators: List[OpT], predecessors: List['BasicBlock'], successors: List['BasicBlock']):
        self.operators = operators
        self.predecessors = predecessors
        self.successors = successors
        self.id = BasicBlock.id_counter
        BasicBlock.id_counter += 1

    def add_successor(self, successor: 'BasicBlock'):
        self.successors.append(successor)
        successor.predecessors.append(self)
    
    def __iter__(self) -> Iterator[OpT]:
        """Iterate over the operators in basic block in order.
        """
        for operator in self.operators:
            yield operator

    def __repr__(self):
        predecessors = ", ".join([str(p.id) for p in self.predecessors])
        operators = "\n".join([repr(operator) for operator in self])
        successors = ", ".join([str(s.id) for s in self.successors])

        return f"predecessors: {predecessors}\n ID = {self.id}\n{operators}\nsuccessors: {successors}"
    
    def __eq__(self, other):
        return id(self) == id(other)
    
    def __hash__(self):
        return id(self)

#
# Function
#
class Function:
    def __init__(self, name: str, basic_blocks: List[BasicBlock], parameters: List[Parameter], node: Node):
        """Initialize a Function object.

        Precondition: The first element of basic_blocks is the functions' entry block.
        """
        assert(len(basic_blocks) > 0)
        assert(len(basic_blocks[0].predecessors) == 0)
        self.name = name
        self.entry_block = basic_blocks[0]
        self.basic_blocks = basic_blocks
        self.parameters = parameters
        self.node = node

    def __iter__(self) -> Iterator[BasicBlock]:
        """Iterate over the functions' basic blocks in an arbitrary order except the first block is the entry block.
        """
        for block in self.basic_blocks:
            yield block

    def __repr__(self) -> str:
        declaration = "function " + self.name + "(" + ", ".join([repr(p) for p in self.parameters]) + ")\n"
        block_representations = [repr(b) for b in self.basic_blocks]
        return declaration + "\n\n".join(block_representations)