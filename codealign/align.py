
from typing import Dict, List, Set, Union, Tuple, Callable, Iterator, Optional, Literal, Iterable, Hashable
from enum import Enum
from queue import Queue
from collections import deque
import functools
import itertools
import math
from abc import ABC

from z3 import Int, sat, BoolRef, Or, Sum, Solver

from .lang.c import parse as parse_c
from .lang.python import parse as parse_python
from . import analysis
from . ir import SSAOperator, FunctionSSAOperator, SSAOperand, Constant, GlobalVariable, Parameter, Function, PHI_OP, BasicBlock

CAND_IDX = 1
REF_IDX = 2
EPSILON = 0.001
CONTROL_CONSTRAINT_WEIGHT = 1.0

SOLVER_TIMEOUT = 30

class Alignment:
    def __init__(self, candidate_ir: Function, 
                 reference_ir: Function,
                 alignment_list: List[Tuple[Optional[SSAOperator], Optional[SSAOperator]]],
                 alignment_map: Dict[SSAOperator, Union[Optional[SSAOperator], List[SSAOperator]]]):
        self.candidate_ir = candidate_ir
        self.reference_ir = reference_ir
        self.alignment_list = alignment_list
        self.alignment_map = alignment_map

    def __getitem__(self, item) -> Union[Optional[SSAOperator], List[SSAOperator]]:
        return self.alignment_map[item]

class InjectiveAlignment(Alignment):
    def __init__(self,
                 alignment: List[Tuple[Optional[SSAOperator], Optional[SSAOperator]]],
                 candidate_ir: Function,
                 reference_ir: Function,
                ):
        alignment_map: Dict[SSAOperator, Optional[SSAOperator]] = {}
        for cand_op, ref_op in alignment:
            if cand_op is not None:
                alignment_map[cand_op] = ref_op
            if ref_op is not None:
                alignment_map[ref_op] = cand_op
        super().__init__(candidate_ir, reference_ir, alignment, alignment_map)
    
    def __repr__(self):
        outstr = f"Alignment(candidate={self.candidate_ir.name}, reference={self.reference_ir.name})"
        for cand_op, ref_op in self.alignment_list:
            outstr += "\n"
            if cand_op is not None:
                outstr += f"\n  {cand_op}"
            if ref_op is not None:
                outstr += f"\n  {ref_op}"
        return outstr
    

class RelationalAlignment(Alignment):
    def __init__(self,
                 alignment: List[Tuple[Optional[SSAOperator], Optional[SSAOperator]]],
                 candidate_ir: Function,
                 reference_ir: Function
                ):
        alignment_map: Dict[SSAOperator, List[SSAOperator]] = {}

        for parameter in itertools.chain(candidate_ir.parameters, reference_ir.parameters):
            alignment_map[parameter] = []
            
        for op in (op for fn in (candidate_ir, reference_ir) for bb in fn for op in bb):
            alignment_map[op] = []
        
        for cand_op, ref_op in alignment:
            if cand_op is not None and ref_op is not None:
                # accumulate(alignment_map, cand_op, ref_op)
                # accumulate(alignment_map, ref_op, cand_op)
                alignment_map[cand_op].append(ref_op)
                alignment_map[ref_op].append(cand_op)
        super().__init__(candidate_ir, reference_ir, alignment, alignment_map)

    def __repr__(self):
        def _sort_key_for_display(op: SSAOperand) -> str:
            """A sort key function for sorting items in equivalence classes for a nice display.
            """
            if isinstance(op, SSAOperator):
                if op.out_repr is None:
                    if len(op.operands) > 0:
                        return " " + str(op.operands[0])
                    else:
                        return " " + str(op)
                else:
                    return op.out_repr
            else:
                return (str(op))
            
        equivalence_classes = UnionFind(operator for fn in (self.candidate_ir, self.reference_ir) for block in fn.basic_blocks for operator in block)
        for left, right in self.alignment_list:
            if isinstance(left, SSAOperator) and isinstance(right, SSAOperator):
                equivalence_classes.union(left, right)

        outstr = f"Alignment(candidate={self.candidate_ir.name}, reference={self.reference_ir.name})"
        for eq_class in equivalence_classes.export_sets():
            outstr += "\n\n  " + "\n  ".join(map(str, sorted(list(eq_class), key=_sort_key_for_display)))
        return outstr

#####################################
#                                   #
#    Main interface to codealign    #
#                                   #
#####################################   
def align(candidate: str, 
          reference: str,
          language: Literal["c", "python"], 
          candidate_function: Optional[str] = None, 
          reference_function: Optional[str] = None,
          injective: bool = False,
          control_dependence: bool = True,
          partial_loops: bool = True,
          alpha: float = 1.0,
          verbose: bool = False
         ) -> Alignment:
    """Main interface to codealign. Aligns two functions.

    :param candidate: code containing one of the functions to align.
    :param reference: code containing the other function to align.
    :param language: the language that the functions are written in. Options include "c" or "python".
    :param candidate_function: if specified, the name of the function to align from the candidate code. If None, align expects "candidate" to contain exactly one function.
    :param reference_function: if specified, the name of the function to align from the reference code. If None, align expects "reference" to contain exactly one function.
    :param injective: if True, force the alignment mapping to be an injective partial function.
    :param control_dependence: if True, use control dependence constraints in determining an alignment.
    :param partial_loops: if True, allows for parts of loops to be proven equivalent rather than all-or-nothing for the whole loop.
    :param alpha: the percentage of lemma arguments that must be proven true to prove the lemma true.
    :param verbose: print information about the proof process.

    :returns: an Alignment object encapsulating the codealign IR representations of the functions as well as the alignment mapping itself. Will be a RelationalAlignment object if injective=False or an InjectiveAlignment if injective=True.
    """
    language = language.lower()
    if language == "c":
        parse_fn: Callable[[bytes], List[Function]] = parse_c
    elif language == "python":
        parse_fn: Callable[[bytes], List[Function]] = parse_python
    else:
        raise ValueError(f"Codealign cannot parse language {language}!")
    
    # At least for now
    assert not injective or not control_dependence, f"Control dependence with injective alignment is currently unsupported."
    assert partial_loops or alpha == 1.0, f"Disabling partial loops with alpha < 1.0 is currently unsupported."
    
    def prepare_ir(code: str, function_name: Optional[str], which: Literal['candidate', 'reference']):
        function_list = parse_fn(bytes(code, "utf8"))

        if function_name is None:
            if len(function_list) > 1:
                raise ValueError(f"Multiple {which} functions supplied but none are specified.")
            elif len(function_list) == 0:
                raise ValueError(f"No {which} functions supplied.")
            ir = function_list[0]
        else:
            for fn in function_list:
                if fn.name == function_name:
                    ir = fn
                    break
            else: # attached to the for loop
                raise ValueError(f"Function {function_name} is not among the {which} functions.")
        
        ir = analysis.convert_to_ssa(ir)
        analysis.copy_propagation(ir)
        return ir

    candidate_ir = prepare_ir(candidate, candidate_function, "candidate")
    reference_ir = prepare_ir(reference, reference_function, "reference")

    if verbose:
        print("-------- Candidate IR --------")
        print(candidate_ir)
        print("\n\n-------- Reference IR --------")
        print(reference_ir)
        print("\n\n")

    return _align_inductive(candidate_ir, reference_ir, injective=injective, control_dependence=control_dependence, partial_loops=partial_loops, alpha=alpha, verbose=verbose)

###
# Utility methods for extracting information from codealign IR.
### 
class FunctionPointer:
    # TODO: We know if certain function pointers are equivalent if they are from parameters or global
    # variables. Update this class to reflect this.
    def __init__(self):
        self.description = "<function pointer>"

    def __eq__(self, other):
        return isinstance(other, FunctionPointer)
    
    def __hash__(self):
        return hash(self.description)
    
    def __repr__(self):
        return self.description

def classify_operators_by_operantion(f: Function):
    classes: Dict[(str, Optional[Union[str, FunctionPointer]]), List[SSAOperator]] = {}
    for operator in (operator for bb in f.basic_blocks for operator in bb):
        if isinstance(operator, FunctionSSAOperator):
            if isinstance(operator.name, str):
                id_tuple = (operator.op, operator.name)
            else:
                assert isinstance(operator.name, SSAOperator) or isinstance(operator.name, Parameter) or isinstance(operator.name, GlobalVariable)
                id_tuple = (operator.op, FunctionPointer())
        else:
            assert isinstance(operator, SSAOperator)
            id_tuple = (operator.op, None)
        
        if id_tuple not in classes:
            classes[id_tuple] = [operator]
        else:
            classes[id_tuple].append(operator)
    return classes

class MissingArgument(Constant):
    """Sometimes, operators may have different numbers of arguments. The equivalent operator code above assumes that equivalent
    operations have the same number of arguments. Additionally, some classes may have keyword arguments. Instances of this class
    represent an argument that one operator has but the other does not.

    The __hash__ and __eq__ methods of this class are set up so that this class simulates being a singleton.
    """
    def __init__(self):
        super().__init__()
        self.value = "<missing argument>"
    
    def __hash__(self):
        return hash(self.value)
    
    def __eq__(self, other):
        return isinstance(other, MissingArgument)

def standardize_operands(candidate_operator: SSAOperator, reference_operator: SSAOperator):
    """Produce argument lists that are identical in length, incorporating keyword arguments. To produce identical length lists, lists
    are padded with MissingArgument where necessary.
    """
    # Make copies because we do not want to modify the original operand lists themselves. We need them in their original form
    # for comparisons against other operators. A shallow copy is fine; we won't mutate any of the lists' contents.
    candidate_operands = candidate_operator.operands.copy()
    reference_operands = reference_operator.operands.copy()

    # First, handle extra positional arguments:
    if len(candidate_operands) > len(reference_operands):
        reference_operands += [MissingArgument()] * (len(candidate_operands) - len(reference_operands))
    elif len(reference_operands) > len(candidate_operands):
        candidate_operands += [MissingArgument()] * (len(reference_operands) -  len(candidate_operands))

    has_keyword_arguments = lambda operator: isinstance(operator, FunctionSSAOperator) and operator.kwargs is not None

    # Handle matching of keyword arguments. We'll put keyword arguments at the same index in the output lists if they 
    # have the same name, and at different places if they have different names.
    if has_keyword_arguments(candidate_operator) and has_keyword_arguments(reference_operator):
        common_kwargs = set(candidate_operator.kwargs).intersection(set(reference_operator.kwargs))
        for argname in common_kwargs:
            candidate_operands.append(candidate_operator.kwargs[argname])
            reference_operands.append(reference_operator.kwargs[argname])
        for argkey, argvalue in candidate_operator.kwargs.items():
            if argkey not in common_kwargs:
                candidate_operands.append(argvalue)
                reference_operands.append(MissingArgument())
        for argkey, argvalue in reference_operator.kwargs.items():
            if argkey not in common_kwargs:
                reference_operands.append(argvalue)
                candidate_operands.append(MissingArgument())
    elif has_keyword_arguments(candidate_operator):
        for argkey, argvalue in candidate_operator.kwargs.items():
            candidate_operands.append(argvalue)
            reference_operands.append(MissingArgument())
    elif has_keyword_arguments(reference_operator):
        for argkey, argvalue in reference_operator.kwargs.items():
            reference_operands.append(argvalue)
            candidate_operands.append(MissingArgument())

    assert len(candidate_operands) == len(reference_operands)
    
    return (candidate_operands, reference_operands)

class OperationEquivalence(Enum):
    EQUIVALENT = 1
    POSSIBLY_EQUIVALENT = 2
    NONEQUIVALENT = 3

def determine_equivalence_level(op1: SSAOperator, op2: SSAOperator) -> OperationEquivalence:
    if isinstance(op1, FunctionSSAOperator) and isinstance(op2, FunctionSSAOperator):
        if isinstance(op1.name, str) and isinstance(op2.name, str):
            # When both names are strings, we know exactly if they are equivalent or not.
            if op1.name == op2.name:
                return OperationEquivalence.EQUIVALENT
            else:
                return OperationEquivalence.NONEQUIVALENT
        if type(op1.name) == type(op2.name):
            return OperationEquivalence.POSSIBLY_EQUIVALENT # Both function pointers; may or may not align.
        else: # Mixed types: string name with variable, variable with SSA operator, parameter with global variable, etc.
            return OperationEquivalence.NONEQUIVALENT
    if op1.op == op2.op:
        return OperationEquivalence.EQUIVALENT
    else:
        return OperationEquivalence.NONEQUIVALENT

@functools.cache
def loopbreaking_phi_nodes(f: Function, control_dependence: bool = False) -> dict[SSAOperator, set[int]]:
    """Identify the arguments the phi operations of function f which have dependency 
    cycles that point back to the same phi operation. Always includes dataflow dependencies,
    control dependencies can be specified with an option.
    """

    operator_control_dependencies = operator_level_control_dependence(f) if control_dependence else None

    def find_loops(operator: SSAOperator, head: SSAOperator, encountered: set[SSAOperator]):
        if operator == head:
            return True
        if operator in encountered:
            return False
        encountered.add(operator)

        result = False
        for operand in operator.operands:
            if isinstance(operand, SSAOperator):
                result = find_loops(operand, head, encountered) or result
        if control_dependence:
            for dependent in operator_control_dependencies[operator]:
                result = find_loops(dependent[0], head, encountered) or result
        return result

    def loop_phi(operator: SSAOperator) -> set[int]:
        return {
            i for i, operand in enumerate(operator.operands) 
            if isinstance(operand, SSAOperator) and find_loops(operand, operator, set())
        }

    loopbreaking_phis: Dict[SSAOperator, Set[int]] = {}
    for loop in analysis.find_loops(f):
        for operator in (x for x in loop.head.operators if x.op == PHI_OP):
            if breaking_indices := loop_phi(operator): # will be an empty set if there aren't any, which is treated as False.
                loopbreaking_phis[operator] = breaking_indices

    return loopbreaking_phis    

# SSA Form already contains backwards pointer, linking uses to definitions. This method creates the 
# opposite: pointers linking definitions to uses.
@functools.cache
def initialize_uses(f: Function) -> Dict[SSAOperand, List[SSAOperator]]:
    def2use: Dict[SSAOperand, List[SSAOperator]] = {}
    for operator in (operator for bb in f.basic_blocks for operator in bb):
        for operand in operator.operands:
            if operand not in def2use:
                def2use[operand] = []
            def2use[operand].append(operator)
    return def2use

def find_function_pointer_definitions(f: Function) -> Dict[SSAOperand, List[SSAOperator]]:
    def2fnptr: Dict[SSAOperand, List[SSAOperator]] = {}
    for operator in (operator for bb in f.basic_blocks for operator in bb):
        if isinstance(operator, FunctionSSAOperator) and isinstance(operator.name, SSAOperator):
            if operator not in def2fnptr:
                def2fnptr[operator] = [operator.name]
            else:
                def2fnptr[operator].append(operator.name)
    return def2fnptr

@functools.lru_cache(1)
def parameter_locations(candidate_ir: Function, reference_ir: Function):
    # Variables hash and compare by ID; therefore we can store candidate and reference parameters in the same dictionary without fear of conflict.
    parameter_locations: Dict[Parameter, int] = {
        var: i for i, var in enumerate(candidate_ir.parameters)
    }
    parameter_locations.update({
        var: i for i, var in enumerate(reference_ir.parameters)
    })
    return parameter_locations


class BackslicePlaceholder(Constant):
    """A filler value used to ensure that slice frontier deques in backslice clone construction are always parallel.

    It is a subtype of Constant since the frontier instance variables of Slice objects are of type deque[SSAOperand].
    """
    def __init__(self):
        self.value = "<backslice placeholder>"

###
# Clone detection
###
class BacksliceClone:
    """Clone created by slicing backward through a dataflow graph represented in SSA.
    """
    def __init__(self, candidate_ir: Function, candidate_seed: SSAOperator, reference_ir: Function, reference_seed: SSAOperator):
        self.candidate_slice = Slice(candidate_ir, candidate_seed)
        self.reference_slice = Slice(reference_ir, reference_seed)
        self.candidate_ir = candidate_ir
        self.reference_ir = reference_ir
        self.op_pairs: List[Equivalence] = []
        self.parameter_locations = parameter_locations(candidate_ir, reference_ir)

        while len(self.candidate_slice.frontier) > 0 and len(self.reference_slice.frontier) > 0:
            # print("---------- New Round --------")
            # print(self.candidate_slice.frontier)
            # print(self.reference_slice.frontier)
            # print()

            cand_operator = self.candidate_slice.explore()
            ref_operator = self.reference_slice.explore()

            if isinstance(cand_operator, SSAOperator) and isinstance(ref_operator, SSAOperator) and \
               BacksliceClone.same_operations(cand_operator, ref_operator, self.parameter_locations):
                self.op_pairs.append(Equivalence(cand_operator, ref_operator))
                candidate_operands, reference_operands = standardize_operands(cand_operator, ref_operator)
                self.candidate_slice.expand(candidate_operands)
                self.reference_slice.expand(reference_operands)
            else:
                self.candidate_slice.skip()
                self.reference_slice.skip()
        
        # Fixing a failure of this assertion will require modifying the condition to add an element
        # to the exploration frontier in Slice.expand().
        assert len({p.left for p in self.op_pairs}) == len(self.op_pairs), "Error: The same operator appears multiple times in the same clone!"
        self.op_pairs.reverse() # The list is build backwards (via backwards slicing) so we must reverse it to be in forward order.
    
    @staticmethod
    def same_operations(cand: SSAOperator, ref: SSAOperator, parameter_locations: Dict[Parameter, int]) -> bool:
        """A wrapper around determine_equivalence_level that always returns true or false by recursively
        exploring the equivalence level of the source of "maybe equivalent" function pointers.
        """
        eq_level = determine_equivalence_level(cand, ref)
        if eq_level == OperationEquivalence.EQUIVALENT:
            return True
        elif eq_level == OperationEquivalence.NONEQUIVALENT:
            return False
        else:
            assert isinstance(cand, FunctionSSAOperator) and isinstance(ref, FunctionSSAOperator)
            if isinstance(cand.name, Parameter) and isinstance(ref.name, Parameter):
                return parameter_locations[cand.name] == parameter_locations[ref.name]
            if isinstance(cand.name, GlobalVariable) and isinstance(ref.name, GlobalVariable):
                return cand.name.name == ref.name.name
            
            # We shouldn't have to worry about loops of function pointers here: all function pointers
            # have a non-function-pointer source, and loops should all involve at least one phi node anyway.
            if isinstance(cand.name, SSAOperator) and isinstance(ref.name, SSAOperator):
                return BacksliceClone.same_operations(cand.name, ref.name, parameter_locations)
            # Some other combination of Parameters/GlobalVariables/SSAOperators for the FunctionSSAOperators' names.
            return False
    
    def __len__(self):
        return len(self.op_pairs)
        
    def __repr__(self):
        return "BacksliceClone\n  " + "\n  ".join([repr(op_pair) for op_pair in self.op_pairs])

class Slice:
    def __init__(self, ir: Function, seed_operator: SSAOperator):
        self.loopbreaking_phis: Dict[SSAOperator, Set[int]] = loopbreaking_phi_nodes(ir)
        self.operators = set()
        self.frontier: deque[SSAOperand] = deque()
        self.frontier.append(seed_operator)
    
    def explore(self) -> SSAOperand:
        return self.frontier[0]

    def expand(self, standardized_operands: List[SSAOperand]):
        new_addition = self.frontier.popleft()
        assert isinstance(new_addition, SSAOperator), f"A backslice can only contain operators; attempted to add {new_addition}."
        self.operators.add(new_addition)
        for i, operand in enumerate(standardized_operands):
            # Potential efficiency improvement: have a set tracking the contents of self.frontier and check for membership there.
            # It is fine to have multiple instances of the same constant in the frontier, but not the same SSAOperator.
            # The same operator can already be in the frontier or slice due to phi nodes.
            if (not isinstance(operand, SSAOperator) or operand not in self.frontier) and operand not in self.operators \
               and (new_addition not in self.loopbreaking_phis or i not in self.loopbreaking_phis[new_addition]):
                    self.frontier.append(operand)
            else:
                self.frontier.append(BackslicePlaceholder())
        
        # Handle function pointers
        if isinstance(new_addition, FunctionSSAOperator) and isinstance(new_addition.name, SSAOperator) and new_addition.name not in self.frontier and new_addition.name not in self.operators:
            self.frontier.append(new_addition.name)
        else:
            self.frontier.append(BackslicePlaceholder())
    
    def skip(self):
        self.frontier.popleft() # Remove from the frontier, and don't do anything to it.

def find_premises_for_clone(clone: 'CloneMerger.MClone') -> Set['Equivalence']:
    """Find the set of operators that must be shown equivalent for the clones to be considered equivalent in context.

    In essence, this computes a cut set separating the rest of the clone from the operators earlier in the function.
    """
    equivalences: Dict[SSAOperator, Equivalence] = {}
    for equivalence in clone:
        # Clones are by definition isomorphic so it doesn't matter if we explore the clone via the left or right.
        equivalences[equivalence.left] = equivalences
    
    premises = set()
    for equivalence in clone:
        if len(equivalence.left.operands) == 0:
            premises.add(equivalence)
        else:
            all_non_operator_operands = True
            for operand in equivalence.left.operands:
                if isinstance(operand, SSAOperator):
                    if operand not in equivalences:
                        premises.add(equivalence)
                        break
                    else:
                        all_non_operator_operands = False
            
            if all_non_operator_operands:
                premises.add(equivalence)
    
    return premises

class IDClone(ABC):
    def __init__(self, id_num: int):
        self.id = id_num

    def __iter__(self) -> Iterator['Equivalence']:
        raise NotImplementedError("IDClone has no iterator method.")
    
    def __hash__(self):
        return self.id

class CloneMerger:
    """A data structure which helps subsume and merge clones.

    A clone that is subsumed is completely contained within another clone.
    A clone that is merged share some operator mappings but not others.
    """
    class Node:
        def __init__(self, equivalence: 'Equivalence', clone: 'CloneMerger.MClone'):
            self.equivalence = equivalence
            self.clone = clone
            clone.add_node(self)
        
        def __repr__(self):
            return f"Node({self.equivalence}, clone id={self.clone.id})"
    
    class MClone(IDClone):
        def __init__(self, id_num: int):
            self.id = id_num
            self.mappings: List[CloneMerger.Node] = []
        
        def __eq__(self, other):
            return isinstance(other, CloneMerger.MClone) and self.id == other.id
        
        def __hash__(self):
            return self.id
        
        def __iter__(self) -> Iterator['Equivalence']:
            return (m.equivalence for m in self.mappings)

        def add_node(self, node: 'CloneMerger.Node'):
            self.mappings.append(node)
        
        def __repr__(self):
            return f"MergedClone(id={self.id}, members=\n    " + "\n    ".join(repr(m.equivalence) for m in self.mappings) + ")"


    def __init__(self):
        self.id_src = 0
        self.index: Dict[Equivalence, CloneMerger.Node] = {}
        self.clones: List[Optional[CloneMerger.MClone]] = []
    
    def add_clone(self, clone: BacksliceClone):
        assert len(clone) > 0, f"Recieved empty clone: {clone}"
        self._add_clone(iter(clone.op_pairs), None)

    def __iter__(self) -> Iterator[MClone]:
        return iter(clone for clone in self.clones if clone is not None)
    
    def _add_clone(self, clone_iter: Iterator['Equivalence'], as_clone: Optional[MClone]) -> MClone:
        try:
            equivalence = next(clone_iter)
            if equivalence in self.index:
                mclone = self.index[equivalence].clone # mclone: the merged clone associated with this value.
                as_clone = self._add_clone(clone_iter, mclone)
                if not (as_clone is None or mclone == as_clone):
                    # The arguments here cannot be swapped. (Or, if they are, _merge_clones must return
                    # the merged clone and that must be assigned to as_clone.)
                    self._merge_clones(as_clone, mclone)
            else:
                as_clone = self._add_clone(clone_iter, as_clone)
                as_clone = self._add_node(equivalence, as_clone).clone
        except StopIteration:
            pass
        return as_clone

    def _add_node(self, eq: 'Equivalence', as_clone: Optional[MClone] = None) -> Node:
        assert eq not in self.index, f"Cannot add node for equivalence {eq} to the CloneMerger because it already exists."
        assert isinstance(eq, Equivalence)
        clone = self._make_new_clone() if as_clone is None else as_clone
        new_node = CloneMerger.Node(eq, clone)
        self.index[eq] = new_node
        return new_node
    
    def _make_new_clone(self) -> MClone:
        clone = CloneMerger.MClone(self.id_src)
        self.id_src += 1
        self.clones.append(clone)
        return clone
    
    def _merge_clones(self, left: MClone, right: MClone):
        for equivalence in right:
            node = self.index[equivalence]
            node.clone = left
            left.add_node(node)
        
        self.clones[right.id] = None
        
    
    def __repr__(self):
        return "CloneMerger\nNodes:\n  " + "\n  ".join([repr(n) for _, n in self.index.items()]) + \
        "\nClones:\n  " + "\n  ".join(repr(c) for c in self.clones)

class CompositeClone(IDClone):
    def __init__(self, id_num: int, equivalences: Iterable['Equivalence'], ambiguities: List['Ambiguity'], ambiguity_frontier: Iterable['Equivalence']):
        super().__init__(id_num)
        self.equivalences = equivalences
        self.ambiguities = ambiguities
        self.ambiguity_frontier = ambiguity_frontier

    def __iter__(self) -> Iterator['Equivalence']:
        yield from self.equivalences

    def __repr__(self):
        return f"CompositeClone(id={self.id}, members= {', '.join([repr(eq) for eq in self.equivalences])})\n-- Ambiguities:\n" + \
        '\n'.join([repr(a) for a in self.ambiguities]) + \
        '\nAmbiguityFrontier: ' + ", ".join(repr(eq) for eq in self.ambiguity_frontier)


    @classmethod
    def create_composite_clone(cls, id_num: int, unambiguous_equivalences: Iterable['Equivalence'], ambiguities: List['Ambiguity']):
        # print("---------- Creating Composite Clone ----------")
        # print(f"Number of ambiguities: {len(ambiguities)}")

        equivalences = set(unambiguous_equivalences)
        ambiguity_frontier: Set[Equivalence] = set()
        # InternalClones include at least one nonambiguous differentiator that occurs after the clone. We must filter that out (hence the if clause in the set comprehension directly below.)
        ambiguous_equivalences: Set[Equivalence] = {eq for ambiguity in ambiguities for clone in ambiguity.clones for eq in clone if eq not in equivalences}

        def find_ambiguity_frontier(ambiguity: Ambiguity):
            # InternalClones include at least one nonambiguous differentiator that occurs after the clone. We must filter that out (hence the if clause in the set comprehension directly below.)
            this_ambiguity_equivalences = {eq for clone in ambiguity.clones for eq in clone if eq not in equivalences}
            for equivalence in this_ambiguity_equivalences:
                for cand_operand, ref_operand in zip(*standardize_operands(equivalence.left, equivalence.right)):
                    if isinstance(cand_operand, SSAOperator) and isinstance(ref_operand, SSAOperator):
                        eq = Equivalence(cand_operand, ref_operand)
                        if eq in equivalences:
                            ambiguity_frontier.add(eq)
                        # assert eq in this_ambiguity_equivalences or eq not in ambiguous_equivalences, "Violation of assumption: distinct ambiguity instances nonetheless have a direct data dependency between them."
                        if not (eq in this_ambiguity_equivalences or eq not in ambiguous_equivalences):
                            print("---- Assertion Error ----")
                            print(eq)
                            print("\n The ambiguity in question")
                            print(ambiguity)
                            raise ValueError("Violation of assumption: distinct ambiguity instances nonetheless have a direct data dependency between them.")

        for ambiguity in ambiguities:
            find_ambiguity_frontier(ambiguity)

        return cls(id_num, equivalences, ambiguities, ambiguity_frontier)

class ProxyCloneOperator:
    def __init__(self, id_num: int, equivalences: List['Equivalence']):
        self.id = id_num
        self.op = f"clone{id_num}"
        self.equivalences = equivalences

    def __iter__(self):
        yield from self.equivalences

    def __repr__(self):
        return f"ProxyCloneOperator({self.id}):\n    " + "\n    ".join([repr(eq) for eq in self.equivalences])
    
    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        return isinstance(other, ProxyCloneOperator) and self.id == other.id

    @classmethod
    def from_idclone(cls, clone: IDClone):
        if isinstance(clone, CompositeClone):
            return ProxyCompositeCloneOperator.from_composite_idclone(clone)
        if isinstance(clone, InternalClone):
            equivalence_iterable = clone.original_equivalences
        else:
            equivalence_iterable = clone
        return cls(
            clone.id,
            [equivalence for equivalence in equivalence_iterable]
        )

class ProxyCompositeCloneOperator(ProxyCloneOperator):
    def __init__(self, id_num: int, equivalences: List['Equivalence'], ambiguities: List['Ambiguity'], ambiguity_frontier: Set['Equivalence']):
        super().__init__(id_num, equivalences)
        self.ambiguities = ambiguities
        self.ambiguity_frontier = ambiguity_frontier

    def __repr__(self):
        return f"ProxyCompositeCloneOperator({self.id}):\n    " + "\n    ".join([repr(eq) for eq in self.equivalences]) + \
        "\n  AmbiguityFrontier:\n    " + "\n    ".join([repr(eq) for eq in self.ambiguity_frontier])

    @classmethod
    def from_composite_idclone(cls, clone: CompositeClone):
        return cls(
            clone.id,
            [equivalence for equivalence in clone],
            clone.ambiguities,
            clone.ambiguity_frontier
        )

class Ambiguity:
    def __init__(self, clones: Set[CloneMerger.MClone]):
        self.clones: List[ProxyCloneOperator] = [ProxyCloneOperator.from_idclone(c) for c in clones]
    
    def __repr__(self):
        return "Ambiguity\n  " + "\n  ".join(
            [repr(c) for c in self.clones]
        )
    
class IDSrc:
    def __init__(self, start: int):
        self.next_id = start

    def make_id(self):
        id = self.next_id
        self.next_id += 1
        return id
    
class InternalClone(IDClone):
    def __init__(self, id_num: int, equivalences: List['Equivalence']):
        super().__init__(id_num)
        self.original_equivalences = equivalences.copy() # Equivalences will be extended later to provide a differentiating factor.
        self.equivalences = equivalences

    def __hash__(self) -> int:
        return super().__hash__()
    
    def __eq__(self, other) -> bool:
        return isinstance(other, InternalClone) and self.id == other.id
    
    def __iter__(self) -> Iterator['Equivalence']:
        return iter(self.equivalences)
    
    def __repr__(self) -> str:
        return f"InternalClone(id={self.id}, members=" + ", ".join(repr(e) for e in self.equivalences) + ")"

def is_contradictory_clone(clone: CloneMerger.MClone):
    # basic assumption: the same equivalence should not be present multiple times in the same clone.
    candidate_operators = set()
    reference_operators = set()

    for equivalence in clone:
        if equivalence.left in candidate_operators:
            return True
        candidate_operators.add(equivalence.left)
        if equivalence.right in reference_operators:
            return True
        reference_operators.add(equivalence.right)
    
    return False

class DisambiguationCluster:
    def __init__(self, clones: Optional[Set[CloneMerger.MClone]] = None, reprs: Optional[Dict[CloneMerger.MClone, str]] = None):
        self.clones = set() if clones is None else clones
        self.chosen_members = 0
        self.hashed = False
        self.reprs = reprs
    
    def add_clone(self, clone):
        assert not self.hashed, "Modifying a cluster after hashing may be problematic!"
        assert self.chosen_members == 0, "Add clones only during initialization, not after disambiguation has begun"
        self.clones.add(clone)
    
    def __len__(self):
        return len(self.clones)
    
    def __contains__(self, item):
        return item in self.clones
    
    def __hash__(self):
        self.hashed = True
        # A commutitive operation (like xor) is important here because it is important that the order cannot matter.
        # This is also the approach taken by python's native frozenset.
        return functools.reduce(lambda x, y: x ^ y, map(hash, self.clones))
    
    def __eq__(self, other):
        return isinstance(other, DisambiguationCluster) and other.clones == self.clones
    
    def __repr__(self):
        return " + ".join(f"x{c.id}" for c in self.clones) + f", chosen_members={self.chosen_members}"
    
class UnionFind:
    def __init__(self, items: Iterable[Hashable]):
        self.index: Dict[Hashable, Hashable] = {item: item for item in items}
        self.rank: Dict[Hashable, int] = {item: 0 for item in self.index}
    
    def find(self, item: Hashable) -> int:
        if self.index[item] != item:
            self.index[item] = self.find(self.index[item])
        return self.index[item]
    
    def union(self, a: Hashable, b: Hashable):
        x = self.find(a)
        y = self.find(b)
        if x == y:
            return
        
        if self.rank[x] > self.rank[y]:
            self.index[y] = x
        elif self.rank[x] < self.rank[y]:
            self.index[x] = y
        else:
            self.index[x] = y
            self.rank[y] += 1

    def export_sets(self) -> List[Set[Hashable]]:
        sets: Dict[Hashable, Set[Hashable]] = {}
        for item in self.index:
            representative = self.find(item)
            if representative in sets:
                sets[representative].add(item)
            else:
                sets[representative] = {item}
        return [s for _, s in sets.items()]
    
    
def select_clones(clones: Iterable[IDClone], allow_resolvable_contradictions: bool = False) -> Optional[tuple[List[IDClone], List[Ambiguity]]]:
    """Choose as many non-conflicing clones as possible; prefer larger clones over smaller ones.

    Sometimes, clones cannot be differentiated. In these cases, return an Ambiguity object.
    Other times, the clones are fudamentally inconsitent. In these cases, we return None.

    :param clones: The clones from which to select.
    :param allow_resolvable_contradictions: Some contradictions can be isolated to a certain part of the
    system of equations and ignored. If this flag is set, ignore those when possible.
    """
    operator2cluster: Dict[SSAOperator, DisambiguationCluster] = {}
    for clone in clones:
        for equivalence in clone:
            left = equivalence.left
            right = equivalence.right
            if left in operator2cluster:
                operator2cluster[left].add_clone(clone)
            else:
                operator2cluster[left] = DisambiguationCluster({clone})
            if right in operator2cluster:
                operator2cluster[right].add_clone(clone)
            else:
                operator2cluster[right] = DisambiguationCluster({clone})

    # Must deduplicate (using a set comprehension here) because some clones may share several of the same
    # operators and thus the same constraint will be completed twice.
    clusters = list({c for _, c in operator2cluster.items()})

    # print("--- Constraints ---")
    # for cluster in clusters:
    #     print(cluster)

    # print("---- Indexed Constraints ----")
    # for op, cluster in operator2cluster.items():
    #     print(f"{op}: {cluster}")
    
    # Find anchor clones: clones that uniquely involve some operators that are not used by any other clone
    anchor_clones = list(map(
        lambda x: next(iter(x.clones)),
        filter(lambda x: len(x) == 1, clusters)
    ))

    clone_set = set(anchor_clones)

    # Increment cluster counts for each known anchor clone.
    for cluster in clusters:
        for element in clone_set:
            if element in cluster:
                cluster.chosen_members += 1


    # 0 chosen members: ambiguous (either could work)
    # 1 chosen member: one clone subsumes the others
    # 2+ chosen members: conflicting (report both; hopefully can differentiate based on context.)

    if not allow_resolvable_contradictions and any(map(lambda x: x.chosen_members > 1, clusters)):
        return None

    # We can use information learned in one cluster to disambiguate others. This is similar to 
    # solving a triangular system of linear equations.
    # In this analogy, clusters are variables and disambiguation clusters are expressions set equal
    # to 1.

    ### Setup for solving triangular equations

    # At this point, anchor_clones contains only clones from equations with one element.
    var_value_deque: deque[tuple[CloneMerger.MClone, int]] = deque()
    for clone in anchor_clones:
        var_value_deque.append((clone, 1))
    
    var2value: Dict[CloneMerger.MClone, int] = {v: 1 for v in anchor_clones}
    clone2cluster: Dict[CloneMerger.MClone, List[DisambiguationCluster]] = {}
    # Because the deque used to control the solving loop contains the variable from each one-variable equation (e.g. x0 = 1),
    # we need at least a dummy empty array in case there are no other equations containing that variable (for instance, a 
    # single clone that doesn't contradict with any others).
    for clone in anchor_clones:
        clone2cluster[clone] = []
    for cluster in filter(lambda x: x.chosen_members <= 1 and len(x) > 1, clusters):
        for clone in cluster.clones:
            if clone in clone2cluster:
                clone2cluster[clone].append(cluster)
            else:
                clone2cluster[clone] = [cluster]

    # Solving involves lots of hashing and comparing CloneMerger.MClones. This is fine from an
    # efficiency standpoint because hashing and comparisons are done based on unique integer IDs.

    ### Solve the triangular equations.
    solved_equations: Set[DisambiguationCluster] = set()
    while len(var_value_deque) > 0:
        variable, value = var_value_deque.popleft()
        for equation in clone2cluster[variable]:
            if equation in solved_equations:
                continue
            # CNF for 'value = 1 => equation.chosen_members = 1
            assert value != 1 or equation.chosen_members == 1
            
            if value == 1:
                # All other values in the equation must be zero
                for eq_var in equation.clones:
                    if eq_var != variable:
                        assert eq_var not in var2value or var2value[eq_var] == 0, f"Conflicting values for variable x{eq_var.id}"
                        var2value[eq_var] = 0 # This might override the existing 0 with 0, which is fine.
                        var_value_deque.append((eq_var, 0))
                solved_equations.add(equation)
            else:
                assert value == 0, f"Invalid value {value} for variable x{variable.id}"
                num_unknown = 0
                # If all other variables in the equation are zero, the last remaining one must be one.
                for eq_var in equation.clones:
                    if eq_var in var2value:
                        assert var2value[eq_var] == 0, "All equations with a known '1' should be in solved_equations"
                    else:
                        num_unknown += 1
                        unknown = eq_var
                
                # We can only conclude that the remaining variable is 1 if all others are 0.
                if num_unknown == 1:
                    var2value[unknown] = 1
                    var_value_deque.appendleft((unknown, 1)) # Process variable=1 values first.
                    solved_equations.add(equation)
                    for unk_eq in clone2cluster[unknown]: # This will include the current value of the variable 'equation'.
                        unk_eq.chosen_members += 1
                    # If we continue to iterate over equations here, we may encounter another instance of the "unknown" variable from above
                    # in another equation. (The unknown variable is necessarily different from the current variable "variable" popped 
                    # from the var_value_deque because the current variable always is associated with a value). In this circumstance, the
                    # "All equations with a known '1' should be in solved_equations" assertion above would be triggered. Using deque.appendleft
                    # forces the algorithm to iterate over all equations associated with 'unknown' anyway.
                    break

    # print("---- After Solving: Constraints ----")
    # for cluster in clusters:
    #     print(cluster)
    # print()
    # print("Varable values")
    # for v in sorted(var2value.keys(), key=repr):
    #     print(f"x{v.id}: {var2value[v]}")
    # print()
    # print()
    
    # Sanity check
    for equation in solved_equations:
        assert sum(var2value[variable] for variable in equation.clones) == 1 and equation.chosen_members == 1

    # It may the the case that some subset of the equations are not triangular and thus do not have
    # a unique solution. In this case, we return each such subset as an Ambiguity object.
    # There can be multiple unique sets of equations which are internally ambiguous. We distinguish
    # between sets of equations which have no common variables.

    ambiguous_clusters: List[List[IDClone]] = []
    for cluster in filter(lambda x: x.chosen_members == 0, clusters):
        # Include only the clones for which we don't know if they should be selected or not.
        ambiguous_clusters.append([c for c in cluster.clones if c not in var2value])

    # This means that there is a contradiction somewhere in this set of equations. What this condition
    # means is that there's an equation of the form sum(x_i) = 1 but we have found from other equations
    # that x_i = 0 for all i. 
    # 
    # Unlike with the simpler case of x1 = 1, x2 = 1, x1 + x2 = 1, which we handle above, we can't 
    # isolate which part of the system is contradictory. In this case, we just return None to indicate
    # an inconsistent set of equations.
    if any(map(lambda x: len(x) == 0, ambiguous_clusters)):
        assert not any(map(lambda x: len(x) == 1, ambiguous_clusters)), "Cluster with one unknown variable should have the value of that variable inferred."
        return None
    
    # This will be a superset of anchor_clones from above.
    selected_clones = [clone for clone, select in var2value.items() if select == 1]

    disjoint_sets: UnionFind[IDClone] = UnionFind(
        [c for cluster in ambiguous_clusters for c in cluster]
    )

    for cluster in ambiguous_clusters:
        citer = iter(cluster)
        representative = next(citer)
        for current in citer:
            disjoint_sets.union(representative, current)

    ambiguities = [Ambiguity(s) for s in disjoint_sets.export_sets()]
    
    # print("\n\n---- Final clones ----")
    # for clone in selected_clones:
    #     print(clone)
    #     print(is_contradictory_clone(clone))
    #     print()
    
    # print("\n---- Ambiguities ----")
    # for a in ambiguities:
    #     print(a)
    #     print()
    # print("----------------------")

    return selected_clones, ambiguities


def resolve_contradictory_clone(clone: IDClone, 
                                candidate_uses: Dict[SSAOperand, List[SSAOperator]],
                                reference_uses: Dict[SSAOperand, List[SSAOperator]],
                                id_generator: IDSrc
                               ) -> tuple[List['Equivalence'], List[Ambiguity]]:
    """Some clones have internal ambiguities. These ambiguities can be identified and sometimes 
    resolved by using the same methods for other types of clones.

    This function identifies and, if possible, resolves those ambiguities. This function returns
    the resolved ambiguities in the form of clones and unresolved ambiguities as Ambiguity objects.
    """
    # print("*" * 50)
    # print("Call to resolve_contradictory_clone")
    # print("*" * 50)
    # print(clone)
    # print()

    equivalences: Dict[tuple[SSAOperator, SSAOperator], Equivalence] = {}
    operators: List[SSAOperator] = []

    # Build up some other convenient representations of the clone.
    for equivalence in clone:
        oppair = (equivalence.left, equivalence.right)
        equivalences[oppair] = equivalence
        operators.append(equivalence.left)
        operators.append(equivalence.right)

    # Find which operators belong to the ambiguous part(s) of this clone. Represents a "horizontal slice" across the dataflow in the function pairs.
    operator_co_occurrences = UnionFind(operators)
    for equivalence in clone:
        operator_co_occurrences.union(equivalence.left, equivalence.right)

    # Operators used in only exactly one equivalence will be in a set of size 2 (one for the candidate and one for the reference).
    # Operators used in more than one equivalence will be in a larger set. We'd like to differentiate between these operators if possible.
    ambiguity_classes = list(filter(lambda x: len(x) > 2, operator_co_occurrences.export_sets())) # 
    ambiguous_operators: Set[SSAOperator] = {operator for c in ambiguity_classes for operator in c} # Operators that are part of an ambiguity
    ambiguous_equivalences: Set[tuple[SSAOperator, SSAOperator]] = set() # Equivalences that are part of an ambiguity
    unambiguous_equivalences: List[Equivalence] = []

    # Isolate only the equivalences that are directly ambiguous.
    for equivalence in clone:
        if equivalence.left in ambiguous_operators or equivalence.right in ambiguous_operators:
            ambiguous_equivalences.add((equivalence.left, equivalence.right))
        else:
            unambiguous_equivalences.append(equivalence)

    # Ambiguous clones may span several horizontal slices (e.g. there may be sequences of identical operations for which there is ambiguity).
    # Thus, we group them into "vertical slices"; that is, along the edges of the dataflow graph. Each vertical slice is a clone.
    subclone_builder = UnionFind([equivalences[a] for a in ambiguous_equivalences])
    for equivalence in (equivalences[a] for a in ambiguous_equivalences):
        for operands in zip(*standardize_operands(equivalence.left, equivalence.right)):
            if operands in ambiguous_equivalences:
                subclone_builder.union(equivalence, equivalences[operands])
        if isinstance(equivalence.left, FunctionSSAOperator) and isinstance(equivalence.right, FunctionSSAOperator) and \
           isinstance(equivalence.left.name, SSAOperator) and isinstance(equivalence.right.name, SSAOperator):
            sources = (equivalence.left.name, equivalence.right.name)
            # In the typical case of a method call decomposed into an attribute access and function call
            # (e.g. self.fn(1) is decomposed into t0 = self.fn and t0(1)), we would expect the if condition to be true.
            # However, there are situations where it may validly be false. Consider the following aligned with itself:
            # fn = self.fn
            # fn(1)
            # fn(2)
            if sources in ambiguous_equivalences:
                subclone_builder.union(equivalence, equivalences[sources])

    subclones = [InternalClone(id_generator.make_id(), list(s)) for s in subclone_builder.export_sets()]

    # print("---- Subclones ----")
    # for subclone in subclones:
    #     print(subclone)

    if len(subclones) == 1 and len(unambiguous_equivalences) == 0:
        # This will cause infinate recursion. Allow inference to attempt to differentiate between operators.
        # We do not want to return the clones as an ambiguity, because an ambiguity is meant to represent several
        # clones which contradict each-other. In contrast, the single subclone here is internally contradictory.
        return [], []

    undifferentiated_selection = select_clones(subclones)
    if undifferentiated_selection is None:
        # We can't work out which clone or clones are best. Hopefully inference can sort it out by filling in the clones' premises.
        # Return all clones as an ambiguity.
        selected_clones = []
        ambiguities = [Ambiguity(subclones)]
    # Sometimes there's enough information in the undifferentiated selection process to completely differentiate clones. When this
    # happens, we're done; there's of course no need to add differentiating factors.
    elif len(undifferentiated_selection[1]) == 0: # undifferentiated_selection[1] is the (potentially empty) list of ambiguities.
        selected_clones, ambiguities = undifferentiated_selection
    else:
        # print("Attempting to differentiate based on future information.")

        # Subclones currently only contains the ambiguous parts of the clone. The information that may help
        # differentiate these ambiguous parts is future information about each part; i.e., how each part is used.
        for subclone in subclones:
            differentiators: List[Equivalence] = []
            for equivalence in subclone:
                if equivalence.left in candidate_uses:
                    for leftuse in candidate_uses[equivalence.left]:
                        if equivalence.right in reference_uses:
                            for rightuse in reference_uses[equivalence.right]:
                                ops = (leftuse, rightuse)
                                if ops in equivalences and ops not in ambiguous_equivalences:
                                    differentiators.append(equivalences[ops])
            subclone.equivalences.extend(differentiators)

        # We want allow_resolvable_contradictions to be true here. Adding future use information can result in
        # some contradictions (as happens when values from multiple clones are condensed into the same phi-node).
        # allow_resolvable_contradictions lets us ignore these cases while hopefully recieving differentiating 
        # information from the other uses. This is okay, because the differentiating factors do not become a part
        # of the final clone.
        selection = select_clones(subclones, allow_resolvable_contradictions=True)
        if selection is not None:
            selected_clones, ambiguities = selection
        else:
            # Adding the future information lead to fundamentally contradictory clones. We remove those
            # parts of the clones and reduce the to where they were in our first call to select_clones (which did work).
            # print("Adding use information caused inconsistencies. Using reduced internal clones.")
            for subclone in subclones:
                subclone.equivalences = subclone.original_equivalences # This aliases the two lists, but that should be fine, as InternalClones are read-only outside of this function.
            selected_clones, ambiguities = undifferentiated_selection

    # print("---- Recursively Selected Clones ----")
    # for subclone in selected_clones:
    #     print(subclone)
    
    # print()
    # for ambiguitiy in ambiguities:
    #     print(ambiguitiy)
    # print()
    # print()

    for subclone in selected_clones:
        if is_contradictory_clone(subclone):
            resolved, subambiguous = resolve_contradictory_clone(subclone, candidate_uses, reference_uses, id_generator)
            unambiguous_equivalences.extend(resolved)
            ambiguities.extend(subambiguous)
        else:
            unambiguous_equivalences.extend(subclone) # Clones are iterable

    # Filter out ambiguities that have at least one internally contradictory subclone.
    # If only the contradictory subclone is removed, the noncontradictory clones will be favored
    # in conflict resolution. Noncontradictory clones do not always correspond to an intuitive
    # alignment, however.
    # An alternative solution is to nest CompositeClones inside ambiguities (i.e. call resolve_contradictory_clone
    # on each internally contradictory clone in the ambiguity.)
    ambiguities = [ambiguity for ambiguity in ambiguities if all(
        not is_contradictory_clone(c) for c in ambiguity.clones
    )]

    return unambiguous_equivalences, ambiguities


########################################
#                                      #
#    Core method for finding clones    #
#                                      #
########################################
def find_optimal_clones(candidate_ir: Function, 
                reference_ir: Function, 
                copbyid: Dict[Tuple[str, Optional[Union[FunctionPointer, str]]], List[SSAOperator]],
                ropbyid: Dict[Tuple[str, Optional[Union[FunctionPointer, str]]], List[SSAOperator]],
                verbose: bool = False
               ) -> tuple[List[ProxyCloneOperator], List[Ambiguity]]:
    """Find the largest sets of clones in each function pair that don't conflict with eachother.

    Return sets of conflicting clones that can't be differentiated at this stage as Ambiguity objects.
    """
    param_locs = parameter_locations(candidate_ir, reference_ir)

    clones: List[BacksliceClone] = []
    for opid in set(copbyid.keys()).intersection(set(ropbyid.keys())):
        # Function pointers are only equivalent if their sources are equivalent.
        for cand_op in copbyid[opid]:
            for ref_op in ropbyid[opid]:
                if not isinstance(opid[1], FunctionPointer) or BacksliceClone.same_operations(cand_op, ref_op, param_locs):
                    clone = BacksliceClone(candidate_ir, cand_op, reference_ir, ref_op)
                    if len(clone) > 1: # clones of size 1 are simply pairs of operators with the same operation - something that's already encoded in copbyid and ropbyid.
                        clones.append(clone)

    # Subsume and merge clones using a CloneMerger. This is easier when the clones are sorted by size
    # (more specifically, it simplifies the implementation of CloneMerger).
    clones.sort(key = lambda x: len(x), reverse=True) # "reverse=True" means "descending"
    clone_merger = CloneMerger()
    if verbose:
        print("\n\n\n---- Now Inserting Clones ----")
    for clone in clones:
        if verbose:
            print(clone)
            print()
        clone_merger.add_clone(clone)

    if verbose:
        print("\n---- Clone Graph ----")
        print(clone_merger)

    ambiguities = []
    out_clones = []
    id_src = IDSrc(clone_merger.id_src)
    for clone in clone_merger:
        if is_contradictory_clone(clone):
            unambiguous_equivalences, internal_ambiguities = resolve_contradictory_clone(clone, initialize_uses(candidate_ir), initialize_uses(reference_ir), id_src)
            ambiguities.extend(internal_ambiguities)
            out_clones.append(CompositeClone.create_composite_clone(clone.id, unambiguous_equivalences, internal_ambiguities))

            # print("---- Composite Clone ----")
            # print(out_clones[-1])
            # print()
            # print()
        else:
            out_clones.append(clone)
    
    initialize_uses.cache_clear() # If this is used later in the process, move this after then.

    out_clones = list(map(ProxyCloneOperator.from_idclone, out_clones))
    return (out_clones, ambiguities)

###
# Classes for supporting an inductive proof.
###
class Clause(ABC):
    """Represents an arbitrarily complicated expression.
    """
    def contradicts(self):
        raise NotImplementedError("Clause supertype does not implement method 'contradicts'.")

class Equivalence(Clause):
    def __init__(self, left, right):
        """Represents two objects being equivalent.
        """
        self.left = left
        self.right = right
    
    def __eq__(self, other):
        return isinstance(other, Equivalence) and other.left == self.left and other.right == self.right
    
    def __hash__(self):
        return hash(self.left) + hash(self.right)
    
    def contradicts(self, other) -> bool:
        """Contradicts on the assumption that three variables cannot be all equivalent, which is not
        true in general but is for our specific problem.
        """
        if isinstance(other, CloneClause):
            return other.contradicts(self)
        assert isinstance(other, Equivalence), f"Expected Equivalence or CloneClause; got {type(other)}"
        # != is the same operation as xor
        return (self.left == other.left) != (self.right == other.right)
    
    def __repr__(self):
        return f"{self.left} == {self.right}"
    
class CloneClause(Clause):
    def __init__(self, clone_id: int, equivalences: List[Equivalence]):
        self.clone_id = clone_id
        self.equivalences = equivalences
    
    def __eq__(self, other):
        return isinstance(other, CloneClause) and other.clone_id == self.clone_id
    
    def __iter__(self) -> Iterator[Equivalence]:
        yield from self.equivalences

    def __len__(self) -> int:
        return len(self.equivalences)
    
    def __hash__(self):
        return self.clone_id
    
    def __repr__(self):
        return f"Clone{self.clone_id}"
    
    def contradicts(self, other):
        if isinstance(other, Equivalence):
            for_comparison = [other]
        else:
            assert isinstance(other, CloneClause)
            for_comparison = other.equivalences
        
        for other_eq in for_comparison:
            for this_eq in self.equivalences:
                if other_eq.contradicts(this_eq):
                    return True
        return False

class CompositeCloneClause(CloneClause):
    def __init__(self, clone_id: int, equivalences: List[Equivalence], ambiguities: List[Ambiguity], ambiguity_frontier: Set[Equivalence]):
        super().__init__(clone_id, equivalences)
        self.ambiguities = ambiguities
        self.ambiguity_frontier = ambiguity_frontier

    def __repr__(self):
        return f"CompositeClone{self.clone_id}"
    
class TrueClause(Clause):
    def __init__(self):
        self.description = "true"
    
    def __eq__(self, other):
        return isinstance(other, TrueClause)
    
    def __hash__(self):
        return hash(self.description)
    
    def __repr__(self):
        return self.description
    
class WeightedImplication(Clause):
    def __init__(self, condition: Clause, conclusion: Clause, weight: float):
        # Mathematically, you can do this, but the custom solver here isn't set up to handle this case.
        assert not isinstance(condition, WeightedImplication), "Cannot use an implication as the condition of another implication."
        self.condition = condition
        self.conclusion = conclusion
        self.weight = weight # Weight of 1.0 means 100% confidence in its truth value.
    
    def __repr__(self):
        return f"{self.condition} => {self.conclusion}, weight={self.weight}"

class ContradictionError(Exception):
    pass

class Contradiction:
    def __init__(self, boundary: Set[Clause], removed: Set[Clause], implications: List[WeightedImplication]):
        self.boundary = boundary
        self.removed = removed
        self.implications = implications
    
    def __repr__(self):
        return "Contradiction\n  " + \
        "removed\n    " + "\n    ".join([repr(r) for r in self.removed]) + "\n  " + \
        "boundary\n    "  + "\n    ".join(repr(b) for b in self.boundary) + "\n  " + \
        "implications\n    " + "\n    ".join([repr(i) for i in self.implications])


class FactGraph:
    """A FactGraph is a directed, acyclic graph (DAG) which models a proof done in first-order logic.
    Nodes are propositional logic statements for which we have evidence towards being true, and edges
    represent the implications used to provide that evidence.

    In the context of finding an alignment, the FactGraph represents the isomorphic subgraphs of the 
    two functions that are aligned.
    """
    class Node:
        """A node in a FactGraph represents a fact for which there is at least some evidence towards its
        truth. A node which is a fact is considered true and can be used to provide evidence towards
        additional facts.
        """
        def __init__(self, fact: Clause, truth_threshold: float = 0.0, is_fact: bool = False):
            """
            Create a node in the fact graph.
            
            :param fact: The statement that this node represents.
            :param truth_threshold: the confidence value at which this node was declared true.
            :param is_fact: whether to use this node in proving other nodes and adding those nodes to the graph.
            """
            assert not isinstance(fact, WeightedImplication), "Implications function as graph edges, not nodes."
            assert 0.0 <= truth_threshold <= 1.0, "Truth threshold must be between 0 and 1, inclusive."
            self.fact = fact
            self.truth_threshold = truth_threshold
            self.is_fact = is_fact
            self.out_edges: List[FactGraph.Edge] = []
            self.successors: List[FactGraph.Node] = [] # To make removing nodes easier.
        
        def add_edge(self, impl: WeightedImplication):
            self.out_edges.append(FactGraph.Edge(impl))
            self.truth_threshold += impl.weight

        def remove_edge(self, impl: WeightedImplication):
            for i, edge in enumerate(self.out_edges):
                if edge.implication == impl:
                    break
            else:
                raise ValueError(f"Implication {impl} not found at node:\n{self}")
            self.out_edges.pop(i)
            self.truth_threshold -= impl.weight

        def __repr__(self):
            return f"Node({self.fact}, {self.is_fact}, {self.truth_threshold})\n    " + "\n    ".join(repr(e) for e in self.out_edges)
    
    class GroundTruthNode(Node):
        """A GroundTruthNode repesents a premise to the proof problem that is assumed true."""
        def __init__(self, fact: Clause):
            super().__init__(fact, 1.0)
        
        def add_edge(self, impl: WeightedImplication):
            raise NotImplementedError(f"Cannot add edge to ground-truth node (fact {self.fact}).")
        
        def remove_edge(self, impl: WeightedImplication):
            raise NotImplementedError(f"Cannot remove edge from ground-truth node (fact {self.fact}).")
        
        def __repr__(self):
            return f"GroundTruthNode({self.fact})"

    class Edge:
        """An edge (here a directed edge) in a FactGraph represents an implication used to 
        provide evidence that a fact (node) is true.
        """
        def __init__(self, implication: WeightedImplication):
            self.implication = implication
        
        def __repr__(self):
            return f"Edge({self.implication})"

    def __init__(self, ground_truth: Iterable[Clause]):
        # Graph is implemented as an adjacency list.
        self.fact2node: Dict[Clause, FactGraph.Node] = {gt: FactGraph.GroundTruthNode(gt) for gt in ground_truth}

    def apply_implication(self, impl: WeightedImplication, conclusion_is_fact: bool):
        assert impl.condition in self.fact2node, f"Condition of implication {impl} must be in the fact graph already for the condition to apply."
        assert impl.condition in self.fact2node, f"Condition of implication {impl} must be considered a fact in order to be used to prove other facts."
        # self.nodes[impl.condition].add_edge()
        if impl.conclusion not in self.fact2node:
            self.fact2node[impl.conclusion] = FactGraph.Node(impl.conclusion, is_fact=conclusion_is_fact)
        conclusion_node = self.fact2node[impl.conclusion]
        conclusion_node.add_edge(impl)
        conclusion_node.is_fact = conclusion_is_fact
        # Set the successor of the condition node. Note that this is a pointer to the node, not the connecting edge.
        self.fact2node[impl.condition].successors.append(conclusion_node)
        # if conclusion_is_fact and isinstance(impl.conclusion, CompositeCloneClause):
        #     for frontier_fact in impl.conclusion.ambiguity_frontier:
        #         assert frontier_fact not in self.fact2node
        #         frontier_node = FactGraph.AmbiguityFrontierNode(frontier_fact, conclusion_node.truth_threshold)
        #         frontier_node.add_edge(impl.conclusion)
        #         conclusion_node.successors.append(frontier_node)

    def remove_implication(self, impl: WeightedImplication):
        """Remove an implication from the FactGraph.
        """
        self.fact2node[impl.conclusion].remove_edge(impl)

    def unwind(self, contradiction: Clause) -> Tuple[Set[Clause], Set[Clause], Set[Clause], Set[WeightedImplication]]:
        """Removes a given fact from the fact graph and all future facts proved from this fact.
        Returns the implications used to prove all removed facts, as well as the facts used in the
        conditions of those implications (not including the contradiction itself.)
        """
        assert contradiction in self.fact2node, f"Can only remove a fact from the FactGraph that is already in it; {contradiction} is not in the FactGraph"
        contradiction_root = self.fact2node[contradiction]
        assert contradiction_root.is_fact, f"Only a fully-realized fact can be a contradiction."
        assert not isinstance(contradiction_root, FactGraph.GroundTruthNode), f"Cannot remove ground-truth node {contradiction}."

        # Boundary facts: these facts are not removed from the graph, but some nodes that have been proven based on this
        # node have been removed.
        boundary_facts: Set[Clause] = set()
        removed_facts: Set[Clause] = set()
        unproven_propositions: Set[Clause] = set() # Removed propositions that are not yet considered facts.
        removed_impls: Set[WeightedImplication] = set()

        # Recursively remove nodes from the graph.
        def remove(node: FactGraph.Node):
            for edge in node.out_edges:
                implication = edge.implication
                assert implication.condition in self.fact2node
                removed_impls.add(implication)
                boundary_facts.add(implication.condition)
                self.fact2node[implication.condition].successors.remove(node)
            # The call to set() on node.successors is important for two reasons:
            # 1. It is possible to have two different edges from one node to another (due to clones; 
            #    the different edges come from different operators within the clone). However, it only
            #    takes one remove(successor) to eliminate all successor edges for a given node.
            # 2. The .remove() method of the successor list is called in the recursive call; using the
            #    node.successors list directly would mess up the iterator.
            for successor in set(node.successors):
                if successor.fact not in removed_facts and successor.fact not in unproven_propositions:
                    remove(successor)
            # Note: for dealing with cycles in the FactGraph, it may be necessary to move the following if statement
            # (whose condition is node.is_fact) before the for loop above.
            # 
            # removed_facts will be deleted from the FactRegistry's self.facts set (which contains only proven true facts)
            # and will be used in conflict resolution. Facts which have not been fully proven should be involved with neither
            # of these things. Thus, we do not add partially proven facts to removed_facts.
            if node.is_fact:
                removed_facts.add(node.fact)
            else: # We must record unproven facts to remove them from the FactRegistry's fact2weight dictionary.
                unproven_propositions.add(node.fact)
            del self.fact2node[node.fact]
        
        remove(contradiction_root)

        boundary_facts = boundary_facts - removed_facts
        return boundary_facts, removed_facts, unproven_propositions, removed_impls
    
    def assert_fact_true(self, fact: Clause):
        assert fact in self.fact2node, "Cannot assert that a fact is true if it does not have a node in the fact graph!"
        self.fact2node[fact].is_fact = True

    def get_weight(self, fact: Clause) -> float:
        return self.fact2node[fact].truth_threshold

    def __repr__(self):
        return "FactGraph\n  " + "\n  ".join([repr(n) for _, n in self.fact2node.items()]) + "\n"    

class FactRegistry:
    """A class which manages facts that have been proven so far as well as implications used for
    proving more facts.
    """
    def __init__(self, ground_truth: Iterable[Clause], injective: bool = True):
        self.facts: Set[Clause] = set(ground_truth)
        self.fact2weight: Dict[Clause, float] = {}
        self.implications: Dict[Clause, List[WeightedImplication]] = {}
        self.graph = FactGraph(ground_truth)
        self.injective = injective

    def flattened_facts(self) -> Iterator[Equivalence]:
        for fact in self.facts:
            if isinstance(fact, Equivalence):
                yield fact
            elif isinstance(fact, CloneClause):
                for equivalence in fact.equivalences:
                    yield equivalence
            elif isinstance(fact, TrueClause):
                continue
            else:
                raise TypeError(f"Unexpected fact type in FactRegitry: {type(fact)}")
    
    def add_implication(self, implication: WeightedImplication):
        if implication.condition not in self.implications:
            self.implications[implication.condition] = [implication]
        else:
            self.implications[implication.condition].append(implication)
    
    def satisfy_conditions(self, condition: Clause) -> List[Union[WeightedImplication, Contradiction]]:
        if condition in self.implications:
            impls = self.implications[condition]
            del self.implications[condition]
            results = []
            for impl in impls:
                if isinstance(impl.conclusion, WeightedImplication):
                    self.add_implication(impl.conclusion)
                    # The main control loop assumes that all implications exist at the beginning
                    # of iteration, and thus each condition need only be seen once to satisfy all
                    # corresponding implications. This is not so in the case of function pointers,
                    # where satisfying a condition may reveal an implication with a condition that has
                    # already been satisfied. As a result, we return that (already proven true) condition
                    # again.
                    if impl.conclusion.condition in self.facts:
                        results.append(impl.conclusion.condition)
                else:
                    result = self.apply_implication(impl)
                    if result is not None:
                        results.append(result)
            return results
        return []
    
    def add_fact(self, fact: Clause) -> Optional[Contradiction]:
        assert not isinstance(fact, WeightedImplication), "Implications should go in the implication registry."
        self.facts.add(fact) # It's cleaner to add this now and remove it if it contradicts.
        if not self.injective:
            return None # Contradictions only apply in injective mode.
        for existing_fact in self.facts:
            if isinstance(existing_fact, TrueClause):
                continue
            if fact.contradicts(existing_fact):
                # print("\n\n\n!!!!!!!!!!!! Contradiction Reached !!!!!!!!!!!!!!!")
                # print(f"{fact} contradicts with {existing_fact}.")
                # print(self.graph)
                # print()
                # print(f"---- Attempting to remove {fact} ----")
                boundary, removed, unproven, impls = self.graph.unwind(fact)
                boundary2, removed2, unproven2, impls2 = self.graph.unwind(existing_fact)
                removed.update(removed2)
                boundary.update(boundary2)
                boundary = boundary - removed # the second call to unwind can remove items in the first call's boundary.
                for rm_fact in removed:
                    self.facts.remove(rm_fact)
                    del self.fact2weight[rm_fact]
                for rm_proposition in itertools.chain(unproven, unproven2):
                    del self.fact2weight[rm_proposition]
                return Contradiction(boundary, removed, list(itertools.chain(impls, impls2)))
    
    def remove_fact(self, fact: Clause) -> Contradiction:
        assert fact in self.facts, f"Cannot remove {fact} because this fact has not yet been proven!"
        boundary, removed, unproven, impls = self.graph.unwind(fact)
        for rm_fact in removed:
            self.facts.remove(rm_fact)
            del self.fact2weight[rm_fact]
        for rm_proposition in unproven:
            del self.fact2weight[rm_proposition]
        return Contradiction(boundary, removed, list(impls))
    
    def apply_implication(self, impl: WeightedImplication) -> Optional[Union[Clause, Contradiction]]:
        # An implication's conclusion may already be in self.facts if it's been assumed true (possible when alpha < 1.0).
        # This can also happen when partial_loops=False.
        if impl.conclusion in self.facts:
            self.fact2weight[impl.conclusion] += impl.weight
            self.graph.apply_implication(impl, True)
            return None

        if impl.conclusion in self.fact2weight:
            self.fact2weight[impl.conclusion] += impl.weight
        else:
            self.fact2weight[impl.conclusion] = impl.weight
        
        # A fact with weight 1 means 100% confidence of truth.
        if self.fact2weight[impl.conclusion] > 1.0 or math.isclose(self.fact2weight[impl.conclusion], 1.0):
            self.graph.apply_implication(impl, True)
            result = self.add_fact(impl.conclusion)
            if result is None:
                return impl.conclusion
            else:
                return result # A contradiction has been found.
        else:
            self.graph.apply_implication(impl, False)
        return None
    
    def revoke_implication(self, impl: WeightedImplication) -> Optional[Contradiction]:
        """Remove an implication from a partially or completely built proof, including 
        the fact graph. Decreases the proven weight of the conclusion, then removes and returns
        the conclusion and all subsequent facts fully or partially proved if the conclusion is
        no longer valid.

        This method is intended to be used to remove implications used for loopbreaking cycles
        of dependencies. Such implications, by nature, always have a condition of type TrueClause.
        This method currently checks that this is the case.

        :param impl: the implication to remove.
        :returns: None if the conclusion of the implication is still valid after removal, and the 
        items removed (in a Contradiction object) otherwise.
        """
        assert impl.conclusion in self.fact2weight, "Cannot revoke an implication that has not been applied!"
        assert isinstance(impl.condition, TrueClause), f"Should only revoke loop bootstrapping edges (which start with TrueClause()) but are revoking {impl}."
        self.fact2weight[impl.conclusion] -= impl.weight
        updated_weight = self.fact2weight[impl.conclusion]
        if not math.isclose(updated_weight, 1.0) and updated_weight < 1.0:
            # Removing this implication causes the fact to be invalid. Remove the fact, all other implications to it,
            # and all subsequent facts proved from this fact.
            return self.remove_fact(impl.conclusion)
        else:
            # This fact is still valid, but we should ensure that the graph is updated.
            self.graph.remove_implication(impl)
            return None
    
    def __repr__(self):
        return "FactRegistry" + \
            ("\n  " + "\n  ".join([repr(f) for f in self.facts]) if len(self.facts) > 0 else "") + \
            ("\n  " + "\n  ".join([f"{f} weight={w}" for f, w in self.fact2weight.items()]) if len(self.fact2weight) > 0 else "") + \
            ("\n  " + "\n  ".join([repr(i) for _, impls in self.implications.items() for i in impls]) if len(self.implications) > 0 else "") + "\n"
    
    def assert_most_likely_facts_true(self, alpha: float) -> Iterable[Clause]:
        """Asserts that the most likely facts that don't contradict any existing fact are true and
        returns those facts. Only facts with a weight of at least alpha are considered.
        """
        max_weight: float = 0.0
        facts_to_assume: List[Clause] = []
        # TODO: compatible_with_known_facts should only be used in injective mode.
        compatible_with_know_facts = lambda fact: all(not fact.contradicts(f) for f in self.facts if not isinstance(f, TrueClause))
        for fact, weight in self.fact2weight.items():
            if fact in self.facts:
                continue
            # Check with math.isclose() first in case weight and max_weight are close but weight is very sligtly
            # more that max_weight.
            if math.isclose(weight, max_weight) and compatible_with_know_facts(fact):
                assert weight > 0, f"Fact with zero weight is in self.fact2weight! ({fact}: {weight})"
                facts_to_assume.append(fact)
            elif weight > max_weight and weight >= alpha and compatible_with_know_facts(fact):
                facts_to_assume = [fact]
                max_weight = weight

        # Update the internal state of the FactRegistry and its corresponding graph to reflect that these
        # clauses are now facts.
        for fact in facts_to_assume:
            # We already checked that these facts do not contradict anything that already exists; directly adding to self.facts
            # here avoids us having to check this again, which is expensive.
            self.facts.add(fact)
            self.graph.assert_fact_true(fact)

        return facts_to_assume
    
class ContradictionManager(UnionFind):
    class Group:
        def __init__(self):
            self.removed = []
            self.implications = []
            self.boundary = []

    def __init__(self, contradictions: List[Contradiction], fact_registry: FactRegistry, candidate_ir: Function, reference_ir: Function):
        operators: Set[SSAOperator] = set() # Contains operators that are involved in a contradiction.

        def _add_contradiction_operators(contra: Contradiction):
            for clause in contra.removed:
                if isinstance(clause, Equivalence):
                    operators.add(clause.left)
                    operators.add(clause.right)
                else:
                    assert isinstance(clause, CloneClause)
                    for equivalence in clause.equivalences:
                        operators.add(equivalence.left)
                        operators.add(equivalence.right)
        
        for contra in contradictions:
            _add_contradiction_operators(contra)

        # Will be updated later with additional facts from the fact registry that contradict with the supplied contradictions.
        self.removed = {fact for contradiction in contradictions for fact in contradiction.removed}
        
        # There may be facts in the graph that aren't included in the contradictions passed to this function that
        # nonetheless contradict with one of the operators in the contradictions. This is due to the order that 
        # facts are added to the graph. (This happens when there are an odd number > 1 of contradicting operators).

        # Removing clauses can cause the 'operators' set to grow (when a clone is removed.) When this happens, there 
        # may be other proven facts that contradict with the new operators. Therefore, we loop through again and find 
        # those facts if there indeed were new operators added. Typically, however, no new operators are added and 
        # there is only one iteration.
        prior_iter_size = 0
        while len(operators) > prior_iter_size:
            # Record facts to remove here and remove them in a subsequent loop to avoid damaging the iterator.
            facts_to_remove: Set[Clause] = set()
            for fact in fact_registry.facts:
                if isinstance(fact, Equivalence):
                    if fact.left in operators and fact.right in operators:
                        facts_to_remove.add(fact)
                if isinstance(fact, CloneClause):
                    for equivalence in fact:
                        if equivalence.left in operators and equivalence.right in operators:
                            facts_to_remove.add(fact)
                            break
            
            prior_iter_size = len(operators)
            for fact in facts_to_remove:
                if fact not in self.removed:
                    remove_result = fact_registry.remove_fact(fact)
                    _add_contradiction_operators(remove_result)
                    contradictions.append(remove_result)
                    self.removed.update(remove_result.removed)    

        super().__init__(operators)
        for contradiction in contradictions:
            self.add_contradiction(contradiction)
        self.contradictions = contradictions
        self.fact_registry = fact_registry
        self.candidate_ir = candidate_ir
        self.reference_ir = reference_ir

    def removed_equivalences(self, contradiction: Contradiction) -> Iterator[Equivalence]:
        for clause in contradiction.removed:
            if isinstance(clause, Equivalence):
                yield clause
            else:
                assert isinstance(clause, CloneClause)
                for equivalence in clause.equivalences:
                    yield equivalence
    
    def add_contradiction(self, contradiction: Contradiction):
        removed = list(self.removed_equivalences(contradiction))
        for clause in removed:
            self.union(clause.left, clause.right)

        if len(removed) > 1:
            for eq1, eq2 in zip(itertools.islice(removed, 1, None), removed):
                self.union(eq1.left, eq2.left) # Doing with the right as well is redundant, since left and right have already been merged.

    def resolve(self):
        # Sort the contradiction objects themselves into classes where they share operations.
        operator_sets = self.export_sets()

        # print("---- Opsets ----")
        # for opset in self.export_sets():
        #     print(opset)
        #     print()
        # print()
        # print()

        assert isinstance(operator_sets, List), "ContradictionManager expects a list of sets, not an arbitrary iterable."
        contradiction_groups: List[List[Contradiction]] = [[] for _ in range(len(operator_sets))] # Maintain a parallel array.
        for contradiction in self.contradictions:
            operators = {op for equivalence in self.removed_equivalences(contradiction) for op in (equivalence.left, equivalence.right)}
            for i, op_set in enumerate(operator_sets):
                if operators.issubset(op_set):
                    contradiction_groups[i].append(contradiction)
                    break
            else:
                class InvariantError(Exception):
                    pass
                raise InvariantError("Contradiction does not belong to any group!")

        boundary: List[Clause] = []
        # print(f"******* Contradiction Groups ({len(contradiction_groups)} total) *******")
        for i, group in enumerate(contradiction_groups):
            # print(f"-- Group {i}")
            # for c in group:
            #     print(c)
            boundary.extend(self.resolve_contradiction_group(group))
            # print()
            # print()

        # print("************ End of contradiction resolution ************")
        
        return boundary
        
    def resolve_contradiction_group(self, group: List[Contradiction]) -> List[Clause]:
        """Resolve contradiction using the following strategy:

        Let the SMT solver select from the available options.
        - If there is a big clone that dominates the options, that will be selected.
        - If it is truly ambiguous, an arbitrary solution will be chosen by the solver.
        - If there is no solution, pick one arbitrarily.

        There is much room to improve the precision of this method, perhaps by using control alignment,
        and/or control flow partial-ordering.
        """
        options: List[Iterable[Equivalence]] = []
        for possible_fact in (r for contradiction in group for r in contradiction.removed):
            if isinstance(possible_fact, Equivalence):
                options.append([possible_fact])
            else:
                assert isinstance(possible_fact, CloneClause)
                options.append(possible_fact) # CloneClauses are Iterable[Equivalence] already.
        
        solution: List[Iterable[Equivalence]] = nonconflicting_clones(options)
        if solution is None:
            # Solution is None; this group is fundamentally contradictory. Choose arbitrarily from within the group.
            # Greedily choose the example that aligns the most operators, then attempt to choose subsequent operators that
            # do not conflict.
            options.sort(key=len, reverse=True)
            solution = []
            for option in options:
                if isinstance(option, List):
                    assert len(option) == 1
                    option = option[0]
                if all(map(lambda x: not option.contradicts(x), solution)):
                    solution.append(option)
        else:
            # Unwrap the single equivalences from the list wrappers used for the call to nonconflicting_clones.
            solution = [x[0] if isinstance(x, List) else x for x in solution]

        selected_facts = set(solution)
        # Add back the implications that will help prove this fact.
        for implication in (impl for contradiction in group for impl in contradiction.implications):
            if implication.condition in selected_facts or implication.conclusion in selected_facts: 
                self.fact_registry.add_implication(implication)

        # May include duplicate boundary conditions. This should be harmless, but may be slightly less efficient.
        boundary = [b for contradiction in group for b in contradiction.boundary if b not in self.removed]

        # This may not be true; if it isn't, we'll have to do one of the following:
        # - Have a more complicated boundary-resolution process (possibly involving all contradictions)
        # - Assert the facts directly in the graph and treat them like GroundTruthNodes (would require a new node type).
        assert all(map(lambda x: x in self.fact_registry.facts, boundary)), "Attempting to prove the results of conflict resolution; however, we cannot assume the boundary."

        return boundary


def nonconflicting_clones(clones: Iterable[Iterable[Equivalence]]) -> Optional[List[Iterable[Equivalence]]]:
    """Determine if the clones conflict. If they do, return None, if they don't, return one possible solution.
    """
    constraints: List[BoolRef] = []
    equations: Dict[SSAOperator, Set[Int]] = {}
    var2clone: Dict[Int, ProxyCloneOperator] = {}
    for i, clone in enumerate(clones):
        clone_var = Int(f"x{i}")
        var2clone[clone_var] = clone
        constraints.append(Or(clone_var == 0, clone_var == 1))
        for equivalence in clone:
            if equivalence.left in equations:
                equations[equivalence.left].add(clone_var)
            else:
                equations[equivalence.left] = {clone_var}
            if equivalence.right in equations:
                equations[equivalence.right].add(clone_var)
            else:
                equations[equivalence.right] = {clone_var}
    
    for _, vars in equations.items():
        constraints.append(Sum(vars) == 1)
    constraints = list(set(constraints))

    solver = Solver()
    solver.add(constraints)
    solver.set("timeout", SOLVER_TIMEOUT)
    if solver.check() != sat:
        return None
    solution = solver.model()
    
    selected_clones: List[CloneMerger.MClone] = []
    for clone_var, clone in var2clone.items():
        if solution.get_interp(clone_var).as_long() == 1:
            selected_clones.append(clone)

    return selected_clones
    
            
def phi_operands_equivalent(cand_operand: SSAOperand, ref_operand: SSAOperand, cand_loopbreak: bool, ref_loopbreak: bool) -> Optional[Clause]:
    """Precondition: this operand is a PHI_OP
    """
    if cand_loopbreak and ref_loopbreak and isinstance(cand_operand, SSAOperator) and isinstance(ref_operand, SSAOperator):
        # These are both phi nodes at the head of a loop with the loops coming from the same index.
        return TrueClause()
    return operands_equivalent(cand_operand, ref_operand)


def operands_equivalent(cand_operand: SSAOperand, ref_operand: SSAOperand) -> Optional[Clause]:
    if isinstance(cand_operand, SSAOperator) and isinstance(ref_operand, SSAOperator):
        return Equivalence(cand_operand, ref_operand)
    if isinstance(cand_operand, Parameter) and isinstance(ref_operand, Parameter):
        return Equivalence(cand_operand, ref_operand) # Technically we already know if this is true, or not, so we could shortcut here. Would require a map from parameter to index, though.
    if isinstance(cand_operand, GlobalVariable) and isinstance(ref_operand, GlobalVariable):
        return TrueClause() if cand_operand.name == ref_operand.name else None
    return TrueClause() if cand_operand == ref_operand else None

def ordered_control_dependencies(fn: Function, strict_dependencies: dict[BasicBlock, list[BasicBlock]]) -> dict[BasicBlock, list[BasicBlock]]:
    """Return control dependencies in a consistent order such that two operators align
    only if their dependencies align in that order. This function expects strict control
    dependencies, i.e., those where a given block does not depend on itself.

    This function treats the argument as immutable and returns a copy if the argument may
    need to be modified.
    """
    if all(len(dependencies) <= 1 for dependencies in strict_dependencies.values()):
        # All basic blocks have at most one dependent so each block's dependencies are trivially ordered.
        # This is usually the case.
        return strict_dependencies
    
    # Make a copy because we'll need to access strict_dependencies while sorting.
    # We'll sort and return this copy.
    sorted_dependencies = {bb: deps.copy() for bb, deps in strict_dependencies.items()}

    def indirectly_depends_on(x: BasicBlock, y: BasicBlock, encountered: set[BasicBlock]) -> bool:
        if x == y:
            return True
        encountered.add(x)
        return any(indirectly_depends_on(dependency, y, encountered) 
                   for dependency in strict_dependencies[x] 
                   if dependency not in encountered
                  )

    dfo_order: dict[BasicBlock, int] | None = None
    def dfo_traversal():
        nonlocal dfo_order
        rpo = analysis.postorder_traversal(fn.entry_block, [], set())
        rpo.reverse()
        dfo_order = {bb: i for i, bb in enumerate(rpo)}

    def ctrl_cmp(l: BasicBlock, r: BasicBlock) -> int:
        lr_true = indirectly_depends_on(l, r, set())
        rl_true = indirectly_depends_on(r, l, set())
        if lr_true == rl_true:
            if dfo_order is None:
                dfo_traversal()
            return -1 if dfo_order[l] < dfo_order[r] else 1
        if lr_true:
            return 1
        else: # rl_true
            return -1
    
    for dependencies in sorted_dependencies.values():
        if len(dependencies) > 1:
            dependencies.sort(key=functools.cmp_to_key(ctrl_cmp))
        # else the dependencies are trivially sorted.

    return sorted_dependencies

@functools.lru_cache(2)
def operator_level_control_dependence(fn: Function) -> dict[SSAOperator, list[tuple[SSAOperator, int]]]:
    strict_dependencies = {bb: [d for d in dependencies if d != bb] for bb, dependencies in analysis.control_dependence(fn).items()}

    def target_block_reachable_from(current: BasicBlock, target: BasicBlock, encountered: set[BasicBlock]):
        if current in encountered:
            return False
        encountered.add(current)
        if current == target:
            return True
        else:
            return any(target_block_reachable_from(successor, target, encountered) for successor in current.successors)

    branches = {}
    for bb, dependence in strict_dependencies.items():
        bb_branches = {}
        for dependency in dependence:
            first = target_block_reachable_from(dependency.successors[0], bb, set(dependence))
            second = target_block_reachable_from(dependency.successors[1], bb, set(dependence))
            assert first != second, f"Control dependency branch property violated!"
            # 0: no branches reach bb
            # 1: the first branch reaches bb
            # 2: the second branch reaches bb
            # 3: both branches reach bb
            # Note: in the case of control dependencies specifically, this should
            # always be 1 or 2 (hence the assert above)
            bb_branches[dependency] = 1 * first + 2 * second
        branches[bb] = bb_branches

    operator_dependence: dict[SSAOperator, list[SSAOperator]] = {}
    for bb, dependence in ordered_control_dependencies(fn, strict_dependencies).items():
        dependent = [(d.operators[-1], branches[bb][d]) for d in dependence]
        for op in bb:
            operator_dependence[op] = dependent

    return operator_dependence

def loopbreaking_control_edges(fn: Function) -> dict[SSAOperator, set[SSAOperator]]:
    """Though rare, it is possible to have the head of a loop be control dependent on one
    or more basic blocks from inside the loop. This can create cyclic dependencies, of either
    entirely control dependencies or control and dataflow dependencies. This function identifies 
    the minimal set of control dependence edges that are assumed true in order to build an 
    alignment.

    :param fn: a function
    :returns: a dict relating the last operator at the start of the cycle-breaking edge to a set
    of the operators at the end of the edge.
    """
    loopbreaking_edges: dict[SSAOperator, set[SSAOperator]] =  {}
    loops = analysis.find_loops(fn)
    strict_dependencies = {bb: [d for d in dependencies if d != bb] for bb, dependencies in analysis.control_dependence(fn).items()}

    def add_loopbreaking_edges(dependency: BasicBlock, dependent: BasicBlock):
        if dependency.operators[-1] in loopbreaking_edges:
            loopbreaking_edges[dependency.operators[-1]].update(dependent.operators)
        else:
            loopbreaking_edges[dependency.operators[-1]] = set(dependent.operators)

    for loop in loops:
        internal_dependencies = tuple(d for d in strict_dependencies[loop.head] if d in loop.body)
        if len(internal_dependencies) > 0:
            body: set[BasicBlock] = set(loop.body)
            state: dict[BasicBlock, tuple[BasicBlock,...]] = {}
            worklist: deque[tuple[BasicBlock, tuple[BasicBlock,...]]] = deque()
            worklist.append((loop.head, internal_dependencies))
            while len(worklist) > 0:
                block, pre_deps = worklist.popleft()
                post_deps = tuple(d for d in pre_deps if d != block)
                add_successors = True
                if block in state:
                    if state[block] == post_deps:
                        add_successors = False
                    else:
                        post_deps = tuple(sorted(set(state[block]).intersection(post_deps), key=lambda x: x.id))
                        state[block] = post_deps
                else:
                    state[block] = post_deps
                if add_successors:
                    for successor in block.successors:
                        if successor in body and successor != loop.head:
                            worklist.append((successor, post_deps))
                            
            for block, dependencies in state.items():
                if len(dependencies) > 0: # not strictly necessary but saves computation.
                    matching_dependencies = tuple(d for d in dependencies if d in strict_dependencies[block])
                    if len(matching_dependencies) > 0:
                        for d in matching_dependencies:
                            add_loopbreaking_edges(d, block)
    
    return loopbreaking_edges

###############################################  
#                                             #
#    Main function for inductive alignment    #
#                                             #
###############################################  
def _align_inductive(candidate_ir: Function, reference_ir: Function, injective: bool, control_dependence: bool, partial_loops: bool, alpha: float, verbose: bool = False):
    copbyid: Dict[(str, Optional[str]), List[SSAOperator]] = classify_operators_by_operantion(candidate_ir)
    ropbyid: Dict[(str, Optional[str]), List[SSAOperator]] = classify_operators_by_operantion(reference_ir)

    if injective:
        clones, ambiguities = find_optimal_clones(candidate_ir, reference_ir, copbyid, ropbyid, verbose)
    else:
        clones = []
        ambiguities = []

    if verbose:
        print("\n---- Final Clones ----")
        for c in clones:
            print(c)
        print("\n---- Final Ambiguities ----")
        for a in ambiguities:
            print(a)

    factq = Queue()
    ground_truth = []
    # Create parameter-based argument equivalence pairs.
    for c_param, r_param in zip(candidate_ir.parameters, reference_ir.parameters):
        equiv = Equivalence(c_param, r_param)
        factq.put(equiv)
        ground_truth.append(equiv)
    trueclause = TrueClause()
    factq.put(trueclause)
    ground_truth.append(trueclause)
    del trueclause

    fact_registry = FactRegistry(ground_truth, injective)

    if control_dependence:
        candidate_op_control_dependence = operator_level_control_dependence(candidate_ir)
        reference_op_control_dependence = operator_level_control_dependence(reference_ir)

        cand_loopbreaking_control = loopbreaking_control_edges(candidate_ir)
        ref_loopbreaking_control = loopbreaking_control_edges(reference_ir)

    cand_loopbreaking_phis = loopbreaking_phi_nodes(candidate_ir, control_dependence)
    ref_loopbreaking_phis = loopbreaking_phi_nodes(reference_ir, control_dependence)

    # Operators hash/compare based on their python IDs (memory addresses) so we can safely put 
    # both candidate and reference operators in the same set without worrying about overwriting
    # a candidate opeartor with a reference operator or vice versa.
    clone_operators: Set[SSAOperator] = set()
    cloneq2clause: Dict[Equivalence, CloneClause] = {}

    clone_clauses: List[CloneClause] = []

    # Must build the entire clone_operators set first before checking for membership in it.
    for clone in itertools.chain(clones, (c for ambiguity in ambiguities for c in ambiguity.clones)):
        for equivalence in clone:
            clone_operators.add(equivalence.left)
            clone_operators.add(equivalence.right)
        if isinstance(clone, ProxyCompositeCloneOperator):
            conclusion = CompositeCloneClause(clone.id, clone.equivalences, clone.ambiguities, clone.ambiguity_frontier)
        else:
            conclusion = CloneClause(clone.id, clone.equivalences)
        for equivalence in clone:
            cloneq2clause[equivalence] = conclusion
        clone_clauses.append(conclusion)

    # Treat clones as operators. Their operands are all of the external operands of the members of the clone
    # (that is, the operands of operators in the clone that don't aren't already inside the clone.)
    for clone in clone_clauses:
        this_clone_components: Set[Equivalence] = set(clone)

        if isinstance(clone, CompositeCloneClause):
            composite_ambiguity_equivalences: Set[Equivalence] = {eq for ambiguity in clone.ambiguities for c in ambiguity.clones for eq in c}
        else:
            composite_ambiguity_equivalences = set()

        new_implication_conditions: List[WeightedImplication] = []
        frontier_size: int = 0 # the number of operands to the clone (whether or not they are equivalent.)
        for equivalence in clone:
            argument_pairs = standardize_operands(equivalence.left, equivalence.right)
            if len(argument_pairs[0]) == 0:
                new_implication_conditions.append(TrueClause())
                frontier_size += 1
            else:
                for i, (c_arg, r_arg) in enumerate(zip(*argument_pairs)):
                    if equivalence.left.op == PHI_OP:
                        assert equivalence.right.op == PHI_OP, "Only two operators with identical operations may be considered equivalent!"
                        op_equiv = phi_operands_equivalent(c_arg, r_arg, 
                                                           equivalence.left in cand_loopbreaking_phis and i in cand_loopbreaking_phis[equivalence.left], 
                                                           equivalence.right in ref_loopbreaking_phis and i in ref_loopbreaking_phis[equivalence.right]
                                                          )
                    else:
                        op_equiv = operands_equivalent(c_arg, r_arg)
                    # - op_equiv not in cloneq2clause or cloneq2clause[op_equiv] != clone: the operands are in the clone itself, which we model as a single operator. This represents an implication inside that 
                    # operator, which is irrelevant.
                    # - op_equiv not in composite_ambiguity_equivalences: this represents an operator inside one of this operator's ambiguities. Using and ambiguous equivalence from an ambiguity in a composite 
                    # clone as a condition to prove that composite clone creates a cyclic dependency.
                    if op_equiv not in this_clone_components and op_equiv not in composite_ambiguity_equivalences:
                        if op_equiv is not None:
                            if op_equiv in cloneq2clause:
                                new_implication_conditions.append(cloneq2clause[op_equiv])
                            else:
                                new_implication_conditions.append(op_equiv)
                        frontier_size += 1
        
        # This is a temporary provision making sure the implementation of clone_conditions is correct.
        # Note: the code above has been updated, meaning that clone_conditions no longer matches the code above.
        # new_implication_conditions_prime, frontier_size_prime = clone_conditions(clone, [cloneq2clause, composite_ambiguity_equivalences], cand_loopbreaking_phis, ref_loopbreaking_phis)
        # assert new_implication_conditions == new_implication_conditions_prime and frontier_size == frontier_size_prime, "Implementation of clone_conditions is incorrect."
        
        for condition in new_implication_conditions:
            fact_registry.add_implication(WeightedImplication(condition, clone, 1 / frontier_size))
    
    # Keep track of the edges assumed true to break cycles of dependencies. If partial_loops is False, these assumptions will be removed later.
    if not partial_loops:
        loopbreaking_edges: list[WeightedImplication] = []

    # Build the implications for building the isomorphic subgraph based on operators alone.
    opids = set(itertools.chain(copbyid.keys(), ropbyid.keys()))
    for opid in opids:
        if opid in copbyid and opid in ropbyid:
            for cand_op in copbyid[opid]:
                for ref_op in ropbyid[opid]:
                    # These operators are handled (and possibly disambiguated) by clones. In either
                    # case, adding implications related to them here may result in unexpected contradictions
                    # during inference. By skipping them with continue, we avoid this problem.
                    if cand_op in clone_operators and ref_op in clone_operators:
                        continue
                    conclusion = Equivalence(cand_op, ref_op)
                    candidate_arguments, reference_arguments = standardize_operands(cand_op, ref_op)

                    # Equally weight all constraints: control flow and data flow.
                    num_constraints = 1 if len(candidate_arguments) == 0 else len(candidate_arguments)
                    if control_dependence:
                        max_dependencies = max(len(candidate_op_control_dependence[cand_op]), len(reference_op_control_dependence[ref_op]))
                        num_constraints += max_dependencies
                    weight = 1 / num_constraints

                    if not partial_loops:
                        num_loopbreaking_edges = 0 # only used when partial_loops if False.

                    new_implications = []
                    if len(candidate_arguments) == 0:
                        new_implications.append(WeightedImplication(TrueClause(), conclusion, weight))
                    else:
                        for i, (c_arg, r_arg) in enumerate(zip(candidate_arguments, reference_arguments)):
                            if opid[0] == PHI_OP:
                                cand_idx_is_loopbreaking = cand_op in cand_loopbreaking_phis and i in cand_loopbreaking_phis[cand_op]
                                ref_idx_is_loopbreaking = ref_op in ref_loopbreaking_phis and i in ref_loopbreaking_phis[ref_op]
                                if partial_loops:
                                    op_equiv = phi_operands_equivalent(c_arg, r_arg, cand_idx_is_loopbreaking, ref_idx_is_loopbreaking)
                                else:
                                    op_equiv = operands_equivalent(c_arg, r_arg)
                                    num_loopbreaking_edges += (cand_idx_is_loopbreaking and ref_idx_is_loopbreaking)
                                    # These assertions should be true because in loopbreaking_phi_nodes, we only consider the arguments of phi nodes
                                    # when they are SSAOperators.
                                    assert not cand_idx_is_loopbreaking or isinstance(c_arg, SSAOperator)
                                    assert not ref_idx_is_loopbreaking or isinstance(r_arg, SSAOperator)
                            else:
                                op_equiv = operands_equivalent(c_arg, r_arg)
                            if op_equiv is not None:
                                if isinstance(op_equiv, Equivalence) and op_equiv in cloneq2clause:
                                    op_equiv = cloneq2clause[op_equiv]
                                new_implications.append(WeightedImplication(op_equiv, conclusion, weight))
                            # Do nothing in the false case; these operands get no weight.
                    # Add the control flow constraints
                    if control_dependence:
                        # Control dependencies are ordered, so we simply zip them together.
                        for (cand_dep, cand_branch_idx), (ref_dep, ref_branch_idx) in zip(candidate_op_control_dependence[cand_op], reference_op_control_dependence[ref_op]):
                            # No sense in adding an if == loop edge, because that'll never be true under codealign's assumptions.
                            # Also, check that the candidate and ref have the same branch idx (i.e. both correspond to either the true or false branch.)
                            if cand_dep.op == ref_dep.op and cand_branch_idx == ref_branch_idx:
                                if cand_dep in cand_loopbreaking_control and cand_op in cand_loopbreaking_control[cand_dep] and \
                                ref_dep in ref_loopbreaking_control and ref_op in ref_loopbreaking_control[ref_dep]:
                                    if partial_loops:
                                        new_implications.append(WeightedImplication(TrueClause(), conclusion, weight=weight))
                                    else:
                                        # Add the same edge as in the non-loopbreaking case
                                        new_implications.append(WeightedImplication(Equivalence(cand_dep, ref_dep), conclusion, weight=weight))
                                        num_loopbreaking_edges += 1
                                else:
                                    new_implications.append(WeightedImplication(Equivalence(cand_dep, ref_dep), conclusion, weight=weight))
                    if math.isclose(max_provable_weight := len(new_implications) / num_constraints, alpha) or max_provable_weight > alpha:
                        if not partial_loops and num_loopbreaking_edges > 0:
                            consolidated_loopbreaking_edge = WeightedImplication(TrueClause(), conclusion, num_loopbreaking_edges * weight)
                            new_implications.append(consolidated_loopbreaking_edge)
                            loopbreaking_edges.append(consolidated_loopbreaking_edge)
                        if isinstance(opid[1], FunctionPointer):
                            for impl in new_implications:
                                fact_registry.add_implication(WeightedImplication(operands_equivalent(cand_op.name, ref_op.name), impl, weight=1.0))
                        else:
                            for impl in new_implications:
                                fact_registry.add_implication(impl)

    if verbose:
        print("---- Initial State ----")
        print(fact_registry)
        print()
        print(fact_registry.graph)
        print()

    contradictions: List[Contradiction] = []
    contradicted: Set[Clause] = set()

    # Main control loop
    while not factq.empty():
        current_fact = factq.get()
        if verbose:
            print(f"---- Round: {current_fact} ----")
        if current_fact not in contradicted:
            for new_fact in fact_registry.satisfy_conditions(current_fact):
                if isinstance(new_fact, Clause):
                    factq.put(new_fact)
                else:
                    assert isinstance(new_fact, Contradiction)
                    if verbose:
                        print("-- Contradiction reached!")
                        print(new_fact)
                        print()
                    # Store this for processing later; prove as much as you can with the remaining premises.
                    # contradiction_manager.add_contradiction(new_fact)
                    contradictions.append(new_fact)
                    contradicted.update(new_fact.removed)
            if verbose:
                print(fact_registry.graph)
        
        if factq.empty():
            if len(contradictions) > 0:
                # print("*" * 150)
                # print("* Resolving Contradictions")
                # print("*" * 150)
                manager = ContradictionManager(contradictions, fact_registry, candidate_ir, reference_ir)
                for boundary_fact in manager.resolve():
                    factq.put(boundary_fact)

                contradictions: List[Contradiction] = []
                # reset contradicted after it is empty. Contradicted acts as a filter for facts that 
                # were proven via means that were later shown to be contradictory. After factq is empty,
                # however, no such contradictions exist in the queue. Some (though not all) of those facts
                # may actually end up being true; so it's important we empty contradicted here.
                contradicted = set()
            
            else:
                if verbose:
                    print("---- Assuming new facts ----")
                for assumed_fact in fact_registry.assert_most_likely_facts_true(alpha):
                    if verbose:
                        print(assumed_fact)
                    factq.put(assumed_fact)

    # If partial_loops=False, undo loops that haven't been proven equivalent.
    if not partial_loops:
        if verbose:
            print("-" * 20 + " Fact Graph Before Loop Testing " + "-" * 20)
            print(fact_registry.graph)
            print()

        for consolidated_loopbreaking_edge in loopbreaking_edges:
            if consolidated_loopbreaking_edge.conclusion in fact_registry.facts:
                fact_registry.revoke_implication(consolidated_loopbreaking_edge)
        
    if verbose:
        print("-" * 20 + " Final Fact Registry and Graph " + "-" * 20)
        print(fact_registry)
        print()
        print(fact_registry.graph)
        print()

    # Possible causes:
    # - if control_dependence is true and there are basic blocks each with two or more control dependences, then the total weights
    #   of all implications ending with operators in that block will be greater than 1.0. Too many of these were proven true.
    # - when partial_loops is false, implications corresponding to loops will have their weight added twice: once for loops and
    #   once for the 'loopbreaking' implications that bootstrap proofs of loops.
    # - perhaps some other bug.
    for fact, weight in fact_registry.fact2weight.items():
        assert weight < 1.0 or math.isclose(weight, 1.0), f"Fact {fact} has weight {weight} > 1.0 at end of inference which is invalid!"

    loopbreaking_phi_nodes.cache_clear()
    
    # Package results for returning.
    alignment_list = []
    aligned = set()
    for fact in fact_registry.flattened_facts():
        if isinstance(fact, TrueClause):
            continue
        assert isinstance(fact, Equivalence)
        alignment_list.append((fact.left, fact.right))
        aligned.add(fact.left)
        aligned.add(fact.right)
    
    for operator in (op for bb in candidate_ir for op in bb):
        if operator not in aligned:
            alignment_list.append((operator, None))
    for operator in (op for bb in reference_ir for op in bb):
        if operator not in aligned:
            alignment_list.append((None, operator))
    
    if injective:
        return InjectiveAlignment(alignment_list, candidate_ir, reference_ir)
    else:
        return RelationalAlignment(alignment_list, candidate_ir, reference_ir)
