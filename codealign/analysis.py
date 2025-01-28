"""Program analysis code necessary for code alignment
"""
import itertools
from queue import Queue
from typing import Callable, TypeVar, Iterable, List, Dict, Set, Tuple
import copy

from .ir import *


#### Generic dataflow analysis framework

T = TypeVar("T") # Lattice element T

def dataflow(function: Function, 
             transfer_fn: Callable[[List[T], BasicBlock], List[T]], # Note: the output List[T] MUST NOT alias the input List[T].
             meet: Callable[[T, T], T], # Correctness is not guaranteed if T meet treats T as a mutable object.
             forward: bool,
             start: List[T],
             top: T
            ) -> Dict[BasicBlock, List[T]]:
    assert len(start) == len(function.basic_blocks), "Starting dataflow state (entry in forward, exit in backward) must have one element for each basic block in the function."
    num_blocks = len(start)
    top_vector = [top] * num_blocks

    ordering = postorder_traversal(function.entry_block, [], set())
    if forward:
        ordering.reverse() # Use reverse postorder for forward analyses.
    worklist = Queue() # Better data structure than a list for a queue.
    for item in ordering:
        worklist.put(item)

    in_states: Dict[BasicBlock, List[T]] = {} # represents the INPUT states to each block (if forward) or the OUTPUT states (if backwards)
    out_states: Dict[BasicBlock, List[T]] = {} # represents the OUTPUT states to each block (if forward) or the INPUT states (if backwards)
    
    for basic_block in function.basic_blocks:
        if is_starting_block(basic_block, function, forward):
            out_states[basic_block] = transfer_fn(start, basic_block)
        else:
            out_states[basic_block] = transfer_fn(top_vector, basic_block)
    
    # Begin worklist loop
    while not worklist.empty():
        current = worklist.get()

        if is_starting_block(current, function, forward):
            block_in = start
        else:
            block_in = top_vector
            for predecessor in predecessors(current, forward):
                block_in = vector_meet(block_in, out_states[predecessor], meet) # Generates a new list.
        
        in_states[current] = block_in
        block_out = transfer_fn(block_in, current)

        if block_out != out_states[current]:
            for successor in successors(current, forward):
                worklist.put(successor)
        
        out_states[current] = block_out
    
    return in_states


#### Helper methods for dataflow analysis   
def is_starting_block(basic_block: BasicBlock, function: Function, forward: bool):
    if forward:
        return id(basic_block) == id(function.entry_block)
    else:
        return len(basic_block.successors) == 0

def vector_meet(l: List[T], r: List[T], meet: Callable[[T, T], T]):
    output = []
    for l_i, r_i, in zip(l, r):
        output.append(meet(l_i, r_i))
    return output

def successors(basic_block: BasicBlock, forward: bool) -> Iterable[BasicBlock]:
    if forward:
        return basic_block.successors
    else:
        return basic_block.predecessors

def predecessors(basic_block: BasicBlock, forward: bool) -> Iterable[BasicBlock]:
    if forward:
        return basic_block.predecessors
    else:
        return basic_block.successors

def postorder_traversal(basic_block: BasicBlock, ordering: List[BasicBlock], encountered: Set[BasicBlock]):
    # Ensure we don't record the same block twice and don't get stuck in a loop.
    if basic_block in encountered:
        return ordering
    encountered.add(basic_block)

    for successor in reversed(basic_block.successors):
        postorder_traversal(successor, ordering, encountered)
    
    ordering.append(basic_block)
    return ordering # The algorithm builds the postorder list via pass-by-reference, but returns the list here for the convenience of the original caller.




class Dominance:
    # Convention: True -> dominates, False -> does not dominate.
    def __init__(self, function: Function, forward=True):
        bb2idx = {}
        for i, basic_block in enumerate(function.basic_blocks):
            bb2idx[basic_block] = i
        
        def transfer(in_state: List[bool], bb: BasicBlock) -> List[bool]:
            out_state = in_state.copy()
            out_state[bb2idx[bb]] = True
            return out_state

        def meet(l: bool, r: bool) -> bool:
            return l and r
        
        self.strict_dominance_info: Dict[BasicBlock, List[bool]] = dataflow(
            function, transfer, meet, forward, [False] * len(function.basic_blocks), True
        )
        self.bb2idx = bb2idx
        self.function = function
        self.forward = forward
    
    def strictly_dominates(self, x: BasicBlock, y: BasicBlock) -> bool:
        """Returns true if x sdom y.
        Precondition: x and y are basic blocks in the function that correspond to this Dominance instance.
        """
        assert x in self.strict_dominance_info, "BasicBlock x must be in the function that corresponds to this Dominance instance!"
        assert y in self.strict_dominance_info, "BasicBlock y must be in the function that corresponds to this Dominance instance!"

        idx = self.bb2idx[x]
        return self.strict_dominance_info[y][idx]

    
    def dominates(self, x: BasicBlock, y: BasicBlock) -> bool:
        """returns true if x dom y
        """
        if id(x) == id(y):
            return True
        return self.strictly_dominates(x, y)
    
    def dominance_frontier(self, x: BasicBlock) -> List[BasicBlock]:
        """Compute the dominance frontier of x
        """
        frontier = []
        for y in self.function.basic_blocks:
            if not self.strictly_dominates(x, y) and any(map(lambda pred_y: self.dominates(x, pred_y), predecessors(y, self.forward))):
                frontier.append(y)
        
        return frontier
    

class DominatorTree:
    class Node:
        def __init__(self, block: BasicBlock):
            self.block: BasicBlock = block
            self.children: Set[DominatorTree.Node] = set()
        
        def __repr__(self):
            children_string = ", ".join([str(c.block.id) for c in self.children])
            return f"DominatorTree.Node({self.block.id}; children = {children_string})"
        
        def __hash__(self):
            return id(self.block)
        
        def __eq__(self, other):
            if not isinstance(other, DominatorTree.Node):
                return False
            return id(self.block) == id(other.block)
        
        def __iter__(self):
            """Iterate over the children of the DominatorTree Node.
            
            Ensures a consistent iteration order, unlike iterating directly from the set.
            """
            children = list(self.children)
            children.sort(key=lambda x: x.block.id)
            for child in children:
                yield child
    
    def __init__(self, function: Function, dominance: Dominance):
        self.function = function
        self.dominance = dominance
        self.node_count = 0
        self.visited = set()
        self.tree, undominated = self.make_tree(function.entry_block)
        assert len(undominated) == 0, "All blocks should be dominated by the entry node."
        del self.visited
        

    def make_tree(self, block: BasicBlock):
        # TODO: Make this work forward or backward (currently works for just forward).
        node = DominatorTree.Node(block)
        self.node_count += 1
        undominated = set() # This block does not dominate these basic blocks, but perhaps its parent might.
        for successor in block.successors:
            if successor not in self.visited:
                self.visited.add(successor)
                child_node, deeper_undominated = self.make_tree(successor)

                if self.dominance.dominates(block, successor):
                    node.children.add(child_node)
                    for subnode in deeper_undominated:
                        if self.dominance.dominates(block, subnode.block):
                            node.children.add(subnode)
                        else:
                            undominated.add(subnode)
                else:
                    undominated.add(child_node)
                    undominated.update(deeper_undominated)
        
        return node, undominated
    
    def __repr__(self):
        out = ""
        def build_repr(node):
            nonlocal out
            out += repr(node) + "\n"
            for child in node:
                build_repr(child)
        build_repr(self.tree)
        return out


###### Conversion to SSA Form ######
SSAProxyOperand = Union[VarOperator, Constant, Parameter, GlobalVariable]
class PhiNodeProxy:
    def __init__(self, variable: Variable, out_repr: Optional[str] = None):
        self.variable = variable
        self.out_repr = out_repr
        self.definitions: Set[SSAProxyOperand] = set()
        self.deflist: List[SSAProxyOperand] = []
        self.ref_count = 0 # Records how many times this PhiNodeProxy is used.
        self.created = False
        self.ssa_node = SSAOperator(PHI_OP, []) # Will initialize these when possible.
    
    def __len__(self):
        return len(self.definitions)
    
    def __hash__(self) -> int:
        return id(self)
    
    def __eq__(self, other):
        return id(self) == id(other)
    
    def add_var_definition(self, var_operator: SSAProxyOperand):
        if var_operator not in self.definitions:
            self.deflist.append(var_operator)
            self.definitions.add(var_operator)
    
    def createPhiNode(self, var2ssa: Dict[VarOperator, SSAOperator]):
        if self.created:
            return self.ssa_node

        phi_operands = []
        for definition in self.deflist:
            if isinstance(definition, VarOperator):
                assert definition in var2ssa
                phi_operands.append(var2ssa[definition])
            else:
                phi_operands.append(definition) # definition is a parameter, global variable, constant, etc.

        self.created = True
        self.ssa_node.out_repr = self.out_repr
        self.ssa_node.operands = phi_operands
        return self.ssa_node
    
    def __repr__(self):
        argstrings = []
        for arg in self.deflist:
            if isinstance(arg, PhiNodeProxy):
                argstrings.append(f"{arg.variable} = PhiNodeProxy(...)") # prevent infinate loop in printing
            else:
                argstrings.append(repr(arg))
        argstring = ", ".join(argstrings)
        return f"{self.variable} = PhiNodeProxy[{self.ref_count}]({argstring})"

###############################################
### Main function for the conversion to SSA ###
###############################################
def convert_to_ssa(function: Function, in_place=True) -> Function:
    """Convert a function to single-static-assignment form.

    :param function: the function to convert
    :param in_place: whether or not to modify the function or return a new function with the original unchanged. Currently,
    only in_place=True is supported because tree-sitter AST-nodes can't be copied. To call the function with in_place=False,
    first set all ast_node operator fields in the function to None.
    """
    if not in_place:
        function = copy.deepcopy(function)
    dominance = Dominance(function)

    # Information for placing phi nodes.
    orig: Dict[BasicBlock, Dict[Variable, VarOperator]] = {}
    defsites: Dict[Variable, Set[BasicBlock]] = {}

    # Find all of the variables and global variables in the function. Track global variables separately; they
    # are handled differently during initialization
    variables: List[Variable] = [] # includes parameters.
    global_variables: List[GlobalVariable] = []
    for basic_block in function.basic_blocks:
        for operator in basic_block:
            if isinstance(operator.result, GlobalVariable):
                global_variables.append(operator.result)
            elif operator.result is not None:
                variables.append(operator.result)
            for operand in operator.operands:
                if isinstance(operand, GlobalVariable):
                    global_variables.append(operand)
                elif isinstance(operand, Variable):
                    variables.append(operand)

    orig[function.entry_block] = {}
    # Define variables at the top of the entry block. For parameters and global variables,
    # this represents them being defined at the start of the function. For local variables,
    # this represents an implicit initialization at the start of the function to the special
    # <uninitialized> value. If a variable is not initialized along a path before it is used,
    # this uninitialized value will be incorporated into the phi node.
    for variable in itertools.chain(variables, global_variables):
        # Temporaries are generated as part of expressions and thus should always dominate their uses.
        # This makes the output representations significantly better, because it prevents out_repr
        # numbers from being allocated to PhiNodeProxies that don't need them.
        if variable.is_temporary:
            continue
        if variable in orig[function.entry_block]:
            assert variable in defsites
        else:
            assert variable not in defsites
            if isinstance(variable, Parameter) or isinstance(variable, GlobalVariable):
                orig[function.entry_block][variable] = variable
            else:
                orig[function.entry_block][variable] = Uninitialized()
            defsites[variable] = {function.entry_block}

    # Collect the definitions of each variable in each basic block, indexed both by basic block and by variable.
    # Store only the last definition of each variable in each basid block (done by overwriting each successive 
    # entry using a python dictionary indexed by variable).
    for basic_block in function.basic_blocks:
        if basic_block not in orig:
            orig[basic_block] = {}
        for operator in basic_block:
            assert isinstance(operator, VarOperator)
            if operator.result is not None:
                orig[basic_block][operator.result] = operator
                if operator.result not in defsites:
                    defsites[operator.result] = set()
                defsites[operator.result].add(basic_block)
    
    # Place the phi nodes
    phi_nodes_to_add: Dict[Variable, Dict[BasicBlock, PhiNodeProxy]] = {}
    for variable, var_defsites in defsites.items():
        # It is unsafe to modify a data structure while iterating over it.
        # Thus we put the items of var_defsites in a queue for safe iteration.
        worklist: Queue[BasicBlock] = Queue()
        for site in var_defsites:
            worklist.put(site)

        # Stores where the phi nodes should be for a given variable (along with a precursor object of the phi node itself.)
        # This is stored in phi_nodes_to_add.
        var_phi_locations: Dict[BasicBlock, PhiNodeProxy] = {}
        
        while not worklist.empty():
            current: BasicBlock = worklist.get()
            for frontier_block in dominance.dominance_frontier(current):
                if frontier_block not in var_phi_locations:
                    phi_node_proxy = PhiNodeProxy(variable)
                    var_phi_locations[frontier_block] = phi_node_proxy
                    if variable not in orig[frontier_block]:
                        worklist.put(frontier_block)
                        # Because this block had no other definitions of this variable (variable not in orig[frontier_block]), 
                        # this is now the downward exposed definition of this variable in this block.
                        orig[frontier_block][variable] = phi_node_proxy
        
        phi_nodes_to_add[variable] = var_phi_locations

    # Determine what definitions need to go in each Phi node.
    for variable, phi_nodes_by_block in phi_nodes_to_add.items():
        for basic_block, phi_node in phi_nodes_by_block.items():
            # explore the predecessors of this block and collect definitions that reach this point.
            explored = set() # keep track of where we've visited to avoid repeatedly following a loop in the CFG.
            exploring: Queue[BasicBlock] = Queue()
            for predecessor in basic_block.predecessors:
                exploring.put(predecessor)
            
            while not exploring.empty():
                current_block = exploring.get()
                explored.add(current_block)
                if variable in orig[current_block]: # We encountered a definition of this variable in current_block.
                    # Add this definition to the phi node.
                    phi_node.add_var_definition(orig[current_block][variable])
                else: # We didn't encounter a definition here
                    for predecessor in current_block.predecessors:
                        if predecessor not in explored:
                            exploring.put(predecessor)

    # For each variable, stores where it was last defined. Used for assigning arguments when converting to SSA.
    var2def: Dict[Variable, List[SSAOperand]] = {}

    for var in variables:
        var2def[var] = [Uninitialized()] # This list is used as a stack
    for parameter in function.parameters:
        var2def[parameter] = [parameter]
    for global_variable in global_variables:
        var2def[global_variable] = [global_variable]

    # Keep track of the var operator from which each ssa operator was generated.
    var2ssa: Dict[VarOperator, SSAOperator] = {}
    
    # To make SSAOperators' printed representations more readable.
    result_idx = 0
    def next_repr():
        nonlocal result_idx
        out_repr = f"%{result_idx}"
        result_idx += 1
        return out_repr

    def rename(dtree_node: DominatorTree.Node, var2value: Dict[Variable, List[SSAOperand]]):
        """Now that phi-nodes have been inserted, "rename" variables such that each definition is unique
        and each definition dominates its uses. Because each definition is unique, we can represent each argument
        as an IR object reference to the operation that computes it.
        """
        nonlocal dominance
        nonlocal phi_nodes_to_add
        nonlocal var2ssa
        new_operators: List[SSAOperator] = []

        # Ensure that phi nodes at the start of this block can be used by its other operators as arguments.
        for variable, phi_nodes_by_block in phi_nodes_to_add.items():
            if dtree_node.block in phi_nodes_by_block:
                phi_node_proxy = phi_nodes_by_block[dtree_node.block]
                if len(phi_node_proxy) > 1: # Phi nodes with one argument are irrelevant.
                    assert phi_node_proxy.out_repr is None
                    phi_node_proxy.out_repr = next_repr()
                    var2value[variable].append(phi_node_proxy)

        # Convert each operator to SSA form, and do associated bookkeeping.
        for var_operator in dtree_node.block:
            out_repr = None if var_operator.result is None else next_repr()
            ssa_operator = operator_var_to_ssa(var_operator, var2value, out_repr) # operator_var_to_ssa updates var2value.
            new_operators.append(ssa_operator)
            var2ssa[var_operator] = ssa_operator # Used to resolve phi nodes' arguments later.
        
        for child in dtree_node:
            rename(child, copy_var2value(var2value))
        
        dtree_node.block.operators = new_operators    

    rename(DominatorTree(function, dominance).tree, var2def)

    # Not all phi nodes are necessary. Phi nodes that are never used by any downstream
    # operators are unnecessary and can be removed. (Phi nodes with only one argument)
    # are also unnecessary, but that isn't handled here). We use the ref_count attribute
    # of PhiNodeProxy to determine how many times this phi node is used by downstream operations.
    # Some phi nodes are only used by other phi nodes. In the worklist algorithm below, we
    # propagate usage information backwards down the chains of phi nodes to ensure that they all
    # have the correct ref_counts. Using a worklist algorithm allows us to do this without
    # concern for the order that we process the PhiNodeProxies in.
    ref_chain_worklist: Queue[PhiNodeProxy] = Queue()
    for phi_node_proxy in (phi for _, phi_nodes_by_block in phi_nodes_to_add.items() for _, phi in phi_nodes_by_block.items()):
        ref_chain_worklist.put(phi_node_proxy)
    ref_count_info_propagated: Set[PhiNodeProxy] = set()
    while not ref_chain_worklist.empty():
        phi_node_proxy = ref_chain_worklist.get()
        # Prevent infinate loops through the graph
        if phi_node_proxy in ref_count_info_propagated:
            continue
        # Phi node proxies of lenth less than 1 will be discarded; we don't want to increase reference
        # counts based on these. Additionally, we only want to increase reference counts for this phi
        # node's arguments if this phi node is itself used.
        if len(phi_node_proxy) > 1 and phi_node_proxy.ref_count > 0:
            ref_count_info_propagated.add(phi_node_proxy)
            for phi_operand in phi_node_proxy.deflist:
                if isinstance(phi_operand, PhiNodeProxy):
                    phi_operand.ref_count += 1
                    # This phi node may have not been considered before because its ref count was not high
                    # enough. Now that it definitely is, add it back to the worklist to be considered again
                    # if necessary.
                    ref_chain_worklist.put(phi_operand)

    # Filter out the irrelevant phi nodes (those that are unused or those that have only
    # one argument) and organize them by basic block. Create the phi nodes. (Note: the phi
    # nodes have technically already been created, but without their arguments. This is done
    # so that operators can point to the phi nodes in those operators' arguments. Calling
    # createPhiNode(var2ssa) here converts that shell of a phi node to a full phi node with 
    # arguments)
    bb2phi: Dict[BasicBlock, List[SSAOperator]] = {}
    for variable, phi_nodes_by_block in phi_nodes_to_add.items():
        for basic_block, phi_node_proxy in phi_nodes_by_block.items():
            if len(phi_node_proxy) > 1 and phi_node_proxy.ref_count > 0:
                if basic_block not in bb2phi:
                    bb2phi[basic_block] = []
                created = phi_node_proxy.createPhiNode(var2ssa)
                for i in range(len(created.operands)):
                    if isinstance(created.operands[i], PhiNodeProxy):
                        operand_phi_proxy = created.operands[i]
                        # Ensure that this phi node will be converted later.
                        assert len(operand_phi_proxy) > 1 and operand_phi_proxy.ref_count > 0, \
                            f"Attempting to create an invalid phi node for {variable} at basic_block {basic_block.id}: {phi_node_proxy}"
                        created.operands[i] = operand_phi_proxy.createPhiNode(var2ssa)
                bb2phi[basic_block].append(created)
    
    # Now that the SSA has been built, add the relevant phi-nodes to their basic blocks.
    for basic_block in function.basic_blocks:
        if basic_block in bb2phi: # This will only be true if there's at least one phi node for that block
            basic_block.operators = bb2phi[basic_block] + basic_block.operators

    # Conversion to SSA complete. Return the function.
    return function



def copy_var2value(var2value: Dict[Variable, List[SSAOperand]]):
    """This is a copy that's in between a shallow and deep copy.
    We want to copy the enclosing dictionary and the lists of the values
    but not the keys or the elements of the value lists.
    """
    new_var2value = {}
    for key, value in var2value.items():
        new_var2value[key] = value[:] # Whole-list list slice shallow-copies the whole list.

    return new_var2value


def operator_var_to_ssa(var_operator: VarOperator, var2value: Dict[Variable, List[SSAOperand]], out_repr=Optional[str]):
    ssa_operands = []
    for var_op in var_operator.operands:
        ssa_operands.append(get_ssa_value(var_op, var2value))
    if isinstance(var_operator, FunctionVarOperator):
        if var_operator.kwargs is not None:
            ssa_kwargs = {}
            for argkey, argvalue in var_operator.kwargs.items():
                ssa_kwargs[argkey] = get_ssa_value(argvalue, var2value)
        else:
            ssa_kwargs = None
        
        function_name = get_ssa_value(var_operator.name, var2value) if isinstance(var_operator.name, Variable) else var_operator.name
        ssa_operator = FunctionSSAOperator(function_name, ssa_operands, kwargs=ssa_kwargs, out_repr=out_repr, ast_node=var_operator.ast_node, var_operator=var_operator)
    else:
        assert(var_operator.op != FUNCTION_CALL_OP)
        ssa_operator = SSAOperator(var_operator.op, ssa_operands, out_repr=out_repr, ast_node=var_operator.ast_node, var_operator=var_operator)
    
    # Update the current value of the variable.
    if var_operator.result is not None:
        var2value[var_operator.result].append(ssa_operator)
    return ssa_operator

def get_ssa_value(var_operand: VarOperand, var2value: Dict[Variable, List[SSAOperand]]):
    # Could be a constant, field name, or type for a typecast
    if not isinstance(var_operand, Variable):
        return var_operand
    if (isinstance(var_operand, Parameter) or isinstance(var_operand, GlobalVariable)) and not var_operand in var2value:
        return var_operand
    value = var2value[var_operand][-1]
    if isinstance(value, PhiNodeProxy):
        value.ref_count += 1
        return value.ssa_node
    return value

def copy_propagation(function: Function):
    """precondition: function is in SSA form

    Modifies the function in place, eliminating all copy operations, replacing them with the copy op's argument.
    """

    val2use: Dict[SSAOperator, List[SSAOperator]] = {} # track where each copy op is used.
    copy_ops: Dict[SSAOperator, BasicBlock] = {} # track where each copy op is located so we can easily delete it from that block later.

    # Find all copy ops. Find where each operator is used.
    for basic_block in function.basic_blocks:
        for operator in basic_block:
            for operand in operator.operands:
                if isinstance(operand, SSAOperator) and operand.op == COPY_OP: # could also be constants, parameters, etc.
                    if operand not in val2use:
                        val2use[operand] = []
                    val2use[operand].append(operator)
            
            if operator.op == COPY_OP:
                assert operator not in copy_ops
                copy_ops[operator] = basic_block
                if operator not in val2use:
                    val2use[operator] = []

    for copy_op, defblock in copy_ops.items():
        assert len(copy_op.operands) == 1
        for use in val2use[copy_op]:
            # use is an SSAOperator
            for i in range(len(use.operands)):
                if use.operands[i] == copy_op:
                    use.operands[i] = copy_op.operands[0]
        
        defblock.operators.remove(copy_op)

### Control dependence
def control_dependence(function: Function) -> Dict[BasicBlock, List[BasicBlock]]:
    dependence: Dict[BasicBlock, List[BasicBlock]] = {}
    dominance = Dominance(function, forward=False)
    for basic_block in function.basic_blocks:
        dependence[basic_block] = dominance.dominance_frontier(basic_block)
    return dependence

def control_equivalence_classes(function: Function) -> Dict[Tuple[BasicBlock,...], List[BasicBlock]]:
    classes: Dict[Tuple[BasicBlock,...], List[BasicBlock]] = {}
    for basic_block, dependencies in control_dependence(function).items():
        # Sort to ensure that the dependencies are in a consistent order. What that order is doesn't matter too much so long as it is consistent.
        # A tuple is requried because it is immutable and can therefore is hashable and suitable for use as a key in a dictionary.
        immutable_dependencies = tuple(sorted(dependencies, key=lambda x: x.id))
        if immutable_dependencies not in classes:
            classes[immutable_dependencies] = []
        classes[immutable_dependencies].append(basic_block)
    # Ensure a consistent output ordering for easier testing and debugging.
    for _, equivalent in classes.items():
        equivalent.sort(key=lambda x: x.id)
    return classes

### Loops
class Loop:
    def __init__(self, head: BasicBlock, body: List[BasicBlock], back_edge: Tuple[BasicBlock, BasicBlock]):
        self.head = head
        self.body = body
        self.back_edge = back_edge

def find_loops(function: Function) -> List[Loop]:
    """Find all loops in the given function.
    """
    dominance = Dominance(function)
    back_edges: List[Tuple[BasicBlock, BasicBlock]] = []
    visited: Set[BasicBlock] = set()

    def find_back_edges(block: BasicBlock):
        if block in visited:
            return
        visited.add(block)
        for successor in block.successors:
            if successor in visited and dominance.dominates(successor, block):
                # back edge found
                back_edges.append((block, successor))
            else:
                find_back_edges(successor)
    
    find_back_edges(function.entry_block)

    # Find a loop for each back edge.
    def find_loop(current: BasicBlock, head: BasicBlock, contents: Set[BasicBlock]):
        if current == head:
            return
        contents.add(current)

        for predecessor in current.predecessors:
            if predecessor not in contents:
                find_loop(predecessor, head, contents)

    loops = []
    for back_edge in back_edges:
        tail, head = back_edge
        contents = set()
        find_loop(tail, head, contents)
        contents.add(head)
        loops.append(Loop(
            head=head,
            body=contents,
            back_edge=back_edge
        ))
    
    return loops
