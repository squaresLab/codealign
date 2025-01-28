"""
"""

import unittest

from codealign.ir import *



class TestIRGeneration(unittest.TestCase):    
    def assertContentsEqual(self, basic_block: BasicBlock, comparison: List[VarOperator]):
        """Determine if a basic block (in var form) has the same contents as the specified list.

        The == magic methods for many of the object type involved work based off those objects' 
        python IDs/memory locations, so that different instances of objects with the same contents
        are not considered equal. This is important for the correct interation of the objects with
        dictionaries in the algorithms included in this package; however, it necessitates a separate
        function (this one) for comparing based on content.
        
        This method ignores the subtleties of variable scoping.

        Consequences of ignoring variable scoping: some solutions which are actually not equivalent
        are considered equivalent by this method. If there are two different variables with the same
        name at different scopes in the original textual code but the basic_block operators treat this
        as one variable, this method will not detect this error.
        """
        assert len(basic_block.operators) == len(comparison)

        def operands_equivalent(bb_operand, comp_operand):
            assert isinstance(bb_operand, VarOperand)
            assert isinstance(comp_operand, VarOperand)
            assert type(bb_operand) == type(comp_operand), f"{bb_operand}: {type(bb_operand)} != {comp_operand}: {type(comp_operand)}"
            
            if isinstance(bb_operand, Variable):
                assert bb_operand.name == comp_operand.name
            else:
                assert isinstance(bb_operand, Constant)
                assert bb_operand.value == comp_operand.value, f"{bb_operand.value} != {comp_operand.value}"
        
        for bb_operator, comp_operator in zip(basic_block.operators, comparison):
            # Check that types are valid
            assert isinstance(bb_operator, VarOperator)
            assert isinstance(comp_operator, VarOperator)
            assert isinstance(bb_operator.op, str)
            assert isinstance(comp_operator.op, str)
            assert bb_operator.result is None or isinstance(bb_operator.result, Variable) # Can be none for control-flow operations like if and return.
            assert comp_operator.result is None or isinstance(comp_operator.result, Variable) # Can be none for control-flow operations like if and return.
            
            assert bb_operator.op == comp_operator.op
            if bb_operator.op == FUNCTION_CALL_OP:
                assert isinstance(bb_operator, FunctionVarOperator)
                assert isinstance(comp_operator, FunctionVarOperator)
                bb_operator.name == comp_operator.name

                if bb_operator.kwargs is not None:
                    assert comp_operator is not None
                    assert len(bb_operator.kwargs) == len(comp_operator.kwargs)

                    for keyword, argument in bb_operator.kwargs.items():
                        assert keyword in comp_operator.kwargs
                        operands_equivalent(argument, comp_operator.kwargs[keyword])
            else:
                assert not isinstance(bb_operator, FunctionVarOperator)
                assert not isinstance(comp_operator, FunctionVarOperator)
            
            assert type(bb_operator.result) == type(comp_operator.result)  
            if bb_operator.result is not None:
                assert bb_operator.result.name == comp_operator.result.name, f"{bb_operator.result.name} != {comp_operator.result.name}"
            
            assert len(bb_operator.operands) == len(comp_operator.operands)
            for bb_operand, comp_operand in zip(bb_operator.operands, comp_operator.operands):
                operands_equivalent(bb_operand, comp_operand)