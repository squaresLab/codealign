import unittest
from typing import Set

from codealign.lang.c import parse
from codealign.analysis import convert_to_ssa, copy_propagation
from codealign.ir import *


class TestPhiNodes(unittest.TestCase):
    def parse(self, code: str) -> Function:
        return parse(bytes(code, "utf8"))[0]

    def test_loop(self):
        code = """
        int foo(int a, int b) {
            a = a + 1;
            do_call(a);
            my_thing(a, b);
            while (a < b){
                a = a + b;
            }
            a = a * 8;
            return a;  
        }
        """
        ssa = convert_to_ssa(self.parse(code))

        assert len(ssa.basic_blocks) == 4
        assert ssa.basic_blocks[0].operators[0].op != PHI_OP
        assert ssa.basic_blocks[1].operators[0].op == PHI_OP
        assert ssa.basic_blocks[1].operators[0].operands[0].op == "+" and ssa.basic_blocks[1].operators[0].operands[1].op == "+"
        assert ssa.basic_blocks[2].operators[0].op != PHI_OP
        assert ssa.basic_blocks[3].operators[0].op != PHI_OP
    
    def test_if_else(self):
        code = """
        int bar(int a) {
            if (a) {
                x = fn(a);
            } else {
                x = fn(1-a);
                x = x + 1;
            }
            return x;
        }
        """
        ssa = convert_to_ssa(self.parse(code))

        assert len(ssa.basic_blocks) == 4
        assert ssa.basic_blocks[0].operators[0].op != PHI_OP
        assert ssa.basic_blocks[1].operators[0].op != PHI_OP
        assert ssa.basic_blocks[2].operators[0].op != PHI_OP
        assert ssa.basic_blocks[3].operators[0].op == PHI_OP
        assert (ssa.basic_blocks[3].operators[0].operands[0].op == "+" and ssa.basic_blocks[3].operators[0].operands[1].name == "fn") or \
               (ssa.basic_blocks[3].operators[0].operands[0].name == "fn" and ssa.basic_blocks[3].operators[0].operands[1].op == "+")
        
    def test_if_while_if(self):
        code = """
            int ifwhileif(int a, int b, int c) {
                int g = 0;
                if (a) {
                    printf("starting.");
                    while (a) {
                        if (b > 0) {
                        c = c + 1;
                        }
                        g += c;
                    }
                    printf("done.");
                } else {
                    printf("Can't loop.");
                }
            return c;
        """
        ssa = convert_to_ssa(self.parse(code))

        assert len(ssa.basic_blocks) == 9
        assert ssa.basic_blocks[0].operators[0].op != PHI_OP
        assert ssa.basic_blocks[1].operators[0].op != PHI_OP

        assert ssa.basic_blocks[2].operators[0].op == PHI_OP
        assert (ssa.basic_blocks[2].operators[0].operands[0].op == "+" and ssa.basic_blocks[2].operators[0].operands[1].op == COPY_OP) or \
               (ssa.basic_blocks[2].operators[0].operands[0].op == COPY_OP and ssa.basic_blocks[2].operators[0].operands[1].op == "+")
        
        assert ssa.basic_blocks[2].operators[1].op == PHI_OP
        assert (isinstance(ssa.basic_blocks[2].operators[1].operands[0], Parameter) and isinstance(ssa.basic_blocks[2].operators[1].operands[1], SSAOperator)) or \
               (isinstance(ssa.basic_blocks[2].operators[1].operands[0], SSAOperator) and isinstance(ssa.basic_blocks[2].operators[1].operands[1], Parameter))

        assert ssa.basic_blocks[3].operators[0].op != PHI_OP
        assert ssa.basic_blocks[4].operators[0].op != PHI_OP

        assert ssa.basic_blocks[5].operators[0].op == PHI_OP
        assert (ssa.basic_blocks[5].operators[0].operands[0].op == PHI_OP and ssa.basic_blocks[5].operators[0].operands[1].op == "+") or \
               (ssa.basic_blocks[5].operators[0].operands[0].op == "+" and ssa.basic_blocks[5].operators[0].operands[1].op == PHI_OP)

        assert ssa.basic_blocks[6].operators[0].op != PHI_OP
        assert ssa.basic_blocks[7].operators[0].op != PHI_OP
        assert ssa.basic_blocks[8].operators[0].op == PHI_OP
        assert (isinstance(ssa.basic_blocks[8].operators[0].operands[0], Parameter) and isinstance(ssa.basic_blocks[8].operators[0].operands[1], SSAOperator)) or \
               (isinstance(ssa.basic_blocks[8].operators[0].operands[0], SSAOperator) and isinstance(ssa.basic_blocks[8].operators[0].operands[1], Parameter))
    
    def test_uninitialized(self):
        code = """
        int event_wait(int a1){
            char v2;
            int v4;
            do {
                v4 = 0;
                v4 = event_translate(a1, &v2);
            }
            while (v4);
            return 1;
        }
        """

        ssa = convert_to_ssa(self.parse(code))

        assert len(ssa.basic_blocks) == 4
        assert len(ssa.basic_blocks[0].operators) == 0
        assert ssa.basic_blocks[2].operators[1].operands[0] == Uninitialized()
        assert ssa.basic_blocks[1].operators[0].op != PHI_OP
        assert ssa.basic_blocks[2].operators[0].op != PHI_OP
        assert ssa.basic_blocks[3].operators[0].op != PHI_OP

    def test_uninitialized_path(self):
        code = """
        int foo(int x) {
            int y;
            if (x) {
                y = myfun();
            }
            return y;
        }
        """

        ssa = convert_to_ssa(self.parse(code))

        assert len(ssa.basic_blocks) == 3
        assert ssa.basic_blocks[0].operators[0].op != PHI_OP
        assert ssa.basic_blocks[1].operators[0].op != PHI_OP
        assert ssa.basic_blocks[2].operators[0].op == PHI_OP
        assert isinstance(ssa.basic_blocks[2].operators[0].operands[0], Uninitialized) and isinstance(ssa.basic_blocks[2].operators[0].operands[1], FunctionSSAOperator)
    
class TestCopyPropagation(unittest.TestCase):
    def parse(self, code: str) -> Function:
        return parse(bytes(code, "utf8"))[0]

    def assertSSAEqual(self, generated: Function, reference: List[List[SSAOperator]]):
        """Compare SSA form of operators. Does not explicitly consider control flow.
        """
        explored: Set[SSAOperator] = set()

        def assertOpsEqual(gen: SSAOperator, ref: SSAOperator):
            nonlocal explored

            if gen in explored:
                return # prevent infinate loops in cycles of phi nodes

            assert gen.op == ref.op
            if gen.op == FUNCTION_CALL_OP:
                assert isinstance(gen, FunctionSSAOperator)
                assert isinstance(ref, FunctionSSAOperator)
                assert gen.name == ref.name
            explored.add(gen)
            
            assert len(gen.operands) == len(ref.operands)
            for goperand, roperand in zip(gen.operands, ref.operands):
                if isinstance(goperand, SSAOperator):
                    assert isinstance(roperand, SSAOperator)
                    assertOpsEqual(goperand, roperand) 
                elif isinstance(goperand, Parameter):
                    assert isinstance(roperand, Parameter)
                    assert goperand.name == roperand.name
                elif isinstance(goperand, GlobalVariable):
                    assert isinstance(roperand, GlobalVariable)
                    assert goperand.name == roperand.name
                elif isinstance(goperand, Constant):
                    assert isinstance(roperand, Constant)
                    assert goperand.value == roperand.value
                else:
                    assert isinstance(roperand, Uninitialized)
                    assert isinstance(goperand, Uninitialized)

        for gen_block, ref_operators in zip(generated.basic_blocks, reference):
            assert len(gen_block.operators) == len(ref_operators)
            for gen_op, ref_op in zip(gen_block.operators, ref_operators):
                assertOpsEqual(gen_op, ref_op)
    
    def test_increment(self):
        code = """
        void foo() {
            int i = 0;
            i++;
        }
        """

        ssa = convert_to_ssa(self.parse(code))
        copy_propagation(ssa)

        reference = [
            SSAOperator("+", [NumberConstant("0"), NumberConstant("1")])
        ]

        self.assertSSAEqual(ssa, [reference])

    def test_for_loop(self):
        code = """
        void foo(int * arr, int len) {
            for (int i = 0; i < len; i++) {
                arr[i] = -1;
            }
        }
        """

        ssa = convert_to_ssa(self.parse(code))
        copy_propagation(ssa)

        iphi = SSAOperator(PHI_OP, [NumberConstant("0")])
        comparison = SSAOperator("<", [iphi, Parameter("len")])
        loop = SSAOperator(LOOP_OP, [comparison])
        condition_block = [iphi, comparison, loop]

        increment = SSAOperator("+", [iphi, NumberConstant("1")])
        iphi.operands.append(increment)
        increment_block = [increment]

        array_access = SSAOperator(SUBSCRIPT_OP, [Parameter("arr"), iphi])
        array_store = SSAOperator(STORE_OP, [array_access, NumberConstant("-1")])
        body_block = [array_access, array_store]

        self.assertSSAEqual(ssa, [[], condition_block, increment_block, body_block])
    
    def test_copy_chain(self):
        code = """
        int foo(int x) {
            int y = x;
            int z = y;
            int w = z;
            if (x) {
                w = w + 5;
            } else {
                w = -1;
            }
            return w;
        }
        """

        ssa = convert_to_ssa(self.parse(code))
        copy_propagation(ssa)

        if_op = SSAOperator(IF_OP, [Parameter("x")])
        if_condition = [if_op]

        update = SSAOperator("+", [Parameter("x"), NumberConstant("5")])
        if_body = [update]

        phi = SSAOperator(PHI_OP, [update, NumberConstant("-1")])
        return_op = SSAOperator(RETURN_OP, [phi])
        post_if = [phi, return_op]

        self.assertSSAEqual(ssa, [if_condition, if_body, [], post_if])