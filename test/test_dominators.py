"""Test the dominators implementation and in turn the generic dataflow analysis framework it depends on.
"""

import unittest

from codealign.lang.c import parse
from codealign.analysis import Dominance, control_dependence, control_equivalence_classes, find_loops
from codealign.ir import Function


class TestDominators(unittest.TestCase):
    def parse(self, code: str) -> Function:
        return parse(bytes(code, "utf8"))[0]

    def test_if(self):
        code = """
        int main() {
            int x = 4;
            if (y > 3) {
                x = 5;
            }
            return x;
        }
        """
        ir = self.parse(code)
        dominance = Dominance(ir)

        assert dominance.dominance_frontier(ir.basic_blocks[0]) == []
        assert dominance.dominance_frontier(ir.basic_blocks[1]) == [ir.basic_blocks[2]]
        assert dominance.dominance_frontier(ir.basic_blocks[2]) == []
    
    def test_nested_if_else(self):
        code = """
        char * foo(int x) {
            char * message;
            if (x) {
               if (x > 0) {
                   message = "x is positive."
               } else {
                   message = "x is negative."
               }
               printf("done!\\n");
            }
            return message;
        }
        """

        ir = self.parse(code)
        dominance = Dominance(ir)

        # Unfortunately, this test case (and others like it) are sensitive to the order in which 
        # basic blocks are returned by the parser, though in theory it shouldn't matter.
        entry_block = ir.entry_block
        inner_if_header = ir.basic_blocks[1]
        inner_if_true = ir.basic_blocks[2]
        inner_if_false = ir.basic_blocks[3]
        inner_post_if = ir.basic_blocks[4]
        exit_block = ir.basic_blocks[5]

        assert dominance.dominance_frontier(entry_block) == []
        assert dominance.dominance_frontier(inner_if_header) == [exit_block]
        assert dominance.dominance_frontier(inner_if_true) == [inner_post_if]
        assert dominance.dominance_frontier(inner_if_false) == [inner_post_if]
        assert dominance.dominance_frontier(inner_post_if) == [exit_block]
        assert dominance.dominance_frontier(exit_block) == []
    
    def test_while_loop(self):
        code = """
        int bar(int a, int b) {
            while (a < 8) {
               a++;
               b *= 7;
            }
            return b;
        }
        """

        ir = self.parse(code)
        dominance = Dominance(ir)
        pdominance = Dominance(ir, False)

        entry_block = ir.entry_block
        loop_header = ir.basic_blocks[1]
        loop_body = ir.basic_blocks[2]
        exit_block = ir.basic_blocks[3]

        # Dominance 
        assert dominance.dominance_frontier(entry_block) == []
        assert dominance.dominance_frontier(loop_header) == [loop_header]
        assert dominance.dominance_frontier(loop_body) == [loop_header]
        assert dominance.dominance_frontier(exit_block) == []

        # Postdominance
        assert pdominance.dominance_frontier(exit_block) == []
        assert pdominance.dominance_frontier(loop_body) == [loop_header]
        assert pdominance.dominance_frontier(loop_header) == [loop_header]
        assert pdominance.dominance_frontier(entry_block) == []
    
    def test_if_while_if(self):
        code = """
        int ifwhileif(int a, int b, int c) {
            if (a) {
                printf("starting.");
                while (a) {
                    if (b > 0) {
                       c++;
                    }
                    b += c;
                }
                printf("done.");
            } else {
                printf("Can't loop.");
            }
        return c;
        }"""

        ir = self.parse(code)
        dominance = Dominance(ir)
        pdominance = Dominance(ir, False)

        entry_block = ir.entry_block
        outer_if_before_while = ir.basic_blocks[1]
        loop_header = ir.basic_blocks[2]
        inner_if_condition = ir.basic_blocks[3]
        inner_if_body = ir.basic_blocks[4]
        after_inner_if = ir.basic_blocks[5]
        outer_if_after_while = ir.basic_blocks[6]
        outer_else = ir.basic_blocks[7]
        exit_block = ir.basic_blocks[8]

        # Dominance
        assert dominance.dominance_frontier(entry_block) == []
        assert dominance.dominance_frontier(outer_if_before_while) == [exit_block]
        assert dominance.dominance_frontier(loop_header) == [loop_header, exit_block]
        assert dominance.dominance_frontier(inner_if_condition) == [loop_header]
        assert dominance.dominance_frontier(inner_if_body) == [after_inner_if]
        assert dominance.dominance_frontier(after_inner_if) == [loop_header]
        assert dominance.dominance_frontier(outer_if_after_while) == [exit_block]
        assert dominance.dominance_frontier(outer_else) == [exit_block]
        assert dominance.dominance_frontier(exit_block) == []

        # Postdominance
        assert pdominance.dominance_frontier(exit_block) == []
        assert pdominance.dominance_frontier(outer_if_after_while) == [entry_block]
        assert pdominance.dominance_frontier(loop_header) == [entry_block, loop_header]
        assert pdominance.dominance_frontier(inner_if_condition) == [loop_header]
        assert pdominance.dominance_frontier(inner_if_body) == [inner_if_condition]
        assert pdominance.dominance_frontier(after_inner_if) == [loop_header]
        assert pdominance.dominance_frontier(outer_if_before_while) == [entry_block]
        assert pdominance.dominance_frontier(outer_else) == [entry_block]
        assert pdominance.dominance_frontier(entry_block) == []

class TestControlDependence(unittest.TestCase):
    def parse(self, code: str) -> Function:
        return parse(bytes(code, "utf8"))[0]
    
    def test_if(self):
        code = """
        int foo(int x) {
            if (x) {
               print("done.");
            }
            return x;
        }
        """

        ir = self.parse(code)
        dependence = control_dependence(ir)

        entry_block = ir.entry_block
        if_body = ir.basic_blocks[1]
        post_if = ir.basic_blocks[2]

        correct = {
            entry_block: [],
            if_body: [entry_block],
            post_if: []
        }

        self.assertDictEqual(dependence, correct)

    def test_for_loop(self):
        code = """
        int foo(int x) {
            for (int i = 0; i < x; i++) {
                printf("%d\\n", i);
            }
            return x;
        }
        """

        ir = self.parse(code)
        dependence = control_dependence(ir)

        entry_block = ir.entry_block
        loop_condition = ir.basic_blocks[1]
        update = ir.basic_blocks[2]
        loop_body = ir.basic_blocks[3]
        exit_block = ir.basic_blocks[4]

        correct = {
            entry_block: [],
            loop_condition: [loop_condition],
            update: [loop_condition],
            loop_body: [loop_condition],
            exit_block: []
        }

        self.assertDictEqual(dependence, correct)
    
    def test_for_if_return(self):
        code = """
        int foo(int x, int n) {
            for (int i = 0; i < n; i++) {
                if (x - i > 2) {
                    return 1;
                }
            }

            return 0;
        }
        """

        ir = self.parse(code)
        dependence = control_dependence(ir)

        entry_block = ir.entry_block
        loop_condition = ir.basic_blocks[1]
        update = ir.basic_blocks[2]
        if_condition = ir.basic_blocks[3]
        if_body = ir.basic_blocks[4]
        exit_block = ir.basic_blocks[5]

        correct = {
            entry_block: [],
            loop_condition: [if_condition],
            update: [if_condition],
            if_condition: [loop_condition],
            if_body: [if_condition],
            exit_block: [loop_condition]
        }

        self.assertDictEqual(dependence, correct)

class TestControlEquivalentClasses(unittest.TestCase):
    def parse(self, code: str) -> Function:
        return parse(bytes(code, "utf8"))[0]
    
    def test_if(self):
        code = """
        int foo(x) {
            if (x) {
                print(x);
            }
            return x;
        }
        """

        ir = self.parse(code)
        classes = control_equivalence_classes(ir)

        entry_block = ir.entry_block
        if_body = ir.basic_blocks[1]
        post_if = ir.basic_blocks[2]

        correct = {
            () : [entry_block, post_if],
            (entry_block,) : [if_body]
        }

        self.assertDictEqual(correct, classes)
    
    def test_if_return(self):
        code = """
        int foo(x) {
            if (x) {
                return 4;
            }
            return -1;
        }
        """

        ir = self.parse(code)
        classes = control_equivalence_classes(ir)

        entry_block = ir.entry_block
        if_body = ir.basic_blocks[1]
        post_if = ir.basic_blocks[2]

        correct = {
            () : [entry_block],
            (entry_block,) : [post_if, if_body]
        }

        self.assertDictEqual(classes, correct)
    
    def test_do_while(self):
        code = """
        int foo(int x) {
            while (x < 4) {
               print(x);
               --x;
            }
            return 2;
        }
        """

        ir = self.parse(code)
        classes = control_equivalence_classes(ir)

        entry_block = ir.entry_block
        loop_condition = ir.basic_blocks[1]
        loop_body = ir.basic_blocks[2]
        exit_block = ir.basic_blocks[3]

        correct = {
            () : [entry_block, exit_block],
            (loop_condition,) : [loop_condition, loop_body]
        }

        self.assertDictEqual(classes, correct)

class TestLoopRecognition(unittest.TestCase):
    def parse(self, code: str) -> Function:
        return parse(bytes(code, "utf8"))[0]
    
    def test_while_loop(self):
        code = """
        int foo(x) {
            while (cond(x)) {
                update(x);
            }
            return x;
        }
        """

        ir = self.parse(code)
        loops = find_loops(ir)

        loop_condition = ir.basic_blocks[1]
        loop_body = ir.basic_blocks[2]

        assert len(loops) == 1

        assert loops[0].head == loop_condition
        assert loops[0].back_edge == (loop_body, loop_condition)
        self.assertSetEqual(loops[0].body, {loop_condition, loop_body})
    
    def test_for_loop(self):
        code = """
        int foo(int x) {
            for (int i = 0; i < x; i++) {
                printf("%d\\n", i);
            }
            return x;
        }
        """

        ir = self.parse(code)
        loops = find_loops(ir)

        loop_condition = ir.basic_blocks[1]
        loop_update = ir.basic_blocks[2]
        loop_body = ir.basic_blocks[3]

        assert len(loops) == 1

        assert loops[0].head == loop_condition
        assert loops[0].back_edge == (loop_update, loop_condition)
        self.assertSetEqual(loops[0].body, {loop_condition, loop_update, loop_body})
    
    def test_while_if(self):
        code = """
        int foo(int x) {
            while (cond(x)) {
                if (x % 3) {
                    print(x);
                }
            }
            return x;
        }
        """

        ir = self.parse(code)
        loops = find_loops(ir)

        assert len(loops) == 2

        loop_condition = ir.basic_blocks[1]
        if_condition = ir.basic_blocks[2]
        if_body = ir.basic_blocks[3]

        # It doesn't matter what order the loops are returned in but the are returned in this order
        # consistently.
        assert loops[0].head == loop_condition
        assert loops[0].back_edge == (if_body, loop_condition)
        self.assertSetEqual(loops[0].body, {loop_condition, if_condition, if_body})

        assert loops[1].head == loop_condition
        assert loops[1].back_edge == (if_condition, loop_condition)
        self.assertSetEqual(loops[1].body, {loop_condition, if_condition})

    def test_for_if_return(self):
        code = """
        int foo(int x, int n) {
            for (int i = 0; i < n; i++) {
                if (x - i > 2) {
                    return 1;
                }
            }

            return 0;
        }
        """

        ir = self.parse(code)
        loops = find_loops(ir)

        loop_condition = ir.basic_blocks[1]
        update = ir.basic_blocks[2]
        if_condition = ir.basic_blocks[3]
        # Not the if body - it exits.

        assert len(loops) == 1

        assert loops[0].head == loop_condition
        assert loops[0].back_edge == (update, loop_condition)
        self.assertSetEqual(loops[0].body, {loop_condition, update, if_condition})
    
    def test_nested_loops(self):
        code = """
        int foo(int x) {
            while (cond(x)) {
                while (x - 2 > 0) {
                    update(x);
                }
                print(x);
            }
            return x;
        }
        """

        ir = self.parse(code)
        loops = find_loops(ir)

        outer_condition = ir.basic_blocks[1]
        inner_condition = ir.basic_blocks[2]
        inner_body = ir.basic_blocks[3]
        outer_latch = ir.basic_blocks[4]

        assert len(loops) == 2

        # Outer loop is returned first. This property doesn't matter too much, but it's consistent
        # so we take advantage of it for easier testing.
        assert loops[0].head == outer_condition
        assert loops[0].back_edge == (outer_latch, outer_condition)
        self.assertSetEqual(loops[0].body, {outer_condition, inner_condition, inner_body, outer_latch})

        # Inner loop is second
        assert loops[1].head == inner_condition
        assert loops[1].back_edge == (inner_body, inner_condition)
        self.assertSetEqual(loops[1].body, {inner_condition, inner_body})


if __name__ == '__main__':
    unittest.main()