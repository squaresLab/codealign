import unittest
from typing import Iterable

from codealign import align, Alignment
from codealign.ir import SSAOperator, Parameter, Function


class TestAlignment(unittest.TestCase):
    def alignment_equivalent(self, mapping: set[tuple[SSAOperator | None, SSAOperator | None]], alignment: Alignment):
        self.assertSetEqual(
            self.get_mapping_from_indices(mapping, alignment), 
            {a for a in alignment.alignment_list if not isinstance(a[0], Parameter)}
        )

    def get_operator(self, ir: Function, block_idx: int, operator_idx: int):
        assert block_idx < len(ir.basic_blocks), f"Function {ir.name} does not have a basic block at index {block_idx}"
        assert operator_idx < len(ir.basic_blocks[block_idx].operators), f"Function {ir.name} does not have an operator at block {block_idx}, index {operator_idx}"
        return ir.basic_blocks[block_idx].operators[operator_idx]

    def get_mapping_from_indices(self, indices: Iterable[tuple[tuple[int, int] | None, tuple[int, int] | None]], alignment: Alignment) -> set[tuple[SSAOperator | None, SSAOperator | None]]:
        """Build a ground-truth alignment from the specified indices.

        indices: the basic block index and operator index for each pair of operators in the candidate and reference functions. 
                 Either set of indices (candidate or reference) may be None, but not both.
        alignment: the alignment (used to obtain the candidate and reference IR.)
        """
        candidate_ir = alignment.candidate_ir
        reference_ir = alignment.reference_ir

        operator_mapping = set()
        for cand_indices, ref_indices in indices:
            operator_mapping.add((
                None if cand_indices is None else self.get_operator(candidate_ir, *cand_indices),
                None if ref_indices is None else self.get_operator(reference_ir, *ref_indices)
            ))
        # Make sure that indices doesn't contain a duplicate entry.
        self.assertEqual(len(indices), len(operator_mapping))
        return operator_mapping
    
    def perfect_indices(self, ir: Function):
        indices: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for i, bb in enumerate(ir):
            for j in range(len(bb.operators)):
                indices.append(((i, j), (i, j)))
        return indices
    
    def perfect_self_alignment(self, code: str):
        alignment = align(code, code, language='c', injective=False, control_dependence=True, partial_loops=False)
        indices = self.perfect_indices(alignment.candidate_ir)
        self.alignment_equivalent(indices, alignment)

    def test_control_alignment_if(self):
        candidate = """
        void foo() {
            print("A");
            if (flag) {
                print("B");
            }
        }
        """

        reference = """
        void bar() {
            print("A");
            if (flag) {}
            print("B");
        }
        """

        alignment = align(candidate, reference, language='c')

        # The indices of the basic blocks and operators in each function in the format (candidate, reference).
        # 'None' means that the operator doesn't align with anything.
        indices = [
            ((0, 0), (0, 0)),
            ((0, 1), (0, 1)),
            ((1, 0), None),
            (None, (2, 0))
        ]

        self.alignment_equivalent(indices, alignment)

    def test_control_alignment_while(self):
        candidate = """
        void foo() {
            while (condition()) {
                print("A");
            }
        }
        """

        reference = """
        void bar() {
            while (condition());
            print("A");
        }
        """

        alignment = align(candidate, reference, "c")

        indices = [
            ((1, 0), (1, 0)),
            ((1, 1), (1, 1)),
            ((2, 0), None),
            (None, (2, 0))
        ]

        self.alignment_equivalent(indices, alignment)

    def test_swapped_if_else(self):
        candidate = """
        int foo(int a) {
            if (a) {
                print("A");
            } else {
                print("B");
            }
            return 0;
        }
        """

        reference = """
        int bar(int a) {
            if (a) {
                print("B");
            } else {
                print("A");
            }
            return 0;
        }
        """

        alignment = align(candidate, reference, "c", control_dependence=True, partial_loops=False)

        indices = [
            ((0, 0), (0, 0)),
            ((1, 0), None),
            (None, (1, 0)),
            ((2, 0), None),
            (None, (2, 0)),
            ((3, 0), (3, 0))
        ]

        self.alignment_equivalent(indices, alignment)

    def test_joint_data_control_cycle_decomposition(self):
        code = """
        int write_response(int fd, char *buf, int len) {
            int i = 1;
            while (i) {
                i = write(fd, buf, len);
            }
            return 0;
        }
        """
        self.perfect_self_alignment(code)

    def test_control_dependence_cycle_decomposition(self):
        code = """
        int main(int argc, char *argv[]) {
            int i = 0;
            while(i++ < 10){
                if(condition()) break;
            }
            return 0;
        }
        """
        self.perfect_self_alignment(code)

    def test_loop_head_dependent_on_body(self):
        code = """
        int main(int a, int year){
            do{
            if (cond()) {
                while (year--) {
                    print(year);
                }
            }
            a += 400;
            if (year > 0) {
                print(year);
            }
            print(a);
            } while(year>=a);
        return a;
        }
        """
        self.perfect_self_alignment(code)

    def test_nested_if_loop_break(self):
        code = """
        int fn(a, n) {
            for (int i; i < n; i++) {
            if (i) {
                if (a[i]) {
                print("Found valid item.");
                break;
                }
            }
            }
            return 0;
        }
        """
        # This should self-align perfectly.
        self.perfect_self_alignment(code)

        # However, this example is also good for testing whether control-dependence loopbreaking 
        # is done correctly in partial-loop scenarios as well because it has operatons with multiple
        # control dependencies.
        alignment = align(code, code, language='c', control_dependence=True, partial_loops=True)
        self.alignment_equivalent(self.perfect_indices(alignment.candidate_ir), alignment)

    def test_slightly_different_loops_no_partials(self):
        candidate = """
        void printmat(int mat, int n, int m) {
            for (int j = 0; j < m; j++) {
                print(" ");
            }
        }
        """

        reference = """
        void printmat(int mat, int n, int m) {
            for (int j = 0; j < m; j += 2) {
                print(j);
            }
        }
        """

        alignment = align(candidate, reference, language='c', control_dependence=True, partial_loops=False)

        indices = [
            ((1, 0), None),
            ((1, 1), None),
            ((1, 2), None),
            ((2, 0), None),
            ((3, 0), None),
            (None, (1, 0)),
            (None, (1, 1)),
            (None, (1, 2)),
            (None, (2, 0)),
            (None, (3, 0))
        ]
        self.alignment_equivalent(indices, alignment)

    def test_identical_control_different_dataflow(self):
        candidate = """
        int loopthing(int mat, int n, int m) {
           int i = 0;
           int total = 0;
           while (doit(i)) {
             i = update(i);
             if (i < 0) {
              break;
             }
             writelog(mat, i);
             total += 1;
           }
           return total;
        }
        """

        reference = """
        int loopthing(int mat, int n, int m) {
           int i = 0;
           int total = 0;
           while (doit(i)) {
             i = update(i);
             if (i < 0) {
              break;
             }
             writelog(mat, i);
             total += 2;
           }
           return total;
        }
        """

        alignment = align(candidate, reference, language='c', control_dependence=True, partial_loops=False)

        indices = [
            ((1, 0), (1, 0)),
            ((1, 1), None),
            (None, (1, 1)),
            ((1, 2), (1, 2)),
            ((1, 3), (1, 3)),
            ((2, 0), (2, 0)),
            ((2, 1), (2, 1)),
            ((2, 2), (2, 2)),
            ((3, 0), (3, 0)),
            ((4, 0), (4, 0)),
            ((4, 1), None),
            (None, (4, 1)),
            ((5, 0), None),
            (None, (5, 0))
        ]

        self.alignment_equivalent(indices, alignment)

    def test_parenthetical_function_name_in_call(self):
        code = """
        int sqidt() {
            return (sqrt)(1);
        }
        """

        self.perfect_self_alignment(code)

    def test_branch_based_control_dependency_ordering(self):
        code = """
        int fn(int a, int m, int flag) {
            int i, j=0, k=0;
                for (i=0;i<m;i++) {
                        if (i==0) {
                            if (flag) {
                                if (a[0][i]) {
                                        print("One");
                                        continue;
                                    }
                            } else if (j==m-1) {
                                if (a[j][i]) {
                                        print("Two");
                                        continue;
                                    }
                            }
                        }
                        if (i==m-1) {
                            if (flag) {
                                if (a[1][i]) {
                                        print("Three");
                                        continue;
                                    }
                            }
                        }
            }
            return 0;
        }
        """
        self.perfect_self_alignment(code)

    def test_multiple_paths_to_same_node_control_branch_ordering(self):
        code = func71 = """
        bool multipaths(var *flags) {
            if (flags[0]) {
                if (input_seek_errno == ESPIPE)
                    return true;
            } else {
                if (flags[1]) {
                    return false;
                    }
                if (flags[2]) {
                    if (flags[3])
                        return true;
                    if (flags[4])
                        print("error");
                    if (flags[5])
                        return true;
                    }
                }
            return false;
        }
        """
        self.perfect_self_alignment(code)
        