import unittest

from codealign.lang.c import parse
from codealign.ir import *
from .utils import TestIRGeneration


# Test that IR is formed correctly.
class TestCIRGeneration(TestIRGeneration):
    def parse(self, code: str) -> Function:
        return parse(bytes(code, "utf8"))[0]
    
    def test_binary_op(self):
        code = """
        int foo(int x, int y) {
            int z = x + y;
        }
        """
        
        fn = self.parse(code)
        
        correct = [
            VarOperator("+", Variable("z"), [Parameter("x"), Parameter("y")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_unary_op(self):
        code = """
        int foo(int x) {
            int z = !x;
        }
        """

        fn = self.parse(code)

        correct = [
           VarOperator("!", Variable("z"), [Parameter("x")]) 
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_function_call(self):
        code = """
        int foo(int x, int y) {
            int z = bar(x, y, gbl, 3);
        }
        """

        fn = self.parse(code)

        correct = [
            FunctionVarOperator("bar", Variable("z"), [Parameter("x"), Parameter("y"), GlobalVariable("gbl"), NumberConstant("3")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_function_pointer_call(self):
        code = """
        int foo(int x, void * fnptr) {
            fnptr(1);
            (*fnptr)(2);
            get_ptr(x)(3);
            (*get_ptr(x))(4);
            (*ident)(5, 6, 7);
        }
        """

        fn = self.parse(code)

        correct = [
            FunctionVarOperator(Variable("fnptr"), Variable("t0"), [NumberConstant("1")]),
            FunctionVarOperator(Variable("fnptr"), Variable("t1"), [NumberConstant("2")]),
            FunctionVarOperator("get_ptr", Variable("t2"), [Parameter("x")]),
            FunctionVarOperator(Variable("t2"), Variable("t3"), [NumberConstant("3")]),
            FunctionVarOperator("get_ptr", Variable("t4"), [Parameter("x")]),
            FunctionVarOperator(Variable("t4"), Variable("t5"), [NumberConstant("4")]),
            FunctionVarOperator(Variable("ident"), Variable("t6"), [NumberConstant("5"), NumberConstant("6"), NumberConstant("7")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_pointer_dereference(self):
        code = """
        int foo(int * x) {
            int z = *x;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(POINTER_DEREFERENCE_OP, Variable("z"), [Parameter("x")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_ternary(self):
        code = """
        int foo(int x) {
            float val = x ? -2.8 : 3.4;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(TERNARY_OP, Variable("val"), [Parameter("x"), NumberConstant("-2.8"), NumberConstant("3.4")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_struct_field_access(self):
        code = """
        int foo(struct node * tree, struct node current) {
            struct node * r = tree->right;
            struct node * l = current.left;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator("->", Variable("r"), [Parameter("tree"), Field("right")]),
            VarOperator(".", Variable("l"), [Parameter("current"), Field("left")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_cast_expression(self):
        code = """
        int foo(long long x) {
            int * z = (int *) x;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(CAST_OP, Variable("z"), [TypeName("int *"), Parameter("x")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_subscript_expression(self):
        code = """
        void foo(int * arr) {
            int x = arr[2];
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(SUBSCRIPT_OP, Variable("x"), [Parameter("arr"), NumberConstant("2")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_sizeof(self):
        code = """
        void foo(int x) {
            x = sizeof(x);
            x = sizeof(struct node);
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(SIZEOF_OP, Parameter("x"), [Parameter("x")]),
            VarOperator(SIZEOF_OP, Parameter("x"), [TypeName("struct node")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_comma_operator(self):
        code = """
        void foo(int x, void * fnptr) {
            while(next(), read()) {};
        }
        """

        fn = self.parse(code)

        correct = [
            FunctionVarOperator("next", Variable("t0"), []),
            FunctionVarOperator("read", Variable("t1"), []),
            VarOperator(LOOP_OP, None, [Variable("t1")])
        ]

        self.assertContentsEqual(fn.basic_blocks[1], correct)
    
    def test_initializer_list(self):
        code = """
        void foo(int x) {
            int arr[4] = {0, x + 2, 3, fn(1)};
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator("+", Variable("t0"), [Parameter("x"), NumberConstant("2")]),
            FunctionVarOperator("fn", Variable("t1"), [NumberConstant("1")]),
            VarOperator(ARRAY_INITIALIZER_OP, Variable("arr"), [NumberConstant("0"), Variable("t0"), NumberConstant("3"), Variable("t1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_plus_equals(self):
        code = """
        void foo(int a) {
            a += 1;
            a -= func();
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator("+", Parameter("a"), [Parameter("a"), NumberConstant("1")]),
            FunctionVarOperator("func", Variable("t0"), []),
            VarOperator("-", Parameter("a"), [Parameter("a"), Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_true_false_null(self):
        code = """
        int fn() {
            bar(NULL, true, TRUE, false, FALSE);
        }
        """
        
        fn = self.parse(code)

        correct = [
            FunctionVarOperator("bar", Variable("t0"), [NumberConstant("0"), NumberConstant("1"), NumberConstant("1"), NumberConstant("0"), NumberConstant("0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_void_parameter(self):
        code = """
        int foo(void) {
            return 1;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(RETURN_OP, None, [NumberConstant("1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_multiple_assignment(self):
        code = """
        void foo(int x) {
           int a;
           int b;
           a = b = x;
           a = b = x % 7;
           a *= b += x;
           a >>= b -= x / 2;
           int c;
           a = b = c = x;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(COPY_OP, Variable("b"), [Parameter("x")]),
            VarOperator(COPY_OP, Variable("a"), [Variable("b")]),
            VarOperator("%", Variable("b"), [Parameter("x"), NumberConstant("7")]),
            VarOperator(COPY_OP, Variable("a"), [Variable("b")]),
            VarOperator("+", Variable("b"), [Variable("b"), Parameter("x")]),
            VarOperator("*", Variable("a"), [Variable("a"), Variable("b")]),
            VarOperator("/", Variable("t0"), [Parameter("x"), NumberConstant("2")]),
            VarOperator("-", Variable("b"), [Variable("b"), Variable("t0")]),
            VarOperator(">>", Variable("a"), [Variable("a"), Variable("b")]),
            VarOperator(COPY_OP, Variable("c"), [Parameter("x")]),
            VarOperator(COPY_OP, Variable("b"), [Variable("c")]),
            VarOperator(COPY_OP, Variable("a"), [Variable("b")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_multiple_declarations_single_line(self):
        code = """
        void foo(int x) {
            int * a = 0, * b, c = x + 2, d = 4 - 3 * x;
            func(a, b, c, d);
        }
        """

        fn = self.parse(code)

        correct  = [
            VarOperator(COPY_OP, Variable("a"), [NumberConstant("0")]),
            VarOperator("+", Variable("c"), [Parameter("x"), NumberConstant("2")]),
            VarOperator("*", Variable("t0"), [NumberConstant("3"), Parameter("x")]),
            VarOperator("-", Variable("d"), [NumberConstant("4"), Variable("t0")]),
            # The use of Variable instead of GlobalVariable indicates that these variables were successfully declared inside the function.
            FunctionVarOperator("func", Variable("t1"), [Variable("a"), Variable("b"), Variable("c"), Variable("d")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_prefix_update(self):
        code = """
        void foo(int x, int * src, int * dst) {
            int y = ++x - 3;
            *(--dst) = *(++src)
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator("+", Parameter("x"), [Parameter("x"), NumberConstant("1")]),
            VarOperator("-", Variable("y"), [Parameter("x"), NumberConstant("3")]),
            VarOperator("-", Parameter("dst"), [Parameter("dst"), NumberConstant("1")]),
            VarOperator(POINTER_DEREFERENCE_OP, Variable("t0"), [Parameter("dst")]),
            VarOperator("+", Parameter("src"), [Parameter("src"), NumberConstant("1")]),
            VarOperator(POINTER_DEREFERENCE_OP, Variable("t1"), [Parameter("src")]),
            VarOperator(STORE_OP, Variable("t0"), [Variable("t0"), Variable("t1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_postfix_update(self):
        code = """
        void foo(int x, int * src, int * dst) {
            int a = x-- * 2;
            *(dst++) = *(src++);
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(COPY_OP, Variable("t0"), [Parameter("x")]),
            VarOperator("-", Parameter("x"), [Parameter("x"), NumberConstant("1")]),
            VarOperator("*", Variable("a"), [Variable("t0"), NumberConstant("2")]),
            VarOperator(COPY_OP, Variable("t1"), [Parameter("dst")]),
            VarOperator("+", Parameter("dst"), [Parameter('dst'), NumberConstant("1")]),
            VarOperator(POINTER_DEREFERENCE_OP, Variable("t2"), [Variable("t1")]),
            VarOperator(COPY_OP, Variable("t3"), [Parameter("src")]),
            VarOperator("+", Parameter("src"), [Parameter("src"), NumberConstant('1')]),
            VarOperator(POINTER_DEREFERENCE_OP, Variable("t4"), [Variable("t3")]),
            VarOperator(STORE_OP, Variable("t2"), [Variable("t2"), Variable("t4")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_function_declarator(self):
        code = """
        void foo(int (*fnptr)(int, int)) {
            int myfndeclarator(int a, float c);
            fnptr(1, 2);
        }
        """

        fn = self.parse(code)

        # fnptr being of type Variable (as opposed to GlobalVariable) is significant. It means that the
        # delcarators have been successfully parsed and the variable name has been extracted from them
        # and added to the variable regsitry.
        correct = [
            FunctionVarOperator(Variable("fnptr"), Variable("t0"), [NumberConstant("1"), NumberConstant("2")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_nested_pointer_declarators(self):
        code = """
        void foo(int ** x) {
            ++x;
        }
        """

        fn = self.parse(code)

        # x being of type Variable (as opposed to GlobalVariable) is significant. It means that the
        # delcarators have been successfully parsed and the variable name has been extracted from them
        # and added to the variable regsitry.
        correct = [
            VarOperator("+", Parameter("x"), [Parameter("x"), NumberConstant("1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_nested_array_declarators(self):
        code = """
        void foo() {
            int arr[4][5];
            func(arr);
        }
        """

        fn = self.parse(code)

        # arr being of type Variable (as opposed to GlobalVariable) is significant. It means that the
        # delcarators have been successfully parsed and the variable name has been extracted from them
        # and added to the variable regsitry.
        correct = [
            FunctionVarOperator("func", Variable("t0"), [Variable("arr")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_declarator_precedence(self):
        # contains: init_declarator, pointer_declarator, array_declarator, function_declarator, and parenthesized_declarator
        code = """
        void foo() {
            int (*x[8])(int, int) = 0;
            init(x);
        }
        """

        fn = self.parse(code)

        # x being of type Variable (as opposed to GlobalVariable) is significant. It means that the
        # delcarators have been successfully parsed and the variable name has been extracted from them
        # and added to the variable regsitry.
        correct = [
            VarOperator(COPY_OP, Variable("x"), [NumberConstant("0")]),
            FunctionVarOperator("init", Variable("t0"), [Variable("x")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_const_qualifier(self):
        code = """
        void foo() {
            char const *a;
            const char b = 'b';
            const char * const c;
            bar(a, b, c);
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(COPY_OP, Variable("b"), [CharLiteral("'b'")]),
            FunctionVarOperator("bar", Variable("t0"), [Variable("a"), Variable("b"), Variable("c")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_return(self):
        code = """
        int foo() {
            return 0;
        }
        """

        fn = self.parse(code)

        correct= [
            VarOperator(RETURN_OP, None, [NumberConstant("0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_compound_expression(self):
        code = """
        void foo(int x, struct values * vals) {
            return vals->init + adjust(x * vals->rate, ADJUSTMENT_FACTOR);
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator("->", Variable("t0"), [Parameter("vals"), Field("init")]),
            VarOperator("->", Variable("t1"), [Parameter("vals"), Field("rate")]),
            VarOperator("*", Variable("t2"), [Parameter("x"), Variable("t1")]),
            FunctionVarOperator("adjust", Variable("t3"), [Variable("t2"), GlobalVariable("ADJUSTMENT_FACTOR")]),
            VarOperator("+", Variable("t4"), [Variable("t0"), Variable("t3")]),
            VarOperator(RETURN_OP, None, [Variable("t4")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_expression_lval_constant_rval(self):
        code = """
        void foo(int * arr) {
            arr[0] = 1;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(SUBSCRIPT_OP, Variable("t0"), [Parameter("arr"), NumberConstant("0")]),
            VarOperator(STORE_OP, Variable("t0"), [Variable("t0"), NumberConstant("1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_expression_lval_expression_rval(self):
        code = """
        void foo(point * pt, int x) {
            pt->x = x + 1;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator("->", Variable("t0"), [Parameter("pt"), Field("x")]),
            VarOperator("+", Variable("t1"), [Parameter("x"), NumberConstant("1")]),
            VarOperator(STORE_OP, Variable("t0"), [Variable("t0"), Variable("t1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_nested_expression_lvals(self):
        code = """
        void foo(point * pt1, point * pt2, int x) {
            func(pt1, FLAG1)->x = func(pt2, FLAG2)->y = x * 2;
        }
        """

        fn = self.parse(code)

        correct  = [
            FunctionVarOperator("func", Variable("t0"), [Parameter("pt1"), GlobalVariable("FLAG1")]),
            VarOperator("->", Variable("t1"), [Variable("t0"), Field("x")]),
            FunctionVarOperator("func", Variable("t2"), [Parameter("pt2"), GlobalVariable("FLAG2")]),
            VarOperator("->", Variable("t3"), [Variable("t2"), Field("y")]),
            VarOperator("*", Variable("t4"), [Parameter("x"), NumberConstant("2")]),
            VarOperator(STORE_OP, Variable("t3"), [Variable("t3"), Variable("t4")]),
            VarOperator(STORE_OP, Variable("t1"), [Variable("t1"), Variable("t3")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_expression_lval_compound_assignment(self):
        code = """
        void foo(int * arr) {
            arr[0] += 1;
            arr[1] *= arr[0] - 2;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(SUBSCRIPT_OP, Variable("t0"), [Parameter("arr"), NumberConstant("0")]),
            VarOperator("+", Variable("t1"), [Variable("t0"), NumberConstant("1")]),
            VarOperator(STORE_OP, Variable("t0"), [Variable("t0"), Variable("t1")]),
            VarOperator(SUBSCRIPT_OP, Variable("t2"), [Parameter("arr"), NumberConstant("1")]),
            VarOperator(SUBSCRIPT_OP, Variable("t3"), [Parameter("arr"), NumberConstant("0")]),
            VarOperator("-", Variable("t4"), [Variable("t3"), NumberConstant("2")]),
            VarOperator("*", Variable("t5"), [Variable("t2"), Variable("t4")]),
            VarOperator(STORE_OP, Variable("t2"), [Variable("t2"), Variable("t5")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_switch(self):
        code = """
        int foo(int c) {
            switch (c) {
                int x = 0;
                case 1: 
                case 2: printf("2"); break;
                case (3): printf("3");
                default:
                    printf("default");
            }
            return 0;
        }
        """

        fn = self.parse(code)

        entry_block = [
            VarOperator("==", Variable("t0"), [Parameter("c"), NumberConstant("1")]),
            VarOperator(IF_OP, None, [Variable("t0")])
        ]
        case2condition = [
            VarOperator("==", Variable("t1"), [Parameter("c"), NumberConstant("2")]),
            VarOperator(IF_OP, None, [Variable("t1")])
        ]
        case2body = [FunctionVarOperator("printf", Variable("t2"), [StringLiteral("\"2\"")]), VarOperator(BREAK_OP, None, [])]
        case3condition = [
            VarOperator("==", Variable("t3"), [Parameter("c"), NumberConstant("3")]),
            VarOperator(IF_OP, None, [Variable("t3")])
        ]
        case3body = [FunctionVarOperator("printf", Variable("t4"), [StringLiteral("\"3\"")])]
        default_block = [FunctionVarOperator("printf", Variable("t5"), [StringLiteral("\"default\"")])]
        exit_block = [VarOperator(RETURN_OP, None, [NumberConstant("0")])]

        self.assertContentsEqual(fn.entry_block, entry_block)
        self.assertContentsEqual(fn.basic_blocks[1], case2condition)
        self.assertContentsEqual(fn.basic_blocks[2], case2body)
        self.assertContentsEqual(fn.basic_blocks[3], case3condition)
        self.assertContentsEqual(fn.basic_blocks[4], case3body)
        self.assertContentsEqual(fn.basic_blocks[5], default_block)
        self.assertContentsEqual(fn.basic_blocks[6], exit_block)

    def test_nested_compound_statements(self):
        code = """
        int main() {
            int x = 0;
            {
                int x = 1;
                printf("%d\\n", x);
            }
            printf("%d\\n", x);
        }
        """

        fn = self.parse(code)

        block0 = [
            VarOperator(COPY_OP, Variable("x"), [NumberConstant("0")])
        ]
        block1 = [
            VarOperator(COPY_OP, Variable("x"), [NumberConstant("1")]),
            FunctionVarOperator("printf", Variable("t0"), [StringLiteral("\"%d\\n\""), Variable("x")])
        ]
        block2 = [
            FunctionVarOperator("printf", Variable("t0"), [StringLiteral("\"%d\\n\""), Variable("x")])
        ]

        self.assertContentsEqual(fn.entry_block, block0)
        self.assertContentsEqual(fn.basic_blocks[1], block1)
        self.assertContentsEqual(fn.basic_blocks[2], block2)
        self.assertEqual(fn.entry_block.operators[0].result, fn.basic_blocks[2].operators[0].operands[1], f"Outer scope 'x' is a different variable.")
        self.assertEqual(fn.basic_blocks[1].operators[0].result, fn.basic_blocks[1].operators[1].operands[1], f"Inner scope 'x' is a different variable.")
        self.assertNotEqual(fn.entry_block.operators[0].result, fn.basic_blocks[1].operators[0].result, f"Variables of the same name in different scopes should be distinct variable objects.")
    
    def test_comma_right_assignment(self):
        code = """
        int main() {
            int a, b, c;
            c = (myfunc(a=4), myfunc(b=5));
            return 0;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(COPY_OP, Variable("a"), [NumberConstant("4")]),
            FunctionVarOperator("myfunc", Variable("t0"), [Variable("a")]),
            VarOperator(COPY_OP, Variable("b"), [NumberConstant("5")]),
            FunctionVarOperator("myfunc", Variable("c"), [Variable("b")]),
            VarOperator(RETURN_OP, None, [NumberConstant("0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_for_comma_initialization(self):
        code = """
        void myfn(int n) {
            int i, j;
            for (i=0,j=1;;);
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(COPY_OP, Variable("i"), [NumberConstant("0")]),
            VarOperator(COPY_OP, Variable("j"), [NumberConstant("1")]),
            VarOperator(COPY_OP, Variable("t0"), [Variable("j")]), # Unnecessarily (but harmlessly) created by bind_expression
        ]

        self.assertContentsEqual(fn.entry_block, correct)
        self.assertContentsEqual(fn.basic_blocks[1], [VarOperator(LOOP_OP, None, [NumberConstant("1")])])

    def test_concatenated_string(self):
        code = """
        void myfn() {
            printf("One" "Two" "Three");
        }
        """

        fn = self.parse(code)

        correct = [
            FunctionVarOperator("printf", Variable("t0"), [StringLiteral("\"OneTwoThree\"")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_compound_literal_expression(self):
        code = """
        void myfn() {
           fn((struct thing){.this = 1 + 7, .that = (44) });
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator("+", Variable("t0"), [NumberConstant("1"), NumberConstant("7")]),
            VarOperator(TUPLE_INITIALIZER_OP, Variable("t1"), [Variable("t0"), NumberConstant("44")]),
            FunctionVarOperator("fn", Variable("t2"), [Variable("t1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_ignorable_struct_specifier_and_empty_statement(self):
        code = """
        int main() {
            struct point {
                int x;
                int y;
            };
            ;
            return 0;
        }
        """

        fn = self.parse(code)

        correct = [
            VarOperator(RETURN_OP, None, [NumberConstant("0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)



# Ensures the CFG is correct. Does not check the contents of the basic blocks.
# These tests are sensitive to the order that the basic blocks are stored in a Function's basic_block list.
class TestCFG(unittest.TestCase):
    def parse(self, code: str):
        return parse(bytes(code, "utf8"))[0]
    
    def test_if(self):
        code = """
        int foo(int x) {
            if (x) {
                print("x is positive.");
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 3
        entry_block = fn.entry_block
        if_body = fn.basic_blocks[1]
        exit_block = fn.basic_blocks[2]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert if_body in entry_block.successors
        assert exit_block in entry_block.successors

        assert if_body.predecessors == [entry_block]
        assert if_body.successors == [exit_block]
        
        assert len(exit_block.predecessors) == 2
        assert entry_block in exit_block.predecessors
        assert if_body in exit_block.predecessors
        assert exit_block.successors == []
    
    def test_terminating_if(self):
        code = """
        void foo(int x) {
            if (x) {
                print("x is positive.");
            }
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 3
        entry_block = fn.entry_block
        if_body = fn.basic_blocks[1]
        exit_block = fn.basic_blocks[2]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert if_body in entry_block.successors
        assert exit_block in entry_block.successors

        assert if_body.predecessors == [entry_block]
        assert if_body.successors == [exit_block]

        assert len(exit_block.predecessors) == 2
        assert entry_block in exit_block.predecessors
        assert if_body in exit_block.predecessors
        assert len(exit_block.successors) == 0
    
    def test_if_else(self):
        code = """
        int foo(int x) {
            if (x)
                print("x is positive.");
            else
                print("x is not positive.");
            
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        if_body = fn.basic_blocks[1]
        else_block = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert if_body in entry_block.successors
        assert else_block in entry_block.successors
        
        assert if_body.predecessors == [entry_block]
        assert if_body.successors == [exit_block]

        assert else_block.predecessors == [entry_block]
        assert else_block.successors == [exit_block]

        assert len(exit_block.predecessors) == 2
        assert if_body in exit_block.predecessors
        assert else_block in exit_block.predecessors
        assert exit_block.successors == []
    
    def test_terminating_if_else(self):
        code = """
        void foo(int x) {
            if (x) {
                print("x is positive.");
            } else {
                print("x is not positive.");
            }
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 3
        entry_block = fn.entry_block
        if_body = fn.basic_blocks[1]
        else_block = fn.basic_blocks[2]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert if_body in entry_block.successors
        assert else_block in entry_block.successors
        
        assert if_body.predecessors == [entry_block]
        assert if_body.successors == []

        assert else_block.predecessors == [entry_block]
        assert else_block.successors == []

    def test_else_if(self):
        code = """
        int foo(int a, int b) {
            if (a > b) {
                printf("A")
            } else if (a < b) {
                print("B")
            }
            return a < b
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 5
        entry_block = fn.entry_block
        true_block = fn.basic_blocks[1]
        false_block = fn.basic_blocks[2]
        false_true_block = fn.basic_blocks[3]
        return_block = fn.basic_blocks[4]

        assert len(entry_block.predecessors) == 0
        assert len(entry_block.successors) ==  2
        assert true_block in entry_block.successors
        assert false_block in entry_block.successors

        assert true_block.predecessors == [entry_block]
        assert true_block.successors == [return_block]

        assert false_block.predecessors == [entry_block]
        assert len(false_block.successors) == 2
        assert false_true_block in false_block.successors
        assert return_block in false_block.successors
        
        assert false_true_block.predecessors == [false_block]
        assert false_true_block.successors == [return_block]

        assert len(return_block.predecessors) == 3
        assert true_block in return_block.predecessors
        assert false_block in return_block.predecessors
        assert false_true_block in return_block.predecessors
        assert len(return_block.successors) == 0
    
    def test_else_if_while_else(self):
        code = """
        int foo(int a, int b) {
            if (a < b) {
                print("A");
            } else if (a > b) {
                while (flag) doit();
            } else {
                print("Same");
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 7
        entry_block = fn.entry_block
        true_block = fn.basic_blocks[1]
        middle_block = fn.basic_blocks[2]
        while_condition = fn.basic_blocks[3]
        while_body = fn.basic_blocks[4]
        else_block = fn.basic_blocks[5]
        return_block = fn.basic_blocks[6]

        assert len(entry_block.predecessors) == 0
        assert len(entry_block.successors) == 2
        assert true_block in entry_block.successors
        assert middle_block in entry_block.successors

        assert true_block.predecessors == [entry_block]
        assert true_block.successors == [return_block]

        assert middle_block.predecessors == [entry_block]
        assert len(middle_block.successors) == 2
        assert while_condition in middle_block.successors
        assert else_block in middle_block.successors

        assert len(while_condition.predecessors) == 2
        assert middle_block in while_condition.predecessors
        assert while_body in while_condition.predecessors
        assert len(while_condition.successors) == 2
        assert return_block in while_condition.successors
        assert while_body in while_condition.successors

        assert while_body.predecessors == [while_condition]
        assert while_body.successors == [while_condition]

        assert else_block.predecessors == [middle_block]
        assert else_block.successors == [return_block]

        assert len(return_block.predecessors) == 3
        assert true_block in return_block.predecessors
        assert while_condition in return_block.predecessors
        assert else_block in return_block.predecessors
        assert len(return_block.successors) == 0
    
    def test_for(self):
        code  = """
        int foo(int x) {
            for (int i = 0; i < x; i++) {
                printf("%d\\n", i);
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 5
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_update = fn.basic_blocks[2]
        loop_body = fn.basic_blocks[3]
        exit_block = fn.basic_blocks[4]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_test]

        assert len(loop_test.predecessors) == 2
        assert entry_block in loop_test.predecessors
        assert loop_update in loop_test.predecessors
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors
        
        assert loop_body.predecessors == [loop_test]
        assert loop_body.successors == [loop_update]

        assert loop_update.predecessors == [loop_body]
        assert loop_update.successors == [loop_test]

        assert exit_block.predecessors== [loop_test]
        assert exit_block.successors == []
    
    def test_terminating_for(self):
        code  = """
        void foo(int x) {
            for (int i = 0; i < x; i++) {
                printf("%d\\n", i);
            }
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 5
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_update = fn.basic_blocks[2]
        loop_body = fn.basic_blocks[3]
        exit_block = fn.basic_blocks[4]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_test]

        assert len(loop_test.predecessors) == 2
        assert entry_block in loop_test.predecessors
        assert loop_update in loop_test.predecessors
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors
        
        assert loop_body.predecessors == [loop_test]
        assert loop_body.successors == [loop_update]

        assert loop_update.predecessors == [loop_body]
        assert loop_update.successors == [loop_test]

        assert exit_block.predecessors == [loop_test]
        assert len(exit_block.successors) == 0
    
    def test_while(self):
        code = """
        int foo(char * message) {
            while (rand() > 0.4) {
                printf(message);
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_body = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_test]

        assert len(loop_test.predecessors) == 2
        assert entry_block in loop_test.predecessors
        assert loop_body in loop_test.predecessors
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors

        assert loop_body.predecessors == [loop_test]
        assert loop_body.successors == [loop_test]

        assert exit_block.predecessors == [loop_test]
        assert exit_block.successors == []
    
    def test_terminating_while(self):
        code = """
        void foo(char * message) {
            while (rand() > 0.4) {
                printf(message);
            }
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_body = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_test]

        assert len(loop_test.predecessors) == 2
        assert entry_block in loop_test.predecessors
        assert loop_body in loop_test.predecessors
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors

        assert loop_body.predecessors == [loop_test]
        assert loop_body.successors == [loop_test]

        assert exit_block.predecessors == [loop_test]
        assert len(exit_block.successors) == 0
    
    def test_do_while(self):
        code = """
        int foo(char * message) {
            do {
                printf(message);
            } while (rand() > 0.4);
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_body = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_body]
        
        assert len(loop_body.predecessors) == 2
        assert entry_block in loop_body.predecessors
        assert loop_test in loop_body.predecessors
        assert loop_body.successors == [loop_test]

        assert loop_test.predecessors == [loop_body]
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors
        
        assert exit_block.predecessors == [loop_test]
        assert exit_block.successors == []
    
    def test_terminating_do_while(self):
        code = """
        int foo(char * message) {
            do {
                printf(message);
            } while (rand() > 0.4);
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_body = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_body]
        
        assert len(loop_body.predecessors) == 2
        assert entry_block in loop_body.predecessors
        assert loop_test in loop_body.predecessors
        assert loop_body.successors == [loop_test]

        assert loop_test.predecessors == [loop_body]
        assert len(loop_test.successors) == 2
        assert loop_body in loop_test.successors
        assert exit_block in loop_test.successors

        assert exit_block.predecessors == [loop_test]
        assert len(exit_block.successors) == 0
    
    def test_if_return(self):
        code = """
        int foo(int x) {
            if (x) {
                return 1;
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 3
        entry_block = fn.entry_block
        if_body = fn.basic_blocks[1]
        exit_block = fn.basic_blocks[2]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert if_body in entry_block.successors
        assert exit_block in entry_block.successors

        assert if_body.predecessors == [entry_block]
        assert if_body.successors == []
        
        assert exit_block.predecessors == [entry_block]
        assert exit_block.successors == []
    
    def test_for_if_return(self):
        code = """
        int foo(int x) {
            int i;
            bar();
            for (int i = 0; i < gbl; i++) {
                if (gbl2 == x){
                    return 1;
                }
            } 
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 6
        entry_block = fn.entry_block
        loop_test = fn.basic_blocks[1]
        loop_update = fn.basic_blocks[2]
        loop_body_if = fn.basic_blocks[3]
        if_body = fn.basic_blocks[4]
        post_loop = fn.basic_blocks[5]

        assert entry_block.predecessors == []
        assert entry_block.successors == [loop_test]
        
        assert len(loop_test.predecessors) == 2
        assert entry_block in loop_test.predecessors
        assert loop_update in loop_test.predecessors
        assert len(loop_test.successors) == 2
        assert loop_body_if in loop_test.successors
        assert post_loop in loop_test.successors

        assert loop_body_if.predecessors == [loop_test]
        assert len(loop_body_if.successors) == 2
        assert if_body in loop_body_if.successors
        assert loop_update in loop_body_if.successors

        assert if_body.predecessors == [loop_body_if]
        assert if_body.successors == [] # has a return statement.

        assert loop_update.predecessors == [loop_body_if]
        assert loop_update.successors == [loop_test]

        assert post_loop.predecessors == [loop_test]
        assert post_loop.successors == []
    
    def test_if_if(self):
        code = """
        int foo(int x, int y) {
            if (x) {
                if (y) {
                    print("done.");
                }
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 4
        entry_block = fn.entry_block
        inner_if = fn.basic_blocks[1]
        inner_if_body = fn.basic_blocks[2]
        exit_block = fn.basic_blocks[3]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert inner_if in entry_block.successors
        assert exit_block in entry_block.successors
        
        assert inner_if.predecessors == [entry_block]
        assert len(inner_if.successors) == 2
        assert inner_if_body in inner_if.successors
        assert exit_block in inner_if.successors

        assert inner_if_body.predecessors == [inner_if]
        assert inner_if_body.successors == [exit_block]

        assert len(exit_block.predecessors) == 3
        assert entry_block in exit_block.predecessors
        assert inner_if in exit_block.predecessors
        assert inner_if_body in exit_block.predecessors
        assert exit_block.successors == []
    
    def test_if_if_if(self):
        code = """
        int foo(int x) {
            if (x > 0) {
                if (x < 30) {
                    if (y) {
                        print("passed.");
                    }
                }
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 5
        entry_block = fn.entry_block
        middle_if = fn.basic_blocks[1]
        inner_if = fn.basic_blocks[2]
        inner_if_body = fn.basic_blocks[3]
        exit_block = fn.basic_blocks[4]

        assert entry_block.predecessors == []
        assert len(entry_block.successors) == 2
        assert middle_if in entry_block.successors
        assert exit_block in entry_block.successors

        assert middle_if.predecessors == [entry_block]
        assert len(middle_if.successors) == 2
        assert inner_if in middle_if.successors
        assert exit_block in middle_if.successors

        assert inner_if.predecessors == [middle_if]
        assert len(inner_if.successors) == 2
        assert inner_if_body in inner_if.successors
        assert exit_block in inner_if.successors

        assert inner_if_body.predecessors == [inner_if]
        assert inner_if_body.successors == [exit_block]

        assert len(exit_block.predecessors) == 4
        assert entry_block in exit_block.predecessors
        assert middle_if in exit_block.predecessors
        assert inner_if in exit_block.predecessors
        assert inner_if_body in exit_block.predecessors
        assert exit_block.successors == []

    def test_switch_return(self):
        code = """
        int compute_size(type) {
            switch (type) {
            case 1:
                return 4;
            case 2:
                return 8;
            default:
                return 12;
            }
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 5

        entry_block = fn.basic_blocks[0]
        case1_block = fn.basic_blocks[1]
        test2_block = fn.basic_blocks[2]
        case2_block = fn.basic_blocks[3]
        default_block = fn.basic_blocks[4]

        assert len(entry_block.predecessors) == 0
        assert len(entry_block.successors) == 2
        assert case1_block in entry_block.successors
        assert test2_block in entry_block.successors

        assert len(case1_block.predecessors) == 1
        assert entry_block in case1_block.predecessors
        assert len(case1_block.successors) == 0

        assert len(test2_block.predecessors) == 1
        assert entry_block in test2_block.predecessors
        assert len(test2_block.successors) == 2
        assert case2_block in test2_block.successors
        assert default_block in test2_block.successors
        
        assert len(case2_block.predecessors) == 1
        assert test2_block in case2_block.predecessors
        assert len(case2_block.successors) == 0

        assert len(default_block.predecessors) == 1
        assert test2_block in default_block.predecessors
        assert len(default_block.successors) == 0
    
    def test_unreachable_block(self):
        code = """
        int bar(int x, int y) {
            if (x) {
                return -y;
            } else {
                return y;
            }
            return 0;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 3

        entry_block = fn.basic_blocks[0]
        true_block = fn.basic_blocks[1]
        false_block = fn.basic_blocks[2]

        assert len(entry_block.predecessors) == 0
        assert len(entry_block.successors) == 2
        assert true_block in entry_block.successors
        assert false_block in entry_block.successors

        assert len(true_block.predecessors) == 1
        assert entry_block in true_block.predecessors
        assert len(true_block.successors) == 0

        assert len(false_block.predecessors) == 1
        assert entry_block in false_block.predecessors
        assert len(false_block.successors) == 0
    
    def test_return_in_nested_compound_statement(self):
        code = """
        int fn(int x) {
            x = x + 4;
            {
                x = x * 7;
                return x;
            }
            x = x + 8;
            return x;
        }
        """

        fn = self.parse(code)

        assert len(fn.basic_blocks) == 2
        
        entry_block = fn.basic_blocks[0]
        return_block = fn.basic_blocks[1]

        assert len(entry_block.predecessors) == 0
        assert len(entry_block.successors) == 1
        assert return_block in entry_block.successors

        assert len(return_block.predecessors) == 1
        assert entry_block in return_block.predecessors
        assert len(return_block.successors) == 0




if __name__ == '__main__':
    unittest.main()