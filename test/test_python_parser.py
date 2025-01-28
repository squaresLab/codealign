import unittest

from codealign.lang.python import parse, Scope, ITERATOR_FN_NAME
from codealign.ir import *
from .utils import TestIRGeneration


class TestScope(unittest.TestCase):
    def test_parameters(self):
        global_scope = Scope()
        fn_scope = Scope(global_scope)

        p1 = fn_scope.create_parameter("p1")
        p2 = fn_scope.create_parameter("p2")

        assert fn_scope.variable_read("p1") == p1
        assert fn_scope.variable_written("p1") == p1
        assert fn_scope.variable_written("p2") == p2
        assert fn_scope.variable_read("p2") == p2
    
    def test_global_read(self):
        global_scope = Scope()
        fn_scope = Scope(global_scope)

        gbl = fn_scope.variable_read("gbl")
        assert isinstance(gbl, GlobalVariable)
        assert gbl == global_scope.variable_read("gbl")

    def test_global_statement_new_var(self):
        global_scope = Scope()
        fn_scope = Scope(global_scope)

        fn_scope.declare_global("gbl")
        assert global_scope.variable_read("gbl") == fn_scope.variable_read("gbl")
        assert isinstance(global_scope.variable_read("gbl"), GlobalVariable)
    
    def test_nonlocal(self):
        global_scope = Scope()
        fn_scope = Scope(global_scope)
        inner_fn_scope = Scope(fn_scope)

        x = fn_scope.variable_written("x")
        inner_fn_scope.declare_nonlocal("x")
        
        assert x == inner_fn_scope.variable_read("x")
    
    def test_deleted(self):
        global_scope = Scope()
        fn_scope = Scope(global_scope)

        old_x = fn_scope.variable_written("x")
        fn_scope.delete_variable("x")
        
        new_x = fn_scope.variable_written("x")
        assert old_x != new_x





class TestPythonIRGeneration(TestIRGeneration):
    def parse(self, code: str) -> Function:
        return parse(bytes(code, "utf8"))[0]
    
    def test_binary_op(self):
        code = """
        def foo(x, y):
            z = x + y
        """

        fn = self.parse(code)
        
        correct = [
            VarOperator("+", Variable("z"), [Parameter("x"), Parameter("y")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_unary_op(self):
        code = """
        def foo(x):
            z = -x
        """

        fn = self.parse(code)

        correct = [
           VarOperator("-", Variable("z"), [Parameter("x")]) 
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_attribute_access(self):
        code = """
        def foo(x):
            z = x.y
        """

        fn = self.parse(code)

        correct = [
            VarOperator(MEMBER_ACCESS_OP, Variable("z"), [Parameter("x"), Field("y")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_function_call(self):
        code = """
        def foo(x, y):
            z = bar(x, y, gbl, 3)
        """

        fn = self.parse(code)

        correct = [
            FunctionVarOperator("bar", Variable("z"), [Parameter("x"), Parameter("y"), GlobalVariable("gbl"), IntegerConstant("3")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_method_call(self):
        code = """
        def foo(x):
            x.bar()
        """

        fn = self.parse(code)

        correct  = [
            VarOperator(MEMBER_ACCESS_OP, Variable("t0"), [Parameter("x"), Field("bar")]),
            FunctionVarOperator(Variable("t0"), Variable("t1"), [])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_function_with_keyword_arguments(self):
        code = """
        def foo(x, y):
            myfn(x, y, total=x+y, bound=mylib.max)
        """

        fn = self.parse(code)

        correct = [
            VarOperator("+", Variable("t0"), [Parameter("x"), Parameter("y")]),
            VarOperator(MEMBER_ACCESS_OP, Variable("t1"), [GlobalVariable("mylib"), Field("max")]),
            FunctionVarOperator(MEMBER_ACCESS_OP, Variable("t2"), [Parameter("x"), Parameter("y")], {"total": Variable("t0"), "bound": Variable("t1")})
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_pattern_list_assignment(self):
        code = """
        def foo(l):
            a, b, c = get_tuple(l)
        """

        fn = self.parse(code)

        correct = [
            FunctionVarOperator("get_tuple", Variable("t0"), [Parameter("l")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("a"), [Variable("t0")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("b"), [Variable("t0")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("c"), [Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_pattern_list_with_internal_tuples(self):
        code = """
        def foo(t):
            a, (b, (c, d), e) = t
        """

        fn = self.parse(code)

        correct = [
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("a"), [Parameter("t")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("b"), [Parameter("t")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("c"), [Parameter("t")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("d"), [Parameter("t")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("e"), [Parameter("t")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_for_with_tuple_pattern(self):
        code = """
        def foo(items):
            for (a, b) in items:
                print(a, b)
        """

        fn = self.parse(code)

        loop_condition = [
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("t0"), [Parameter("items")]),
            VarOperator(LOOP_OP, None, [Variable("t0")])
        ]

        loop_body = [
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("a"), [Variable("t0")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("b"), [Variable("t0")]),
            FunctionVarOperator("print", Variable("t1"), [Variable("a"), Variable("b")])
        ]

        self.assertContentsEqual(fn.basic_blocks[1], loop_condition)
        self.assertContentsEqual(fn.basic_blocks[2], loop_body)

    def test_pattern_list_with_expression_lvals(self):
        code = """
        def foo(l):
            l[0], l[1] = (2, 3)
        """

        fn = self.parse(code)

        correct = [
            VarOperator(TUPLE_INITIALIZER_OP, Variable("t0"), [IntegerConstant("2"), IntegerConstant("3")]),
            VarOperator(SUBSCRIPT_OP, Variable("t1"), [Parameter("l"), IntegerConstant("0")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("t2"), [Variable("t0")]),
            VarOperator(STORE_OP, Variable("t1"), [Variable("t1"), Variable("t2")]),
            VarOperator(SUBSCRIPT_OP, Variable("t3"), [Parameter("l"), IntegerConstant("1")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("t4"), [Variable("t0")]),
            VarOperator(STORE_OP, Variable("t3"), [Variable("t3"), Variable("t4")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_for_loop(self):
        code = """
        def foo(l):
            for item in l:
                print(item)
        """

        fn = self.parse(code)

        control_block = [
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("item"), [Parameter("l")]),
            VarOperator(LOOP_OP, None, [Variable("item")])
        ]

        body_block = [
            FunctionVarOperator("print", Variable("t0"), [Variable("item")])
        ]

        self.assertContentsEqual(fn.basic_blocks[1], control_block)
        self.assertContentsEqual(fn.basic_blocks[2], body_block)
    
    def test_pattern_list_for_loop(self):
        code = """
        def foo(d):
            for k, v in d.items():
                save(k, v)
        """

        fn = self.parse(code)

        entry_block = [
            VarOperator(MEMBER_ACCESS_OP, Variable("t0"), [Parameter("d"), Field("items")]),
            FunctionVarOperator(Variable("t0"), Variable("t1"), [])
        ]

        control_block = [
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("t2"), [Variable("t1")]),
            VarOperator(LOOP_OP, None, [Variable("t2")])
        ]

        body_block = [
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("k"), [Variable("t2")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("v"), [Variable("t2")]),
            FunctionVarOperator("save", Variable("t3"), [Variable("k"), Variable("v")])
        ]

        self.assertContentsEqual(fn.entry_block, entry_block)
        self.assertContentsEqual(fn.basic_blocks[1], control_block)
        self.assertContentsEqual(fn.basic_blocks[2], body_block)
    
    def test_while_loop(self):
        code = """
        def foo(l):
            while l is not None:
                l = l.next
                print(l)
        """

        fn = self.parse(code)

        control_block = [
            VarOperator("!=", Variable("t0"), [Parameter("l"), NullLiteral()]),
            VarOperator(LOOP_OP, None, [Variable("t0")])
        ]

        body_block = [
            VarOperator(MEMBER_ACCESS_OP, Parameter("l"), [Parameter("l"), Field("next")]),
            FunctionVarOperator("print", Variable("t1"), [Parameter("l")])
        ]

        self.assertContentsEqual(fn.basic_blocks[1], control_block)
        self.assertContentsEqual(fn.basic_blocks[2], body_block)

    def test_slice(self):
        code = """
        def foo(l):
            l = l[:]
            l = l[1:]
            l = l[:2]
            l = l[3:4]
            l = l[::5]
            l = l[6:7:8]
        """

        fn = self.parse(code)

        correct = [
            VarOperator(SLICE_OP,  Variable("t0"), [NullLiteral(), NullLiteral(), NullLiteral()]),
            VarOperator(SUBSCRIPT_OP, Parameter("l"), [Parameter("l"), Variable("t0")]),
            VarOperator(SLICE_OP,  Variable("t1"), [IntegerConstant("1"), NullLiteral(), NullLiteral()]),
            VarOperator(SUBSCRIPT_OP, Parameter("l"), [Parameter("l"), Variable("t1")]),
            VarOperator(SLICE_OP,  Variable("t2"), [NullLiteral(), IntegerConstant("2"), NullLiteral()]),
            VarOperator(SUBSCRIPT_OP, Parameter("l"), [Parameter("l"), Variable("t2")]),
            VarOperator(SLICE_OP,  Variable("t3"), [IntegerConstant("3"), IntegerConstant("4"), NullLiteral()]),
            VarOperator(SUBSCRIPT_OP, Parameter("l"), [Parameter("l"), Variable("t3")]),
            VarOperator(SLICE_OP,  Variable("t4"), [NullLiteral(), NullLiteral(), IntegerConstant("5")]),
            VarOperator(SUBSCRIPT_OP, Parameter("l"), [Parameter("l"), Variable("t4")]),
            VarOperator(SLICE_OP,  Variable("t5"), [IntegerConstant("6"), IntegerConstant("7"), IntegerConstant("8")]),
            VarOperator(SUBSCRIPT_OP, Parameter("l"), [Parameter("l"), Variable("t5")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_boolean_operator(self):
        code  = """
        def foo(a, b):
            c = a and b
        """

        fn = self.parse(code)

        correct = [
            VarOperator("&&", Variable("c"), [Parameter("a"), Parameter('b')])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_not_operator(self):
        code  = """
        def foo(a):
            if not a:
                print("done")
        """

        fn = self.parse(code)

        correct = [
            VarOperator(NOT_OP, Variable("t0"), [Parameter("a")]),
            VarOperator(IF_OP, None, [Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_comparison_operator(self):
        code = """
        def foo(a, b, c, d):
            return a < b and d is not None and c not in d
        """

        fn = self.parse(code)

        correct = [
            VarOperator("<", Variable("t0"), [Parameter("a"), Parameter("b")]),
            VarOperator("!=", Variable("t1"), [Parameter("d"), NullLiteral()]),
            VarOperator("&&", Variable("t2"), [Variable("t0"), Variable("t1")]),
            VarOperator(MEMBERSHIP_OP, Variable("t3"), [Parameter("c"), Parameter("d")]),
            VarOperator("!", Variable("t4"), [Variable("t3")]),
            VarOperator("&&", Variable("t5"), [Variable("t2"), Variable("t4")]),
            VarOperator(RETURN_OP, None, [Variable("t5")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_compound_comparison_operator(self):
        code = """
        def foo(x):
           return 0 < x < 10
        """

        fn = self.parse(code)

        correct = [
            VarOperator("<", Variable("t0"), [IntegerConstant("0"), Parameter("x")]),
            VarOperator("<", Variable("t1"), [Parameter("x"), IntegerConstant("10")]),
            VarOperator("&&", Variable("t2"), [Variable("t0"), Variable("t1")]),
            VarOperator(RETURN_OP, None, [Variable("t2")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_yield(self):
        code = """
        def foo(y):
            for x in y:
                yield x
        """

        fn = self.parse(code)

        loop_condition = [
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("x"), [Parameter("y")]),
            VarOperator(LOOP_OP, None, [Variable("x")])
        ]

        loop_body = [
            VarOperator(YIELD_OP, Variable("t0"), [Variable("x")])
        ]

        self.assertContentsEqual(fn.basic_blocks[1], loop_condition)
        self.assertContentsEqual(fn.basic_blocks[2], loop_body)


    
    def test_subscript_read(self):
        code = """
        def foo(arr):
            x = arr[2]
        """

        fn = self.parse(code)

        correct = [
            VarOperator(SUBSCRIPT_OP, Variable("x"), [Parameter("arr"), IntegerConstant("2")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_subscript_read_write(self):
        code = """
        def foo(a, b):
           c = a[b]
           c[1] = gbl[1]
        """

        fn = self.parse(code)

        correct = [
            VarOperator(SUBSCRIPT_OP, Variable("c"), [Parameter("a"), Parameter("b")]),
            VarOperator(SUBSCRIPT_OP, Variable("t0"), [Variable("c"), IntegerConstant("1")]),
            VarOperator(SUBSCRIPT_OP, Variable("t1"), [GlobalVariable("gbl"), IntegerConstant("1")]),
            VarOperator(STORE_OP, Variable("t0"), [Variable("t0"), Variable("t1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_dict(self):
        code = """
        def foo():
            d = {1:"one", 2 + 3:"five"}
        """

        fn = self.parse(code)

        correct = [
            VarOperator("+", Variable("t0"), [IntegerConstant("2"), IntegerConstant("3")]),
            VarOperator(DICTIONARY_INITIALIZER_OP, Variable("d"), [IntegerConstant("1"), StringLiteral("\"one\""), Variable("t0"), StringLiteral("\"five\"")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_list(self):
        code = """
        def foo():
            l = [True, False, not True]
        """

        fn = self.parse(code)

        correct = [
            VarOperator("!", Variable("t0"), [BoolLiteral("true")]),
            VarOperator(ARRAY_INITIALIZER_OP, Variable("l"), [BoolLiteral("true"), BoolLiteral("false"), Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_set(self):
        code = """
        def foo():
            a = {1, 2, 3 + 4}
        """

        fn = self.parse(code)

        correct = [
            VarOperator("+", Variable("t0"), [IntegerConstant("3"), IntegerConstant("4")]),
            VarOperator(SET_INITIALIZER_OP, Variable("a"), [IntegerConstant("1"), IntegerConstant("2"), Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_assert(self):
        code = """
        def foo(x):
            assert x < 10
            assert x >= 0, "x can't be negative"
        """

        fn = self.parse(code)

        correct = [
            VarOperator("<", Variable("t0"), [Parameter("x"), IntegerConstant("10")]),
            FunctionVarOperator("assert", None, [Variable("t0")]),
            VarOperator(">=", Variable("t1"), [Parameter("x"), IntegerConstant("0")]),
            FunctionVarOperator("assert", None, [Variable("t1"), StringLiteral("\"x can't be negative\"")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_dictionary_comprehension(self):
        code = """
        def foo(x):
            d =  {k + m: v + 1 for k, v in get_dict(x) if k is not None for m in range(3)}
            print(d)
        """

        fn = self.parse(code)

        entry = [
            VarOperator(DICTIONARY_INITIALIZER_OP, Variable("d"), []),
            FunctionVarOperator("get_dict", Variable("t0"), [Parameter("x")])
        ]

        outer_for_condition = [
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("t1"), [Variable("t0")]),
            VarOperator(LOOP_OP, None, [Variable("t1")])
        ]

        if_condition = [
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("k"), [Variable("t1")]),
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("v"), [Variable("t1")]),
            VarOperator("!=", Variable("t2"), [Variable("k"), NullLiteral()]),
            VarOperator(IF_OP, None, [Variable("t2")])
        ]

        if_body = [
            FunctionVarOperator("range", Variable("t3"), [IntegerConstant("3")])
        ]

        inner_for_condition = [
            FunctionVarOperator(ITERATOR_FN_NAME, Variable("m"), [Variable("t3")]),
            VarOperator(LOOP_OP, None, [Variable("m")])
        ]

        comprehension_body = [
            VarOperator("+", Variable("t4"), [Variable("k"), Variable("m")]),
            VarOperator("+", Variable("t5"), [Variable("v"), IntegerConstant("1")]),
            VarOperator(SUBSCRIPT_OP, Variable("t6"), [Variable("d"), Variable("t4")]),
            VarOperator(STORE_OP, Variable("t6"), [Variable("t6"), Variable("t5")])
        ]

        terminating_block = [
            FunctionVarOperator("print", Variable("t0"), [Variable("d")])
        ]

        self.assertContentsEqual(fn.entry_block, entry)
        self.assertContentsEqual(fn.basic_blocks[1], outer_for_condition)
        self.assertContentsEqual(fn.basic_blocks[2], if_condition)
        self.assertContentsEqual(fn.basic_blocks[3], if_body)
        self.assertContentsEqual(fn.basic_blocks[4], inner_for_condition)
        self.assertContentsEqual(fn.basic_blocks[5], comprehension_body)
        self.assertContentsEqual(fn.basic_blocks[6], terminating_block)
    
    def test_ternary_expression(self):
        code = """
        def foo(c):
            z = g + 2 if c else 0
        """

        fn = self.parse(code)

        if_condition = [
            VarOperator(IF_OP, None, [Parameter("c")]),
        ]
        
        true_block = [
            VarOperator("+", Variable("t0"), [GlobalVariable("g"), IntegerConstant("2")]),
            VarOperator(COPY_OP, Variable("z"), [Variable("t0")]) # Copy here is not ideal but reflects the implementation
        ]

        false_block = [
            VarOperator(COPY_OP, Variable("z"), [IntegerConstant("0")])
        ]

        self.assertContentsEqual(fn.entry_block, if_condition)
        self.assertContentsEqual(fn.basic_blocks[1], true_block)
        self.assertContentsEqual(fn.basic_blocks[2], false_block)
    
    def test_decorators(self):
        code = """
        @functools.cache
        @property
        def myfn(self):
            return self._fn
        """

        fn = self.parse(code)

        correct = [
            VarOperator(MEMBER_ACCESS_OP, Variable("t0"), [Parameter("self"), Field("_fn")]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_args_kwargs(self):
        code = """
        def myfn(a, *args, **kwargs):
            return (a, args, kwargs)
        """

        fn = self.parse(code)

        correct = [
            VarOperator(TUPLE_INITIALIZER_OP, Variable("t0"), [Parameter("a"), Parameter("args"), Parameter("kwargs")]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_expression_list(self):
        code = """
        def myfn(a, *args, **kwargs):
            return a, args, kwargs
        """

        fn = self.parse(code)

        correct = [
            VarOperator(TUPLE_INITIALIZER_OP, Variable("t0"), [Parameter("a"), Parameter("args"), Parameter("kwargs")]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_separators(self):
        code = """
        def fn(a, /, b, *, c=None):
            return a, b, c
        """

        fn = self.parse(code)

        correct = [
            VarOperator(TUPLE_INITIALIZER_OP, Variable("t0"), [Parameter("a"), Parameter("b"), Parameter("c")]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_lambdas(self):
        code = """
        def mylambas(self):
            call(
                lambda x, y: x + y,
                lambda: self._fn 
            )
        """

        fn = self.parse(code)

        correct = [
            FunctionVarOperator("call", Variable("t0"), [Lambda(2), Lambda(0)])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_string_concatenation(self):
        code = """
        def fn():
            return "a" "b"
        """

        fn = self.parse(code)

        correct = [
            VarOperator("+", Variable("t0"), [StringLiteral("\"a\""), StringLiteral("\"b\"")]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)

    def test_ellipsis_literal(self):
        code = """
        def fn(x):
            return x is ...
        """

        fn = self.parse(code)

        correct = [
            VarOperator("is", Variable("t0"), [Parameter("x"), Ellipsis()]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_global_statement(self):
        code = """
        def increment():
            global gbl
            gbl += 1
        """

        fn = self.parse(code)

        correct = [
            VarOperator("+", GlobalVariable("gbl"), [GlobalVariable("gbl"), IntegerConstant("1")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_raise(self):
        code = """
        def fn(x):
            try:
                return x[0]
            except:
                raise
        """

        fn = self.parse(code)

        try_block = [
            VarOperator(SUBSCRIPT_OP, Variable("t0"), [Parameter("x"), IntegerConstant("0")]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        error_handling_block = [
            VarOperator(RAISE_OP, None, [])
        ]

        self.assertContentsEqual(fn.basic_blocks[0], [])
        self.assertContentsEqual(fn.basic_blocks[1], try_block)
        self.assertContentsEqual(fn.basic_blocks[2], error_handling_block)
    
    def test_raise_exception(self):
        code = """
        def fn():
            raise NotImplementedError()
        """

        fn = self.parse(code)

        correct = [
            FunctionVarOperator("NotImplementedError", Variable("t0"), []),
            VarOperator(RAISE_OP, None, [Variable("t0")])
        ]

        self.assertContentsEqual(fn.entry_block, correct)
    
    def test_except_identifier(self):
        code = """
        def fn(x):
            try:
                return x[0]
            except IndexError:
                print("Error")
        """

        fn = self.parse(code)

        try_block = [
            VarOperator(SUBSCRIPT_OP, Variable("t0"), [Parameter("x"), IntegerConstant("0")]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        except_block = [
            VarOperator(CATCH_OP, Variable("t1"), [ExceptionName("IndexError")])
        ]

        error_handling_block = [
            FunctionVarOperator("print", Variable("t2"), [StringLiteral("\"Error\"")])
        ]

        raise_block = [
            VarOperator(RAISE_OP, None, [])
        ]

        self.assertContentsEqual(fn.basic_blocks[0], [])
        self.assertContentsEqual(fn.basic_blocks[1], try_block)
        self.assertContentsEqual(fn.basic_blocks[2], except_block)
        self.assertContentsEqual(fn.basic_blocks[3], error_handling_block)
        self.assertContentsEqual(fn.basic_blocks[4], raise_block)
    
    def test_except_as(self):
        code = """
        def fn(x):
            try:
                return x[0]
            except IndexError as e:
                print(e)
        """

        fn = self.parse(code)

        try_block = [
            VarOperator(SUBSCRIPT_OP, Variable("t0"), [Parameter("x"), IntegerConstant("0")]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        except_block = [
            VarOperator(CATCH_OP, Variable("e"), [ExceptionName("IndexError")])
        ]

        error_handling_block = [
            FunctionVarOperator("print", Variable("t1"), [Variable("e")])
        ]

        raise_block = [
            VarOperator(RAISE_OP, None, [])
        ]

        self.assertContentsEqual(fn.basic_blocks[0], [])
        self.assertContentsEqual(fn.basic_blocks[1], try_block)
        self.assertContentsEqual(fn.basic_blocks[2], except_block)
        self.assertContentsEqual(fn.basic_blocks[3], error_handling_block)
        self.assertContentsEqual(fn.basic_blocks[4], raise_block)
    
    def test_except_tuple(self):
        code = """
        def fn(x):
            try:
                return x[0]
            except (IndexError, AttributeError):
                print("Error")
        """

        fn = self.parse(code)

        try_block = [
            VarOperator(SUBSCRIPT_OP, Variable("t0"), [Parameter("x"), IntegerConstant("0")]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        except_block = [
            VarOperator(TUPLE_INITIALIZER_OP, Variable("t1"), [ExceptionName("IndexError"), ExceptionName("AttributeError")]),
            VarOperator(CATCH_OP, Variable("t2"), [Variable("t1")])
        ]

        error_handling_block = [
            FunctionVarOperator("print", Variable("t3"), [StringLiteral("\"Error\"")])
        ]

        raise_block = [
            VarOperator(RAISE_OP, None, [])
        ]

        self.assertContentsEqual(fn.basic_blocks[0], [])
        self.assertContentsEqual(fn.basic_blocks[1], try_block)
        self.assertContentsEqual(fn.basic_blocks[2], except_block)
        self.assertContentsEqual(fn.basic_blocks[3], error_handling_block)
        self.assertContentsEqual(fn.basic_blocks[4], raise_block)

    def test_except_tuple_as(self):
        code = """
        def fn(x):
            try:
                return x[0]
            except (IndexError, AttributeError) as e:
                print(e)
        """

        fn = self.parse(code)

        try_block = [
            VarOperator(SUBSCRIPT_OP, Variable("t0"), [Parameter("x"), IntegerConstant("0")]),
            VarOperator(RETURN_OP, None, [Variable("t0")])
        ]

        except_block = [
            VarOperator(TUPLE_INITIALIZER_OP, Variable("t1"), [ExceptionName("IndexError"), ExceptionName("AttributeError")]),
            VarOperator(CATCH_OP, Variable("e"), [Variable("t1")])
        ]

        error_handling_block = [
            FunctionVarOperator("print", Variable("t2"), [Variable("e")])
        ]

        raise_block = [
            VarOperator(RAISE_OP, None, [])
        ]

        self.assertContentsEqual(fn.basic_blocks[0], [])
        self.assertContentsEqual(fn.basic_blocks[1], try_block)
        self.assertContentsEqual(fn.basic_blocks[2], except_block)
        self.assertContentsEqual(fn.basic_blocks[3], error_handling_block)
        self.assertContentsEqual(fn.basic_blocks[4], raise_block)


        