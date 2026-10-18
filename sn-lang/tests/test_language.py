import unittest

from helpers import lines, run
from sn.errors import SNRuntimeError, SNSyntaxError
from sn.lexer import Lexer
from sn.parser import parse


class LexerTests(unittest.TestCase):
    def test_hindi_and_english_keywords_are_equivalent(self):
        a = [(t.type, t.value) for t in Lexer("let x = 1").tokenize()]
        b = [(t.type, t.value) for t in Lexer("मान x = 1").tokenize()]
        self.assertEqual(a, b)

    def test_unicode_identifiers_with_matras(self):
        self.assertEqual(lines("मान संख्या = 3\nprint(संख्या)"), ["3"])

    def test_comments_and_numbers(self):
        self.assertEqual(lines("print(1_000 + 0.5) # comment"), ["1000.5"])

    def test_unterminated_string(self):
        with self.assertRaises(SNSyntaxError):
            parse('print("abc)')

    def test_unmatched_bracket(self):
        with self.assertRaises(SNSyntaxError):
            parse("print((1)")


class ExpressionTests(unittest.TestCase):
    def test_precedence(self):
        self.assertEqual(lines("print(2 + 3 * 4, (2 + 3) * 4, -2 ** 2, 2 ** 3 ** 2)"),
                         ["14 20 -4 512"])

    def test_division_kinds(self):
        self.assertEqual(lines("print(7 / 2, 7 // 2, 7 % 2, -7 // 2)"), ["3.5 3 1 -4"])

    def test_division_by_zero(self):
        with self.assertRaises(SNRuntimeError):
            run("print(1 / 0)")

    def test_logic_returns_operands(self):
        self.assertEqual(lines('print(nil or "x", 0 and 5, not [], 1 < 2 and 2 < 3)'),
                         ["x 0 true true"])

    def test_strings(self):
        self.assertEqual(lines('print("ab" + "cd", "ab" * 3, "x" in "xyz", "hi".upper())'),
                         ["abcd ababab true HI"])

    def test_string_plus_number_is_an_error_with_hint(self):
        with self.assertRaises(SNRuntimeError) as cm:
            run('print("a" + 1)')
        self.assertIn("f-string", cm.exception.message)

    def test_fstring_formats(self):
        self.assertEqual(lines('let x = 3.14159\nprint(f"{x:.2f}|{x*2:8.3f}|{{lit}}|{[1,2][0]}")'),
                         ["3.14|   6.283|{lit}|1"])

    def test_lists_and_slices(self):
        self.assertEqual(lines("let a = [1,2,3,4,5]\nprint(a[1:3], a[::2], a[-1], a[:2] + [9])"),
                         ["[2, 3] [1, 3, 5] 5 [1, 2, 9]"])

    def test_list_index_error(self):
        with self.assertRaises(SNRuntimeError):
            run("print([1][5])")

    def test_maps(self):
        out = lines('let m = {a: 1, "b": 2}\nm.c = 3\nm["d"] = 4\nprint(m, len(m), "a" in m, m.get("z", 0))')
        self.assertEqual(out, ['{"a": 1, "b": 2, "c": 3, "d": 4} 4 true 0'])

    def test_comprehension(self):
        self.assertEqual(lines("print([x*x for x in range(6) if x % 2 == 0])"), ["[0, 4, 16]"])


class StatementTests(unittest.TestCase):
    def test_functions_defaults_kwargs(self):
        src = "fn f(a, b = 10, c = 100) { return a + b + c }\nprint(f(1), f(1, 2), f(1, c = 5))"
        self.assertEqual(lines(src), ["111 103 16"])

    def test_closures(self):
        src = """
fn counter() { let n = 0; return fn() { n += 1; return n } }
let c = counter(); c(); print(c())
"""
        self.assertEqual(lines(src), ["2"])

    def test_recursion(self):
        self.assertEqual(lines("fn f(n) { if n < 2 { return n }; return f(n-1) + f(n-2) }\nprint(f(15))"),
                         ["610"])

    def test_recursion_limit_is_an_error_not_a_crash(self):
        with self.assertRaises(SNRuntimeError):
            run("fn f(n) { return f(n + 1) }\nf(0)")

    def test_loops_break_continue(self):
        src = """
let out = []
for i in range(10) {
    if i == 7 { break }
    if i % 2 == 0 { continue }
    out.push(i)
}
let j = 0
while j < 3 { j += 1 }
print(out, j)
"""
        self.assertEqual(lines(src), ["[1, 3, 5] 3"])

    def test_if_elif_else_both_styles(self):
        src = """
fn grade(n) {
    if n >= 90 { return "A" }
    elif n >= 80 { return "B" }
    else if n >= 70 { return "C" }
    else { return "F" }
}
print(grade(95), grade(85), grade(75), grade(10))
"""
        self.assertEqual(lines(src), ["A B C F"])

    def test_else_on_next_line(self):
        self.assertEqual(lines("if false { print(1) }\nelse { print(2) }"), ["2"])

    def test_unpacking(self):
        src = "let [a, b] = [1, 2]\nlet c, d = [3, 4]\nfor [x, y] in [[5, 6]] { print(a, b, c, d, x, y) }"
        self.assertEqual(lines(src), ["1 2 3 4 5 6"])

    def test_try_catch_throw(self):
        src = """
try { throw {code: 7} } catch e { print(e.code) }
try { let x = [1][3] } catch err { print("caught") }
"""
        self.assertEqual(lines(src), ["7", "caught"])

    def test_pipe_and_lambda(self):
        src = "print(range(1, 6) |> map(x => x * 2) |> filter(x => x > 4) |> sum)"
        self.assertEqual(lines(src), ["24"])

    def test_multiline_pipe(self):
        src = "let r = [3, 1, 2]\n    |> sorted\n    |> reversed\nprint(r)"
        self.assertEqual(lines(src), ["[3, 2, 1]"])

    def test_method_chain_on_next_line(self):
        self.assertEqual(lines('print("  hi  "\n    .strip()\n    .upper())'), ["HI"])

    def test_hindi_program(self):
        src = "कार्य दुगुना(क) { लौटाओ क * 2 }\nअगर दुगुना(4) == 8 { छापो(\"ठीक\") } वरना { छापो(\"गलत\") }"
        self.assertEqual(lines(src), ["ठीक"])

    def test_undefined_variable_suggests(self):
        with self.assertRaises(SNRuntimeError) as cm:
            run("let counter = 1\nprint(countr)")
        self.assertIn("counter", cm.exception.message)

    def test_error_reports_line_and_function(self):
        with self.assertRaises(SNRuntimeError) as cm:
            run("fn bad() {\n  return 1 / 0\n}\nbad()")
        e = cm.exception
        self.assertEqual(e.line, 2)
        self.assertEqual(e.trace[0][0], "bad")

    def test_assign_to_undeclared(self):
        with self.assertRaises(SNRuntimeError):
            run("x = 5")

    def test_import(self):
        import os
        root = os.path.join(os.path.dirname(__file__), "..")
        out = run('import "examples/lib/geometry.sn" as g\nprint(g.rect_area(2, 3))', cwd=root)
        self.assertEqual(out.strip(), "6")


class BuiltinTests(unittest.TestCase):
    def test_conversions(self):
        self.assertEqual(lines('print(int("42"), int(3.9), float("1.5"), str(7) + "!", bool([]))'),
                         ["42 3 1.5 7! false"])

    def test_math(self):
        self.assertEqual(lines("print(round(2.567, 2), floor(2.9), ceil(2.1), abs(-3), sqrt(16), pow(2, 10))"),
                         ["2.57 2 3 3 4.0 1024"])

    def test_collections(self):
        src = 'print(sorted([3,1,2], reverse=true), zip([1,2],[3,4]), enumerate(["a","b"]), unique([1,1,2]), any([0,1]), all([1,0]))'
        self.assertEqual(lines(src), ['[3, 2, 1] [[1, 3], [2, 4]] [[0, "a"], [1, "b"]] [1, 2] true false'])

    def test_stats(self):
        self.assertEqual(lines("print(mean([1,2,3,4]), median([5,1,3]), std([2,4,4,4,5,5,7,9]), percentile([1,2,3,4,5], 50))"),
                         ["2.5 3.0 2.0 3.0"])

    def test_json_roundtrip(self):
        self.assertEqual(lines('print(parse_json(to_json({a: [1, 2]})))'), ['{"a": [1, 2]}'])

    def test_seed_is_reproducible(self):
        src = "seed(5)\nlet a = random()\nseed(5)\nprint(a == random())"
        self.assertEqual(lines(src), ["true"])

    def test_assert(self):
        with self.assertRaises(SNRuntimeError):
            run('assert(1 == 2, "nope")')


if __name__ == "__main__":
    unittest.main()
