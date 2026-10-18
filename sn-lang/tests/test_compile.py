"""Semantics that the closure compiler must preserve."""
import unittest

from helpers import lines, run
from sn.errors import SNRuntimeError


class ScopeTests(unittest.TestCase):
    def test_each_iteration_captures_its_own_variable(self):
        src = "let fs = []\nfor i in range(3) { fs.push(() => i) }\nprint(fs[0](), fs[1](), fs[2]())"
        self.assertEqual(lines(src), ["0 1 2"])

    def test_let_in_loop_body_is_fresh_each_iteration(self):
        src = """
let total = 10
let seen = []
for i in range(3) { let total = total + 1; seen.push(total) }
print(seen, total)
"""
        self.assertEqual(lines(src), ["[11, 11, 11] 10"])

    def test_while_body_scope(self):
        src = "let i = 0\nlet out = []\nwhile i < 3 { let k = i * 2; out.push(k); i += 1 }\nprint(out)"
        self.assertEqual(lines(src), ["[0, 2, 4]"])

    def test_if_block_declarations_do_not_leak(self):
        with self.assertRaises(SNRuntimeError):
            run("if true { let hidden = 1 }\nprint(hidden)")

    def test_loop_variable_does_not_leak(self):
        with self.assertRaises(SNRuntimeError):
            run("for q in [1] { }\nprint(q)")

    def test_comprehension_closures(self):
        src = "let fs = [() => x * 10 for x in range(3)]\nprint(fs[0](), fs[2]())"
        self.assertEqual(lines(src), ["0 20"])

    def test_closure_counter(self):
        src = "fn make() { let n = 0\n return fn() { n += 1\n return n } }\nlet c = make()\nc(); c()\nprint(c())"
        self.assertEqual(lines(src), ["3"])


class ControlFlowTests(unittest.TestCase):
    def test_break_continue_nested(self):
        src = """
let out = []
for i in range(4) {
    for j in range(4) {
        if j == 2 { break }
        if i == 1 { continue }
        out.push([i, j])
    }
}
print(out)
"""
        self.assertEqual(lines(src), ["[[0, 0], [0, 1], [2, 0], [2, 1], [3, 0], [3, 1]]"])

    def test_return_from_nested_loops(self):
        src = """
fn find(xs, t) {
    for i in range(len(xs)) {
        for j in range(len(xs[i])) { if xs[i][j] == t { return [i, j] } }
    }
    return nil
}
print(find([[1, 2], [3, 4]], 4), find([[1]], 9))
"""
        self.assertEqual(lines(src), ["[1, 1] nil"])

    def test_return_inside_while_and_try(self):
        src = """
fn f() {
    let i = 0
    while true {
        i += 1
        try { if i == 3 { return "done " + str(i) } } catch e { return "err" }
    }
}
print(f())
"""
        self.assertEqual(lines(src), ["done 3"])

    def test_continue_in_while(self):
        src = "let i = 0\nlet s = 0\nwhile i < 6 { i += 1; if i % 2 == 0 { continue }; s += i }\nprint(s)"
        self.assertEqual(lines(src), ["9"])

    def test_break_outside_loop_is_an_error(self):
        with self.assertRaises(SNRuntimeError):
            run("break")
        with self.assertRaises(SNRuntimeError):
            run("fn f() { break }\nf()")
        with self.assertRaises(SNRuntimeError):
            run("return 1")

    def test_return_in_match_in_loop(self):
        src = """
fn f(xs) { for x in xs { let r = match x { 2 => { return "two" }, _ => x }
  print(r) }
  return "end" }
print(f([1, 2, 3]))
"""
        self.assertEqual(lines(src), ["1", "two"])

    def test_break_in_match_arm(self):
        src = 'let n = 0\nfor i in range(5) { match i { 3 => { break }, _ => nil }\n n += 1 }\nprint(n)'
        self.assertEqual(lines(src), ["3"])


class ErrorTests(unittest.TestCase):
    def test_error_reports_statement_line(self):
        with self.assertRaises(SNRuntimeError) as cm:
            run("let a = 1\nlet b = 2\nprint(a + nothing)")
        self.assertEqual(cm.exception.line, 3)

    def test_error_inside_function_has_trace(self):
        with self.assertRaises(SNRuntimeError) as cm:
            run("fn boom(x) {\n  return x / 0\n}\nfn outer() {\n  return boom(1)\n}\nouter()")
        self.assertEqual(cm.exception.line, 2)
        self.assertEqual([n for n, _ in cm.exception.trace][:2], ["boom", "outer"])

    def test_catch_gets_message_and_thrown_value(self):
        src = 'try { throw {code: 7} } catch e { print(e.code) }\ntry { 1 / 0 } catch e { print(e) }'
        self.assertEqual(lines(src), ["7", "division by zero"])

    def test_fast_paths_keep_semantics(self):
        self.assertEqual(lines("print(1 + 2, 2.5 * 2, 7 < 8, true + 1, 3 == 3.0, \"a\" + \"b\")"),
                         ["3 5.0 true 2 true ab"])
        self.assertEqual(lines("let x = 5\nx += 2\nx -= 1\nx *= 3\nprint(x)\nlet s = \"a\"\ns += \"b\"\nprint(s)"),
                         ["18", "ab"])

    def test_default_and_keyword_arguments(self):
        src = "fn f(a, b = a * 2, c = 3) { return [a, b, c] }\nprint(f(1), f(1, c = 9), f(1, 2, 3))"
        self.assertEqual(lines(src), ["[1, 2, 3] [1, 2, 9] [1, 2, 3]"])

    def test_deep_recursion_is_reported_not_crashed(self):
        with self.assertRaises(SNRuntimeError):
            run("fn f(n) { return f(n + 1) }\nf(0)")
        self.assertEqual(lines("fn s(n) { if n == 0 { return 0 }\n return n + s(n - 1) }\nprint(s(100))"), ["5050"])


if __name__ == "__main__":
    unittest.main()
