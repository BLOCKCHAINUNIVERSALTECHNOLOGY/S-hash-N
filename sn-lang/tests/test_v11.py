import os
import tempfile
import unittest
import xml.dom.minidom

from helpers import ROOT, lines, run
import sn.tensor as tensor_mod
from sn.errors import SNRuntimeError, SNSyntaxError
from sn.parser import parse


class ClassTests(unittest.TestCase):
    BASE = """
class Animal {
    fn init(name, sound = "...") { self.name = name; self.sound = sound }
    fn speak() { return f"{self.name} says {self.sound}" }
}
class Dog extends Animal {
    fn init(name) { super.init(name, "woof"); self.tricks = [] }
    fn learn(t) { self.tricks.push(t); return self }
    fn speak() { return super.speak() + "!" }
}
"""

    def test_methods_fields_inheritance_super(self):
        src = self.BASE + 'let d = Dog("Rex")\nd.learn("sit").learn("roll")\nprint(d.speak(), d.tricks, type(d))'
        self.assertEqual(lines(src), ['Rex says woof! ["sit", "roll"] Dog'])

    def test_is_a(self):
        src = self.BASE + 'let d = Dog("R")\nprint(is_a(d, Animal), is_a(d, Dog), is_a(Animal("x"), Dog), is_a(3, "int"))'
        self.assertEqual(lines(src), ["true true false true"])

    def test_default_and_custom_repr(self):
        src = self.BASE + 'print(Animal("a"))\nclass P { fn str() { return "custom" } }\nprint(P(), f"{P()}")'
        self.assertEqual(lines(src), ['Animal(name="a", sound="...")', "custom custom"])

    def test_instances_do_not_share_state(self):
        src = self.BASE + 'let a = Dog("a"); let b = Dog("b")\na.learn("x")\nprint(a.tricks, b.tricks)'
        self.assertEqual(lines(src), ['["x"] []'])

    def test_operator_overloading(self):
        src = """
class V {
    fn init(x) { self.x = x }
    fn add(o) { return V(self.x + o.x) }
    fn mul(k) { return V(self.x * k) }
    fn rmul(k) { return self * k }
    fn eq(o) { return self.x == o.x }
    fn lt(o) { return self.x < o.x }
    fn len() { return self.x }
    fn getitem(i) { return i * 10 }
    fn contains(v) { return v == self.x }
}
let a = V(2)
print((a + V(3)).x, (a * 4).x, (5 * a).x, a == V(2), a != V(2), a < V(3), len(a), a[4], 2 in a)
"""
        self.assertEqual(lines(src), ["5 8 10 true false true 2 40 true"])

    def test_missing_operator_is_an_error(self):
        with self.assertRaises(SNRuntimeError) as cm:
            run("class A { fn init() { } }\nA() + A()")
        self.assertIn("add", cm.exception.message)

    def test_unknown_member_suggests(self):
        with self.assertRaises(SNRuntimeError) as cm:
            run("class A { fn init() { self.value = 1 } }\nA().valu")
        self.assertIn("value", cm.exception.message)

    def test_init_arguments_checked(self):
        with self.assertRaises(SNRuntimeError):
            run("class A { fn init(a) { } }\nA()")
        with self.assertRaises(SNRuntimeError):
            run("class A { }\nA(1)")

    def test_extend_non_class(self):
        with self.assertRaises(SNRuntimeError):
            run("class A extends 5 { }")

    def test_only_methods_inside_class(self):
        with self.assertRaises(SNSyntaxError):
            parse("class A { let x = 1 }")

    def test_hindi_class_keywords(self):
        src = "वर्ग गिनती {\n  कार्य init() { खुद.n = 0 }\n  कार्य बढ़ाओ() { खुद.n += 1; लौटाओ खुद.n }\n}\nमान g = गिनती(); g.बढ़ाओ()\nछापो(g.बढ़ाओ())"
        self.assertEqual(lines(src), ["2"])

    def test_error_trace_names_method(self):
        with self.assertRaises(SNRuntimeError) as cm:
            run("class A {\n  fn boom() { return 1 / 0 }\n}\nA().boom()")
        self.assertEqual(cm.exception.trace[0][0], "A.boom")

    def test_instances_in_collections_and_closures(self):
        src = "class C { fn init(n) { self.n = n } }\nlet xs = [C(3), C(1), C(2)]\nprint(sorted(xs, key=c => c.n) |> map(c => c.n))"
        self.assertEqual(lines(src), ["[1, 2, 3]"])


class SyntaxAdditionTests(unittest.TestCase):
    def test_conditional_expression(self):
        src = 'let f = n => "neg" if n < 0 else "pos" if n > 0 else "zero"\nprint(f(-1), f(0), f(2), [1 if v > 2 else 0 for v in range(5)])'
        self.assertEqual(lines(src), ["neg zero pos [0, 0, 0, 1, 1]"])

    def test_conditional_needs_else(self):
        with self.assertRaises(SNSyntaxError):
            parse("let a = 1 if true")

    def test_comprehension_filter_still_works(self):
        self.assertEqual(lines("print([v for v in range(6) if v % 2 == 0])"), ["[0, 2, 4]"])

    def test_hindi_digits(self):
        self.assertEqual(lines("छापो(१२ + ३.५)"), ["15.5"])


class PlotTests(unittest.TestCase):
    def _svg(self, code):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.svg").replace("\\", "/")
            run(code.replace("PATH", p))
            with open(p, encoding="utf-8") as f:
                text = f.read()
        xml.dom.minidom.parseString(text)   # must be well-formed XML
        return text

    def test_line_chart(self):
        svg = self._svg('plot([1, 4, 9], title="T <&>", xlabel="x", ylabel="y").save("PATH")')
        self.assertIn("<polyline", svg)
        self.assertIn("T &lt;&amp;&gt;", svg)

    def test_scatter_with_legend(self):
        svg = self._svg('let f = Plot(); f.scatter([1,2],[3,4], label="A"); f.scatter([2,3],[1,5], label="B"); f.save("PATH")')
        self.assertEqual(svg.count("<circle"), 4)
        self.assertIn(">A<", svg)

    def test_bar_hist_heatmap(self):
        self.assertIn("<rect", self._svg('bar_chart(["a","b"], [3, 5]).save("PATH")'))
        self.assertGreaterEqual(self._svg('histogram([1,2,2,3,3,3], bins=3).save("PATH")').count("<rect"), 4)
        self.assertIn("rgb(", self._svg('heatmap([[1,2],[3,4]], ["a","b"], ["c","d"]).save("PATH")'))

    def test_nan_values_do_not_break_rendering(self):
        self._svg('plot([1, NAN, 3, INF, 5]).save("PATH")')

    def test_html_output_and_bad_extension(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.html").replace("\\", "/")
            run(f'plot([1,2]).save("{p}")')
            self.assertTrue(open(p, encoding="utf-8").read().startswith("<!doctype html>"))
            with self.assertRaises(SNRuntimeError):
                run(f'plot([1,2]).save("{p}.png")')

    def test_validation(self):
        for bad in ('Plot().bar(["a"], [1]).line([1], [1])', 'plot([1, 2], [1])', 'plot(["a"])',
                    'bar_chart(["a", "b"], [1])', 'heatmap([[1], [1, 2]])'):
            with self.subTest(bad=bad), self.assertRaises(SNRuntimeError):
                run(bad)

    def test_save_creates_folders(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a", "b", "c.svg").replace("\\", "/")
            run(f'plot([1,2]).save("{p}")')
            self.assertTrue(os.path.exists(p))


@unittest.skipIf(tensor_mod._np is None, "NumPy is not installed")
class BackendTests(unittest.TestCase):
    def _both(self, code):
        fast = run(code)
        saved, tensor_mod._np = tensor_mod._np, None
        try:
            slow = run(code)
        finally:
            tensor_mod._np = saved
        return fast, slow

    def test_backend_name(self):
        self.assertEqual(lines("print(backend())"), ["numpy"])

    def test_matmul_matches_pure_python(self):
        fast, slow = self._both("seed(3)\nlet a = randn(30, 20)\nlet b = randn(20, 25)\nprint(round((a @ b).sum(), 6), (a @ b).shape)")
        self.assertEqual(fast, slow)

    def test_knn_matches_pure_python(self):
        code = """
seed(4)
let X = randn(120, 4)
let y = [1 if r[0] + r[1] > 0 else 0 for r in X]
let m = KNN(k=5).fit(X, y)
print(m.score(X, y), m.predict(randn(50, 4)) |> sum)
"""
        fast, slow = self._both(code)
        self.assertEqual(fast, slow)

    def test_kmeans_matches_pure_python(self):
        code = "seed(2)\nlet X = randn(200, 3)\nlet km = KMeans(k=3, seed=1).fit(X)\nprint(round(km.inertia, 4), km.labels |> sum)"
        fast, slow = self._both(code)
        self.assertEqual(fast, slow)


if __name__ == "__main__":
    unittest.main()
