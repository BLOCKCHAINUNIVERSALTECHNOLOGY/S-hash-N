import os
import tempfile
import unittest

from helpers import lines, run
from sn.errors import SNRuntimeError, SNSyntaxError


class AutogradTests(unittest.TestCase):
    def test_simple_backward(self):
        out = lines("let w = Var([1.0, 2.0, 3.0])\nlet y = (w * w).sum()\ny.backward()\nprint(w.grad)")
        self.assertEqual(out, ["tensor([2.0, 4.0, 6.0])"])

    def test_scalar_autograd_matches_calculus(self):
        self.assertEqual(lines("print(autograd(x => x ** 3 + 2 * x, 2.0))"), ["14.0"])

    def test_matches_numerical_grad_for_many_functions(self):
        funcs = [
            "w => sum(exp(w) * w)",
            "w => sum(tanh(w) + sigmoid(w))",
            "w => sum(sqrt(w * w + 1.0))",
            "w => sum(log(w * w + 1.0))",
            "w => sum(sin(w) * cos(w))",
            "w => sum(w ** 3 - 2 * w / (w * w + 2.0))",
            "w => mean(relu(w - 0.3) ** 2)",
            "w => sum(abs(w - 0.5))",
            "w => sum(softmax(w) * [1.0, 2.0, 3.0])",
            "w => norm(w)",
        ]
        for f in funcs:
            with self.subTest(f=f):
                src = (f"let f = {f}\nlet x = [0.7, -0.2, 1.3]\n"
                       "print(allclose(autograd(f, x), grad(f, x), 1e-4))")
                self.assertEqual(lines(src), ["true"])

    def test_matmul_gradients_and_broadcast(self):
        src = """
let X = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
let f = W => sum((X @ W + 0.5) ** 2)
let W0 = tensor([[0.1, 0.2], [0.3, -0.4]])
print(allclose(autograd(f, W0), grad(f, W0), 1e-3))
let g = b => mean((X + b) ** 2)
print(allclose(autograd(g, [1.0, -1.0]), grad(g, [1.0, -1.0]), 1e-4))
let h = v => sum(X @ v)
print(autograd(h, [1.0, 1.0]))
"""
        self.assertEqual(lines(src), ["true", "true", "tensor([9.0, 12.0])"])

    def test_indexing_and_reshape(self):
        src = """
let w = Var([[1.0, 2.0], [3.0, 4.0]])
let y = w[0, 1] * 3 + w[1].sum() + w.reshape(4)[0] + w.T[0, 1]
y.backward()
print(w.grad.tolist())
"""
        self.assertEqual(lines(src), ["[[1.0, 3.0], [2.0, 1.0]]"])

    def test_gradient_accumulates_and_zero_grad(self):
        src = """
let w = Var(2.0)
(w * w).backward()
(w * w).backward()
print(w.grad.item())
w.zero_grad()
print(w.grad)
"""
        self.assertEqual(lines(src), ["8.0", "nil"])

    def test_var_used_twice(self):
        # d/dx (x*x + x) = 2x + 1
        self.assertEqual(lines("let x = Var(3.0)\nlet y = x * x + x\ny.backward()\nprint(x.grad.item())"), ["7.0"])

    def test_detach_and_constants(self):
        src = "let x = Var(3.0)\nlet y = x.detach() * x\ny.backward()\nprint(x.grad.item())"
        self.assertEqual(lines(src), ["3.0"])

    def test_non_scalar_backward_is_an_error(self):
        with self.assertRaises(SNRuntimeError):
            run("let w = Var([1.0, 2.0])\n(w * 2).backward()")

    def test_value_and_grad_and_format(self):
        self.assertEqual(lines("let [v, g] = value_and_grad(x => x * x, 3.0)\nprint(v, g)"), ["9.0 6.0"])
        self.assertEqual(lines("let a = Var(1.5)\nprint(f\"{a * 2:.2f}\")"), ["3.00"])

    def test_sgd_and_adam_minimise(self):
        for opt in ("SGD([w], lr=0.1)", "SGD([w], lr=0.05, momentum=0.9)", "Adam([w], lr=0.3)"):
            with self.subTest(opt=opt):
                src = f"""
let w = Var([5.0, -3.0])
let opt = {opt}
for i in range(200) {{
    let loss = ((w - [1.0, 2.0]) ** 2).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
}}
print(allclose(w.value, [1.0, 2.0], 1e-2))
"""
                self.assertEqual(lines(src), ["true"])

    def test_train_xor(self):
        src = """
seed(1)
let X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
let y = tensor([[0], [1], [1], [0]])
let W1 = Var(randn(2, 8) * 0.8)
let b1 = Var(zeros(8))
let W2 = Var(randn(8, 1) * 0.8)
let b2 = Var(zeros(1))
let opt = Adam([W1, b1, W2, b2], lr=0.05)
for e in range(300) {
    let pred = sigmoid(tanh(X @ W1 + b1) @ W2 + b2)
    let loss = ((pred - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
}
print(equal(sigmoid(tanh(X @ W1 + b1) @ W2 + b2).value.round(), y))
"""
        self.assertEqual(lines(src), ["true"])

    def test_unsupported_operator_message(self):
        with self.assertRaises(SNRuntimeError):
            run("print(Var(3.0) // 2)")


class TreeTests(unittest.TestCase):
    DATA = """
let iris = read_csv("examples/data/iris.csv")
let X = iris.select("sepal_length", "sepal_width", "petal_length", "petal_width")
let y = iris["species"]
"""

    def run_iris(self, body):
        from helpers import ROOT
        return lines(self.DATA + body) if False else run(self.DATA + body, cwd=ROOT).strip().splitlines()

    def test_tree_fits_and_explains(self):
        out = self.run_iris("""
let t = DecisionTree(max_depth=3, seed=1).fit(X, y)
print(t.score(X, y) > 0.95, t.depth <= 3, t.n_leaves >= 3)
print(t.predict([[5.0, 3.4, 1.5, 0.2], [6.5, 3.0, 5.5, 2.0]]))
print(t.rules().split("\\n")[0])
print(round(t.feature_importances.sum(), 6))
""")
        self.assertEqual(out[0], "true true true")
        self.assertEqual(out[1], '["setosa", "virginica"]')
        self.assertTrue(out[2].startswith("if petal_length <="))
        self.assertEqual(out[3], "1.0")

    def test_unlimited_tree_memorises(self):
        out = self.run_iris('print(DecisionTree().fit(X, y).score(X, y))')
        self.assertEqual(out, ["1.0"])

    def test_xor_needs_two_levels(self):
        src = """
let X = [[0,0],[0,1],[1,0],[1,1]] * 5
let y = [0,1,1,0] * 5
print(DecisionTree().fit(X, y).score(X, y))
print(DecisionTree(max_depth=1).fit(X, y).score(X, y) < 1)
"""
        self.assertEqual(lines(src), ["1.0", "true"])

    def test_regression_tree_step_function(self):
        src = """
let X = [[i] for i in range(20)]
let y = [1.0 if i < 10 else 5.0 for i in range(20)]
let t = DecisionTree(task="regression", max_depth=1).fit(X, y)
print(t.predict([[2], [15]]))
print(t.rules())
"""
        out = lines(src)
        self.assertEqual(out[0], "tensor([1.0, 5.0])")
        self.assertIn("if x0 <= 9.5", out[1])

    def test_proba_rows_sum_to_one(self):
        out = self.run_iris("""
let f = RandomForest(n_trees=15, seed=2).fit(X, y)
let p = f.predict_proba(X)
print(shape(p), allclose(p.sum(axis=1), ones(150), 1e-9))
""")
        self.assertEqual(out, ["[150, 3] true"])

    def test_forest_accuracy_oob_importance(self):
        out = self.run_iris("""
let f = RandomForest(n_trees=40, seed=3).fit(X, y)
print(f.score(X, y) > 0.97, f.oob_score > 0.9, f.n_estimators)
print(f.importance_table()[0][0] in ["petal_length", "petal_width"])
""")
        self.assertEqual(out, ["true true 40", "true"])

    def test_forest_is_reproducible_with_seed(self):
        out = self.run_iris("""
let a = RandomForest(n_trees=10, seed=9).fit(X, y).predict_proba(X)
let b = RandomForest(n_trees=10, seed=9).fit(X, y).predict_proba(X)
print(equal(a, b))
""")
        self.assertEqual(out, ["true"])

    def test_forest_regression(self):
        from helpers import ROOT
        src = """
let h = read_csv("examples/data/houses.csv")
let X = h.select("area_sqft", "bedrooms", "age_years")
let f = RandomForest(n_trees=20, task="regression", max_depth=5, seed=1).fit(X, h["price_lakh"])
print(f.score(X, h["price_lakh"]) > 0.8)
"""
        self.assertEqual(run(src, cwd=ROOT).strip(), "true")

    def test_save_and_load(self):
        from helpers import ROOT
        with tempfile.TemporaryDirectory() as d:
            for model in ("DecisionTree(max_depth=4)", "RandomForest(n_trees=8, seed=1)"):
                path = os.path.join(d, "m.json").replace("\\", "/")
                out = self.run_iris(f"""
let m = {model}.fit(X, y)
m.save("{path}")
let m2 = load_model("{path}")
print(m2.predict(X) == m.predict(X), type(m2))
""")
                self.assertTrue(out[0].startswith("true"), out)

    def test_errors(self):
        for bad in ('DecisionTree(task="x")', 'DecisionTree(max_depth=0)', 'RandomForest(n_trees=0)'):
            with self.subTest(bad=bad), self.assertRaises(SNRuntimeError):
                run(f"{bad}")
        with self.assertRaises(SNRuntimeError):
            run("DecisionTree().predict([[1]])")
        with self.assertRaises(SNRuntimeError):
            run("DecisionTree(task='regression').fit([[1],[2]], [1,2]).predict_proba([[1]])")


class MatchTests(unittest.TestCase):
    def test_literals_or_and_default(self):
        src = """
fn f(x) { return match x { 0 => "zero", 1 or 2 => "few", -1 => "neg", "a" => "str", true => "t", nil => "n", _ => "other" } }
print(f(0), f(2), f(-1), f("a"), f(true), f(nil), f(9))
"""
        self.assertEqual(lines(src), ["zero few neg str t n other"])

    def test_true_does_not_match_one(self):
        self.assertEqual(lines('print(match true { 1 => "one", true => "bool" })'), ["bool"])
        self.assertEqual(lines('print(match 1 { true => "bool", 1 => "one" })'), ["one"])

    def test_binding_and_guard(self):
        src = """
fn sign(x) { return match x { n if n > 0 => "pos", n if n < 0 => "neg", _ => "zero" } }
print(sign(5), sign(-5), sign(0))
"""
        self.assertEqual(lines(src), ["pos neg zero"])

    def test_guard_ending_in_identifier(self):
        src = "let limit = 10\nprint(match 50 { n if n > limit => \"big\", _ => \"small\" })"
        self.assertEqual(lines(src), ["big"])

    def test_guard_with_lambda_in_parens(self):
        src = 'print(match [1, 2] { l if any(map(l, x => x > 1)) => "has big", _ => "no" })'
        self.assertEqual(lines(src), ["has big"])

    def test_list_patterns(self):
        src = """
fn f(x) { return match x { [] => "empty", [a] => f"one {a}", [a, b] => f"two {a} {b}", [h, ...t] => f"{h} then {len(t)}" } }
print(f([]), f([1]), f([1, 2]), f([1, 2, 3, 4]))
"""
        self.assertEqual(lines(src), ["empty one 1 two 1 2 1 then 3"])

    def test_nested_and_map_patterns(self):
        src = """
let shape = {kind: "rect", size: [3, 4]}
print(match shape {
    {kind: "circle", r} => r
    {kind: "rect", size: [w, h]} => w * h
    _ => 0
})
print(match {"a": 1} { {a} => a + 1 })
"""
        self.assertEqual(lines(src), ["12", "2"])

    def test_block_arms_and_scoping(self):
        src = """
let x = "outer"
let r = match 5 { n => { let x = n * 2
  x + 1 } }
print(r, x)
"""
        self.assertEqual(lines(src), ["11 outer"])

    def test_return_inside_arm(self):
        src = 'fn f(v) { match v { 0 => { return "early" }, _ => nil }\n return "late" }\nprint(f(0), f(1))'
        self.assertEqual(lines(src), ["early late"])

    def test_no_arm_matches_error(self):
        with self.assertRaises(SNRuntimeError) as cm:
            run('match 3 { 1 => "a" }')
        self.assertIn("no match arm", str(cm.exception))

    def test_hindi_keyword(self):
        self.assertEqual(lines('print(मिलान 2 { 1 => "एक", 2 => "दो", _ => "?" })'), ["दो"])

    def test_syntax_errors(self):
        for bad in ("match 1 { }", "match 1 { [a, ...r, b] => 1 }", "match 1 { 1 2 }"):
            with self.subTest(bad=bad), self.assertRaises(SNSyntaxError):
                run(bad)


if __name__ == "__main__":
    unittest.main()
