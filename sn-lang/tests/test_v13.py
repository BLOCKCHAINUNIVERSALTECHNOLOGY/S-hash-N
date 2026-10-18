import os
import tempfile
import unittest

from helpers import ROOT, run
from sn.errors import SNRuntimeError

DATA = """
let iris = read_csv("examples/data/iris.csv")
let X = iris.select("sepal_length", "sepal_width", "petal_length", "petal_width")
let y = iris["species"]
"""


def sn(body):
    return run(DATA + body, cwd=ROOT).strip().splitlines()


class PCATests(unittest.TestCase):
    def test_variance_ratio_and_shapes(self):
        out = sn("""
let p = PCA(n_components=2).fit(X)
print(shape(p.transform(X)), shape(p.components))
print(p.explained_variance_ratio.sum() > 0.97, p.explained_variance_ratio[0] > 0.9)
""")
        self.assertEqual(out, ["[150, 2] [2, 4]", "true true"])

    def test_matches_numpy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        import csv
        rows = list(csv.reader(open(os.path.join(ROOT, "examples/data/iris.csv"))))[1:]
        A = np.array([[float(v) for v in r[:4]] for r in rows])
        w = np.sort(np.linalg.eigvalsh(np.cov(A.T)))[::-1]
        out = sn("print(PCA(4).fit(X).explained_variance.tolist())")
        got = eval(out[0])
        for g, e in zip(got, w):
            self.assertAlmostEqual(g, e, places=8)

    def test_fraction_keeps_enough_components(self):
        out = sn("print(PCA(0.95).fit(X).components.shape)")
        self.assertEqual(out, ["[2, 4]"])

    def test_full_rank_round_trip(self):
        out = sn("let p = PCA(4).fit(X)\nprint(allclose(p.inverse_transform(p.transform(X)), X.to_tensor(), 1e-6))")
        self.assertEqual(out, ["true"])

    def test_components_are_orthonormal(self):
        out = sn("let C = PCA(4).fit(X).components\nprint(allclose(C @ C.T, eye(4), 1e-8))")
        self.assertEqual(out, ["true"])

    def test_errors(self):
        for bad in ("PCA(0)", "PCA(1.5)", "PCA(2).fit([[1, 2]])", "PCA(5).fit([[1, 2], [3, 4], [5, 7]])"):
            with self.subTest(bad=bad), self.assertRaises(SNRuntimeError):
                run(bad)


class NaiveBayesTests(unittest.TestCase):
    def test_accuracy_and_probabilities(self):
        out = sn("""
let m = NaiveBayes().fit(X, y)
print(m.score(X, y) > 0.92, m.classes)
print(allclose(m.predict_proba(X).sum(axis=1), ones(150), 1e-9))
print(m.predict([[5.0, 3.4, 1.5, 0.2]]))
""")
        self.assertEqual(out, ["true [\"setosa\", \"versicolor\", \"virginica\"]", "true", '["setosa"]'])

    def test_single_class_is_an_error(self):
        with self.assertRaises(SNRuntimeError):
            run("NaiveBayes().fit([[1], [2]], [0, 0])")


class BoostingTests(unittest.TestCase):
    def test_classification(self):
        out = sn("""
let g = GradientBoosting(n_trees=40, lr=0.2, max_depth=2, seed=1).fit(X, y)
print(g.score(X, y) > 0.97, g.n_estimators)
print(allclose(g.predict_proba(X).sum(axis=1), ones(150), 1e-9))
print(g.feature_importances.sum())
""")
        self.assertEqual(out[0], "true 40")
        self.assertEqual(out[1], "true")
        self.assertAlmostEqual(float(out[2]), 1.0, places=6)

    def test_regression_learns_a_curve(self):
        src = """
let X = [[i / 10] for i in range(60)]
let y = [sin(i / 10) for i in range(60)]
let g = GradientBoosting(n_trees=150, lr=0.2, max_depth=3, task="regression").fit(X, y)
print(g.score(X, y) > 0.98)
"""
        self.assertEqual(run(src).strip(), "true")

    def test_more_rounds_reduce_training_error(self):
        out = sn("""
let few = GradientBoosting(n_trees=3, seed=1).fit(X, y).predict_proba(X)
let many = GradientBoosting(n_trees=60, seed=1).fit(X, y).predict_proba(X)
let target = one_hot(label_encode(y)["codes"], 3)
print(mse(target, many) < mse(target, few))
""")
        self.assertEqual(out, ["true"])

    def test_subsample_and_seed_reproducible(self):
        out = sn("""
let a = GradientBoosting(n_trees=10, subsample=0.6, seed=4).fit(X, y).predict_proba(X)
let b = GradientBoosting(n_trees=10, subsample=0.6, seed=4).fit(X, y).predict_proba(X)
print(equal(a, b))
""")
        self.assertEqual(out, ["true"])

    def test_errors(self):
        for bad in ("GradientBoosting(lr=0)", "GradientBoosting(n_trees=0)", "GradientBoosting(subsample=2)"):
            with self.subTest(bad=bad), self.assertRaises(SNRuntimeError):
                run(bad)


class SaveLoadTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json").replace("\\", "/")
            for model, same in (("PCA(2)", "equal(m.transform(X), m2.transform(X))"),
                                ("NaiveBayes()", "m.predict(X) == m2.predict(X)"),
                                ("GradientBoosting(n_trees=5, seed=1)", "m.predict(X) == m2.predict(X)")):
                with self.subTest(model=model):
                    out = sn(f'let m = {model}.fit(X, y)\nm.save("{path}")\nlet m2 = load_model("{path}")\nprint({same}, type(m2))')
                    self.assertTrue(out[0].startswith("true"), out)


if __name__ == "__main__":
    unittest.main()
