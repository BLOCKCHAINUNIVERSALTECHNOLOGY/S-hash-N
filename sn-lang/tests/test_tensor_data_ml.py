import os
import tempfile
import unittest

from helpers import ROOT, lines, run
from sn.errors import SNRuntimeError


class TensorTests(unittest.TestCase):
    def test_broadcast_and_ops(self):
        out = run("print((tensor([[1,2],[3,4]]) + [10, 20]).tolist())")
        self.assertEqual(out.strip(), "[[11.0, 22.0], [13.0, 24.0]]")

    def test_matmul_shapes(self):
        self.assertEqual(lines("print((ones(2,3) @ ones(3,4)).shape, (ones(2,3) @ ones(3)).shape, ones(3) @ ones(3))"),
                         ["[2, 4] [2] tensor(3.0)"])

    def test_matmul_mismatch(self):
        with self.assertRaises(SNRuntimeError):
            run("ones(2,3) @ ones(2,3)")

    def test_broadcast_mismatch(self):
        with self.assertRaises(SNRuntimeError):
            run("ones(2,3) + ones(4)")

    def test_reductions(self):
        src = "let t = tensor([[1,2,3],[4,5,6]])\nprint(t.sum(), t.sum(axis=0).tolist(), t.mean(axis=1).tolist(), t.argmax(), t.max(axis=1).tolist())"
        self.assertEqual(lines(src), ["21.0 [5.0, 7.0, 9.0] [2.0, 5.0] 5 [3.0, 6.0]"])

    def test_indexing_and_mask(self):
        src = """
let t = tensor([[1,2,3],[4,5,6]])
print(t[1, 2], t[:, 0].tolist(), t[0].tolist(), t[-1, -1])
let v = tensor([1, -2, 3])
print(v[v > 0].tolist())
v[v < 0] = 0
print(v.tolist())
t[0, 0] = 99
print(t[0, 0])
"""
        self.assertEqual(lines(src), ["6.0 [1.0, 4.0] [1.0, 2.0, 3.0] 6.0", "[1.0, 3.0]", "[1.0, 0.0, 3.0]", "99.0"])

    def test_reshape_transpose(self):
        src = "let t = arange(6).reshape(2, -1)\nprint(t.shape, t.T.shape, t.T.tolist()[0])"
        self.assertEqual(lines(src), ["[2, 3] [3, 2] [0.0, 3.0]"])

    def test_linear_algebra(self):
        src = """
let A = tensor([[2, 1], [1, 3]])
print(allclose(A @ solve(A, [3, 5]), [3, 5]), allclose(A @ inv(A), eye(2)), det(A))
"""
        self.assertEqual(lines(src), ["true true 5.0"])

    def test_singular_matrix(self):
        with self.assertRaises(SNRuntimeError):
            run("inv(tensor([[1, 2], [2, 4]]))")

    def test_softmax_sums_to_one(self):
        self.assertEqual(lines("print(round(softmax(tensor([[1,2,3],[1,1,1]])).sum(axis=1)[0], 6))"), ["1.0"])

    def test_concat_stack_where(self):
        src = "print(concat([ones(2,2), zeros(1,2)]).shape, stack([ones(2), zeros(2)]).shape, where(tensor([1,0,1]) > 0, 5, 7).tolist())"
        self.assertEqual(lines(src), ["[3, 2] [2, 2] [5.0, 7.0, 5.0]"])

    def test_ambiguous_truth_value(self):
        with self.assertRaises(SNRuntimeError):
            run("if ones(3) { print(1) }")


class DataFrameTests(unittest.TestCase):
    def setUp(self):
        self.src = """
let df = DataFrame({name: ["a","b","c","d"], x: [1, 2, 3, 4], y: [10.0, nil, 30.0, 40.0]})
"""

    def test_basics(self):
        self.assertEqual(lines(self.src + "print(df.shape, df.columns, len(df), df[1])"),
                         ['[4, 3] ["name", "x", "y"] 4 {"name": "b", "x": 2, "y": nil}'])

    def test_filter_sort_select(self):
        src = self.src + 'print(df.filter(r => r.x > 2).col("name"), df.sort_by("x", desc=true).col("x"), df.select("x").columns)'
        self.assertEqual(lines(src), ['["c", "d"] [4, 3, 2, 1] ["x"]'])

    def test_missing_values(self):
        src = self.src + 'print(df.drop_na().shape, df.fill_na(0).col("y"))'
        self.assertEqual(lines(src), ["[3, 3] [10.0, 0, 30.0, 40.0]"])

    def test_mask_and_column_assignment(self):
        src = self.src + 'df["z"] = df["x"] * 2\nprint(df[df["x"] > 2].col("z"), df["z"].tolist())'
        self.assertEqual(lines(src), ["[6.0, 8.0] [2.0, 4.0, 6.0, 8.0]"])

    def test_group_by(self):
        src = 'let d = DataFrame({k: ["a","b","a"], v: [1, 2, 3]})\nprint(d.group_by("k", "sum").col("v"), d.group_by("k", "count").col("count"))'
        self.assertEqual(lines(src), ["[4.0, 2.0] [2, 1]"])

    def test_unknown_column_suggests(self):
        with self.assertRaises(SNRuntimeError) as cm:
            run(self.src + 'df["nam"]')
        self.assertIn("name", cm.exception.message)

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.csv").replace("\\", "/")
            out = run(self.src + f'df.to_csv("{p}")\nlet back = read_csv("{p}")\nprint(back.shape, back.col("y"))')
            self.assertEqual(out.strip(), "[4, 3] [10.0, nil, 30.0, 40.0]")

    def test_read_csv_types(self):
        src = 'let d = read_csv("examples/data/iris.csv")\nprint(d.shape, type(d["species"]), type(d["petal_width"]))'
        out = run(src, cwd=ROOT)
        self.assertEqual(out.strip(), "[150, 5] list tensor")


class MLTests(unittest.TestCase):
    def test_linear_regression_recovers_coefficients(self):
        src = """
let X = [[1, 2], [2, 1], [3, 5], [4, 3], [5, 8]]
let y = [3 * r[0] - 2 * r[1] + 1 for r in X]
let m = LinearRegression().fit(X, y)
print(allclose(m.weights, [3, -2], 1e-6), round(m.bias, 6), round(m.score(X, y), 6))
"""
        self.assertEqual(lines(src), ["true 1.0 1.0"])

    def test_gradient_descent_matches_closed_form(self):
        src = """
let X = tensor([[1.0], [2.0], [3.0], [4.0]])
let y = [3, 5, 7, 9]
let a = LinearRegression().fit(X, y)
let b = LinearRegression(lr=0.05, epochs=3000).fit(X, y)
print(allclose(a.weights, b.weights, 1e-3), abs(a.bias - b.bias) < 1e-2)
"""
        self.assertEqual(lines(src), ["true true"])

    def test_logistic_regression_separable(self):
        src = """
let X = [[0, 0], [0, 1], [1, 0], [3, 3], [3, 4], [4, 3]]
let y = ["a", "a", "a", "b", "b", "b"]
let m = LogisticRegression(lr=0.5, epochs=300).fit(X, y)
print(m.predict([[0, 0.5], [3.5, 3.5]]), m.score(X, y))
"""
        self.assertEqual(lines(src), ['["a", "b"] 1.0'])

    def test_knn(self):
        src = 'let m = KNN(k=1).fit([[0], [10]], ["lo", "hi"])\nprint(m.predict([[1], [9]]))'
        self.assertEqual(lines(src), ['["lo", "hi"]'])

    def test_kmeans_finds_two_blobs(self):
        src = """
let X = [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]]
let km = KMeans(k=2, seed=1).fit(X)
let l = km.labels
print(l[0] == l[1] and l[1] == l[2], l[3] == l[4] and l[4] == l[5], l[0] != l[3])
"""
        self.assertEqual(lines(src), ["true true true"])

    def test_neural_net_learns_xor(self):
        src = """
let net = NeuralNet(layers=[2, 6, 2], activation="tanh", lr=0.05, epochs=600, batch_size=4, seed=2)
net.fit([[0,0],[0,1],[1,0],[1,1]], [0,1,1,0])
print(net.predict([[0,0],[0,1],[1,0],[1,1]]), net.history[-1] < net.history[0])
"""
        self.assertEqual(lines(src), ["[0, 1, 1, 0] true"])

    def test_neural_net_regression(self):
        src = """
seed(1)
let X = linspace(-1, 1, 40).reshape(40, 1)
let y = X * 2 + 1
let net = NeuralNet(layers=[1, 8, 1], task="regression", lr=0.02, epochs=300, batch_size=10, seed=1)
net.fit(X, y.flatten())
print(net.score(X, y.flatten()) > 0.95)
"""
        self.assertEqual(lines(src), ["true"])

    def test_wrong_layer_sizes(self):
        with self.assertRaises(SNRuntimeError):
            run('NeuralNet(layers=[3, 4, 2]).fit([[1, 2], [3, 4]], [0, 1])')

    def test_unfitted_model(self):
        with self.assertRaises(SNRuntimeError):
            run("LinearRegression().predict([[1]])")

    def test_scaler(self):
        src = "let s = StandardScaler()\nlet Z = s.fit_transform([[1, 10], [2, 20], [3, 30]])\nprint(allclose(Z.mean(axis=0), [0, 0]), allclose(s.inverse_transform(Z), [[1, 10], [2, 20], [3, 30]]))"
        self.assertEqual(lines(src), ["true true"])

    def test_metrics(self):
        src = 'print(accuracy([1,0,1,1],[1,0,0,1]), mse([1,2],[2,4]), mae([1,2],[2,4]), r2([1,2,3],[1,2,3]), f1_score(["a","b"],["a","b"]))'
        self.assertEqual(lines(src), ["0.75 2.5 1.5 1.0 1.0"])

    def test_split_sizes_and_reproducibility(self):
        src = """
let X = arange(20).reshape(10, 2)
let a = train_test_split(X, range(10), test_size=0.3, seed=3)
let b = train_test_split(X, range(10), test_size=0.3, seed=3)
print(a[0].shape, a[1].shape, len(a[2]), len(a[3]), equal(a[1], b[1]))
"""
        self.assertEqual(lines(src), ["[7, 2] [3, 2] 7 3 true"])

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json").replace("\\", "/")
            src = f"""
let m = LogisticRegression(epochs=50).fit([[0],[1],[2],[3]], [0,0,1,1])
m.save("{p}")
let m2 = load_model("{p}")
print(m.predict([[0],[3]]) == m2.predict([[0],[3]]))
"""
            self.assertEqual(lines(src), ["true"])

    def test_numeric_gradient(self):
        self.assertEqual(lines("print(round(grad(x => x ** 3, 2), 3), round(gradient_descent(x => (x - 4) ** 2, 0, 0.2, 100), 3))"),
                         ["12.0 4.0"])


if __name__ == "__main__":
    unittest.main()
