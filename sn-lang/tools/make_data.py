"""Regenerates the sample CSV files in examples/data.

iris.csv   - Fisher's Iris data set (public domain), exported via scikit-learn.
houses.csv - synthetic house prices (fixed seed) for the regression examples.
"""
import csv
import os
import random

out = os.path.join(os.path.dirname(__file__), "..", "examples", "data")
os.makedirs(out, exist_ok=True)

try:
    from sklearn.datasets import load_iris
    d = load_iris()
    names = [d.target_names[t] for t in d.target]
    with open(os.path.join(out, "iris.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
        for row, n in zip(d.data, names):
            w.writerow([*row, n])
except ImportError:
    print("scikit-learn not installed; keeping existing iris.csv")

rng = random.Random(7)
with open(os.path.join(out, "houses.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["area_sqft", "bedrooms", "age_years", "city", "price_lakh"])
    for _ in range(80):
        area = rng.randint(500, 2500)
        bed = max(1, min(5, round(area / 600 + rng.gauss(0, 0.6))))
        age = rng.randint(0, 40)
        city = rng.choice(["Delhi", "Mumbai", "Pune", "Jaipur"])
        bonus = {"Delhi": 8, "Mumbai": 20, "Pune": 4, "Jaipur": 0}[city]
        price = 0.045 * area + 3.5 * bed - 0.4 * age + bonus + rng.gauss(0, 3)
        w.writerow([area, bed, age, city, round(price, 2)])
