from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib

df = pd.read_csv("data/iris.csv")

# Only numeric features
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "model/model.pkl")
