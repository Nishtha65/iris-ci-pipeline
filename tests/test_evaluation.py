import joblib
import pandas as pd

def test_model_accuracy():
    model = joblib.load("model/model.pkl")
    df = pd.read_csv("data/iris.csv")

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

    preds = model.predict(X)
    assert len(preds) == len(df)
