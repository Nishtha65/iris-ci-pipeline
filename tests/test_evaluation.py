import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load("model/model.pkl")
    X, y = load_iris(return_X_y=True)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.9, f"Low accuracy: {acc}"
