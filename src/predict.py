import pandas as pd
import joblib

# Load model
model = joblib.load("model/model.pkl")

# Load data
df = pd.read_csv("data/iris.csv")
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

# Make predictions
preds = model.predict(X)
df["predicted_species"] = preds
df.to_csv("model/predictions.csv", index=False)

print("Predictions saved to model/predictions.csv")
