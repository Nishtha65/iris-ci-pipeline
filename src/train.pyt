import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

os.makedirs("model", exist_ok=True)

# Load CSV
df = pd.read_csv("data/iris.csv")

# Drop target and non-numeric columns
non_features = ["species", "id", "timestamp"]
X = df.drop(columns=[col for col in non_features if col in df.columns])
y = df["species"]

# Convert to numeric (just in case)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model/model.pkl")
print("Model trained and saved to model/model.pkl")
