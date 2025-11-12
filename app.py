from flask import Flask, request, jsonify
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]])
    prediction = model.predict(features)[0]
    return jsonify({'class': iris.target_names[prediction]})

@app.route('/')
def home():
    return "IRIS Model API running successfully"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
