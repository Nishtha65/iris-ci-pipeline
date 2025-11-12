# iris_poison_mlflow.py
import numpy as np
import random
import os
import json
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def poison_labels(y, frac, seed=SEED):
    y = y.copy()
    n = len(y)
    k = int(np.floor(frac * n))
    idx = np.random.choice(n, k, replace=False)
    classes = np.unique(y)
    for i in idx:
        choices = [c for c in classes if c != y[i]]
        y[i] = np.random.choice(choices)
    return y, idx

def poison_features(X, frac, noise_scale=5.0, seed=SEED):
    X = X.copy()
    n = X.shape[0]
    k = int(np.floor(frac * n))
    idx = np.random.choice(n, k, replace=False)
    X[idx] = X[idx] + np.random.normal(loc=0.0, scale=noise_scale, size=X[idx].shape)
    return X, idx

def plot_confusion(cm, classes, out_path):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def run_experiment(poison_type='label', poison_frac=0.05):
    iris = load_iris()
    X, y = iris.data, iris.target
    # split into train/val/test so test remains clean
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    # apply poisoning only to train+val
    if poison_type == 'label':
        y_train_val_poisoned, poisoned_idx = poison_labels(y_train_val, poison_frac)
        X_train_val_poisoned = X_train_val.copy()
    elif poison_type == 'feature':
        X_train_val_poisoned, poisoned_idx = poison_features(X_train_val, poison_frac, noise_scale=5.0)
        y_train_val_poisoned = y_train_val.copy()
    else:
        raise ValueError("poison_type must be 'label' or 'feature'")

    # keep aside a validation split from poisoned train_val to simulate early stopping/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val_poisoned, y_train_val_poisoned, test_size=0.2, random_state=SEED, stratify=y_train_val_poisoned)

    # train a simple logistic regression
    model = LogisticRegression(max_iter=500, random_state=SEED)
    model.fit(X_train, y_train)

    # evaluate on clean test set (important: test set is untouched)
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
    cm = confusion_matrix(y_test, y_pred_test)

    # also evaluate on validation (poisoned) to observe train/val gap
    y_pred_val = model.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)

    results = {
        'poison_type': poison_type,
        'poison_frac': poison_frac,
        'test_accuracy': float(acc),
        'val_accuracy': float(acc_val),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'n_train_total': int(X_train.shape[0] + X_val.shape[0]),
        'n_poisoned': int(np.floor(poison_frac * (X_train.shape[0] + X_val.shape[0])))
    }

    # save confusion matrix plot
    out_dir = f"artifacts/{poison_type}_{int(poison_frac*100)}"
    os.makedirs(out_dir, exist_ok=True)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion(cm, iris.target_names, cm_path)

    return model, results, cm_path, poisoned_idx

def main():
    mlflow.set_experiment("iris-poisoning-study")
    poison_fracs = [0.05, 0.10, 0.50]
    poison_types = ['label', 'feature']

    for ptype in poison_types:
        for pf in poison_fracs:
            with mlflow.start_run(run_name=f"{ptype}_{int(pf*100)}pct"):
                model, results, cm_path, poisoned_idx = run_experiment(poison_type=ptype, poison_frac=pf)
                # log metrics and model
                mlflow.log_params({'poison_type': ptype, 'poison_frac': pf})
                mlflow.log_metrics({
                    'test_accuracy': results['test_accuracy'],
                    'val_accuracy': results['val_accuracy'],
                    'test_precision': results['test_precision'],
                    'test_recall': results['test_recall'],
                    'test_f1': results['test_f1'],
                })
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")
                # save result json
                res_path = os.path.join("artifacts", f"{ptype}_{int(pf*100)}", "results.json")
                with open(res_path, "w") as f:
                    json.dump(results, f, indent=2)
                mlflow.log_artifact(res_path)
                print(f"Logged run {ptype} {pf*100:.0f}% -> acc: {results['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
