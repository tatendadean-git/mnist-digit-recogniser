import os
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


MODEL_PATH = "model.pkl"


def train_and_save(model_path=MODEL_PATH):
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="liac-arff",
    )
    X = X.astype(np.float32) / 255.0
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use a larger, more expressive RandomForest to improve accuracy.
    # - increased `n_estimators`
    # - allow deeper trees (no `max_depth`) so trees can fit more complex patterns
    # - use `max_features='sqrt'` for better ensemble diversity
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"MNIST RandomForest accuracy: {acc:.3f}")

    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_and_save()
