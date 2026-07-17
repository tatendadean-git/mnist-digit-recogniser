import os
import joblib
from preprocess import preprocess_canvas
from train_model import train_and_save


MODEL_PATH = "model.pkl"


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        train_and_save(path)
    model = joblib.load(path)
    # Ensure loaded model can provide probabilities. If not, retrain a
    # compatible model (e.g. RandomForestClassifier) and reload it.
    if not hasattr(model, "predict_proba"):
        train_and_save(path)
        model = joblib.load(path)
    return model


def predict_from_canvas(canvas, model=None):
    if model is None:
        model = load_model()

    features = preprocess_canvas(canvas)
    prediction = int(model.predict([features])[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]
        confidence = round(float(probabilities.max()) * 100, 1)
        probabilities = probabilities.tolist()
    else:
        probabilities = None
        confidence = 0.0
    return {"prediction": prediction, "confidence": confidence, "probabilities": probabilities}
