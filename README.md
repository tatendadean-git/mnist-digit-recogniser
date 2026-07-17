# Digit Recognition Server

This project is a small HTTP server for recognizing handwritten digits from a browser-based canvas. It uses a local MNIST-trained model and serves a simple web frontend from the same repository.

## Run

```bash
cd /home/dean/Desktop/Proj
python3 -m pip install -r requirements.txt
python3 app.py
```

Then open http://127.0.0.1:8000 in a browser. The first request will train the model if no saved model is present.

## Files

- app.py: HTTP server and request handling
- index.html: browser canvas frontend
- train_model.py: downloads MNIST and trains the model
- predict.py: loads the saved model and predicts from a drawing
- preprocess.py: converts the canvas drawing into MNIST-style input
- test_digit_recognition.py: unit tests for preprocessing and prediction
