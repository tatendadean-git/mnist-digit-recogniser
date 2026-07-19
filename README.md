# MNIST Digit Recognizer

A handwritten digit recognition web app trained on the MNIST dataset. Draw any digit (0–9) on the browser canvas and the model predicts it in real time.

## How It Works

1. You draw a digit on the black canvas in the browser
2. The drawing is sent as a 28x28 grid to the Python HTTP server
3. The server preprocesses and normalizes the drawing to match MNIST style
4. A trained RandomForest model predicts the digit and returns a confidence score
5. The result is displayed instantly in the browser

## Tech Stack

- Backend: Python standard library HTTP server (no Flask or Django)
- Frontend: Plain HTML and JavaScript, no frameworks
- Model: RandomForestClassifier from scikit-learn, trained on MNIST
- Accuracy: ~97% on the MNIST test set

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/mnist-digit-recognizer.git
cd mnist-digit-recognizer

python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Then open http://127.0.0.1:8000 in your browser.

Note: If no saved model is found, the app will automatically download the MNIST dataset
and train a new model on first run. This takes approximately 3-5 minutes and only
happens once. The trained model is saved as model.pkl for all future runs.

## Project Structure

```
mnist-digit-recognizer/
├── app.py                    # HTTP server and request routing
├── index.html                # Browser canvas frontend
├── predict.py                # Model loading and prediction logic
├── preprocess.py             # Normalizes canvas drawings to MNIST format
├── train_model.py            # Downloads MNIST and trains the model
├── test_digit_recognition.py # Unit tests
├── requirements.txt          # Python dependencies
└── README.md
```

## Running Tests

```bash
python -m pytest test_digit_recognition.py -v
```

## Training the Model Manually

The model trains automatically when needed, but you can also trigger it directly:

```bash
python train_model.py
```

This downloads the MNIST dataset (~55 MB), trains the model, prints the test accuracy,
and saves model.pkl.

## API Reference

| Method | Path      | Description                              |
|--------|-----------|------------------------------------------|
| GET    | /         | Serves the web frontend                  |
| GET    | /health   | Returns {"status": "ok"}                 |
| POST   | /predict  | Accepts a 28x28 grid, returns prediction |

POST /predict request body:

```json
{
  "grid": [[0.0, 0.0, ...], ...]
}
```

Response:

```json
{
  "prediction": 7,
  "confidence": 94.3,
  "probabilities": [0.01, 0.02, ...]
}
```

## Known Limitations

- Freehand drawing on a 28x28 grid is naturally messier than clean MNIST samples,
  so confidence scores can be lower than the model's true accuracy
- The preprocessing pipeline centers and normalizes the drawing, but stylistic
  differences from how MNIST digits were written can still affect results
- The server runs locally only and is not configured for deployment out of the box

## Linux Note

On Linux (Ubuntu/Debian/Mint), Tkinter is not bundled with Python. This project uses
a browser canvas instead, so no desktop display is required and it runs on all platforms.

## License

MIT License
