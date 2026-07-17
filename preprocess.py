import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def preprocess_canvas(canvas, size=(28, 28)):
    arr = np.array(canvas, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Canvas must be a 2D array")
    if arr.shape != size:
        arr = np.resize(arr, size)

    arr = np.clip(arr, 0.0, 1.0)

    mask = arr > 0.15
    if np.any(mask):
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        crop = arr[y0:y1 + 1, x0:x1 + 1]
        crop = np.pad(crop, ((2, 2), (2, 2)), mode="constant")
        image = Image.fromarray(np.uint8(crop * 255.0))
        image = image.resize((20, 20), Image.Resampling.LANCZOS)
        canvas_out = np.zeros((28, 28), dtype=np.float32)
        x_start = (28 - 20) // 2
        y_start = (28 - 20) // 2
        canvas_out[y_start:y_start + 20, x_start:x_start + 20] = np.asarray(image, dtype=np.float32) / 255.0
        arr = canvas_out
    else:
        arr = np.zeros(size, dtype=np.float32)

    binary = (arr > 0.15).astype(bool)
    binary = binary_dilation(binary, iterations=1)
    arr = binary.astype(np.float32)

    image = Image.fromarray(np.uint8(arr * 255.0))
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    features = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    return features
