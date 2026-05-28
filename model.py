import numpy as np
import cv2
from scipy.fft import fft2

# ---------------------------------------------------
# Frequency Analysis
# ---------------------------------------------------

def frequency_score(frames):
    scores = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fft_image = np.abs(fft2(gray))

        score = np.mean(fft_image)

        scores.append(score)

    if len(scores) == 0:
        return 0.5

    avg = np.mean(scores)

    normalized = min(1.0, avg / 1000000)

    return normalized


# ---------------------------------------------------
# Blur inconsistency analysis
# ---------------------------------------------------

def blur_score(frames):
    values = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        values.append(blur)

    if len(values) == 0:
        return 0.5

    variation = np.std(values)

    return min(1.0, variation / 1000)


# ---------------------------------------------------
# Compression artifact analysis
# ---------------------------------------------------

def compression_score(frames):
    values = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)

        score = np.mean(edges)

        values.append(score)

    if len(values) == 0:
        return 0.5

    avg = np.mean(values)

    return min(1.0, avg / 50)


# ---------------------------------------------------
# FINAL ENSEMBLE
# ---------------------------------------------------

def predict_video(frames):

    freq = frequency_score(frames)

    blur = blur_score(frames)

    compression = compression_score(frames)

    ai_score = (
        0.4 * freq +
        0.3 * blur +
        0.3 * compression
    )

    ai_score = min(max(ai_score, 0), 1)

    return {
        "ai_score": round(ai_score * 100, 2),
        "real_score": round((1 - ai_score) * 100, 2),
        "prediction": "AI GENERATED" if ai_score > 0.5 else "REAL"
    }
