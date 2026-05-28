import numpy as np
from deepface import DeepFace
from scipy.fft import fft2
import cv2

# 1. Face Embedding Stability (DeepFake Indicator)
def face_embedding_score(frames):
    embeddings = []

    for frame in frames:
        try:
            result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
            embeddings.append(result[0]["embedding"])
        except:
            continue

    if len(embeddings) < 2:
        return 0.5

    embeddings = np.array(embeddings)
    variance = np.var(embeddings)

    score = min(1.0, variance / 10)
    return score


# 2. Frequency Artifact Analysis
def frequency_score(frames):
    scores = []

    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f_transform = np.abs(fft2(gray))
        score = np.mean(f_transform)
        scores.append(score)

    normalized = np.mean(scores)
    return min(1.0, normalized / 1e6)


# 3. Blur inconsistency (fake videos often inconsistent)
def blur_score(frames):
    scores = []

    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores.append(score)

    if len(scores) == 0:
        return 0.5

    return min(1.0, np.std(scores) / 1000)


# 4. FINAL ENSEMBLE
def predict_video(frames):
    emb = face_embedding_score(frames)
    freq = frequency_score(frames)
    blur = blur_score(frames)

    # Weighted ensemble (important part!)
    ai_score = (
        0.5 * emb +
        0.3 * freq +
        0.2 * blur
    )

    return {
        "ai_score": round(ai_score * 100, 2),
        "real_score": round((1 - ai_score) * 100, 2),
        "prediction": "AI GENERATED" if ai_score > 0.5 else "REAL"
    }
