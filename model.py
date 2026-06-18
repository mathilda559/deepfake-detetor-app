import numpy as np
import cv2
from scipy.fft import fft2, fftshift

# ---------------------------------------------------
# Mathematisch kalibrierte Frequenzanalyse (224x224)
# ---------------------------------------------------
def frequency_score(frames):
    """
    KI-Generatoren (GANs, Diffusionsmodelle) hinterlassen feine, regelmäßige 
    Gitter-Strukturen im hochfrequenten Bereich (Schachbrett-Effekt).
    """
    scores = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Fourier-Transformation berechnen
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Hochfrequenten Randbereich isolieren (Zentrum ausblenden)
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        
        # Maske erstellen, um die niedrigen Frequenzen im Zentrum zu ignorieren
        mask = np.ones((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius=40, color=0, thickness=-1)
        
        # Nur die hohen Frequenzen betrachten
        high_freq_vals = magnitude_spectrum[mask == 1]
        
        # Logarithmische Skalierung gegen extreme Ausreißer
        log_mean = np.log1p(np.mean(high_freq_vals))
        scores.append(log_mean)
        
    if not scores:
        return 0.5
        
    avg_log_score = np.mean(scores)
    
    # Kalibrierung auf Basis der 224x224 Bildgröße
    # Wertebereich wird dynamisch normalisiert (0.0 = Echt, 1.0 = KI)
    normalized = (avg_log_score - 4.5) / 3.0
    return min(max(normalized, 0.0), 1.0)


# ---------------------------------------------------
# Unnatürliche Unschärfe-Inkonsistenz (Textur)
# ---------------------------------------------------
def blur_score(frames):
    """
    KI-Gesichter weisen oft lokale Regionen auf, die perfekt scharf sind, 
    während Texturen daneben (z.B. Hautporen, Haare) verschwommen/breiig wirken.
    """
    values = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        values.append(blur_val)
        
    if not values:
        return 0.5
        
    # Standardabweichung über die Zeitachse (Flackern zwischen den Frames)
    variation = np.std(values)
    
    # Kalibrierung des Varianz-Teilers
    normalized = variation / 150.0
    return min(max(normalized, 0.0), 1.0)


# ---------------------------------------------------
# Kantenstruktur & Kompressions-Artefakte
# ---------------------------------------------------
def compression_score(frames):
    """
    Untersucht unnatürliche Kantenübergänge an Objektgrenzen.
    """
    values = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        score = np.mean(edges)
        values.append(score)
        
    if not values:
        return 0.5
        
    avg_edges = np.mean(values)
    
    # Eine gesunde organische Kanten-Dichte liegt meistens um die 12-22.
    if 12.0 <= avg_edges <= 22.0:
        return 0.2  # Wahrscheinlich echt
    elif avg_edges < 12.0:
        return 0.7  # Zu weich / verwaschen (typisch für KI-Haut)
    else:
        return 0.8  # Zu verrauscht / künstlich generierte Kantenmuster


# ---------------------------------------------------
# GEWICHTETES ENSEMBLE (OPTIMIERT)
# ---------------------------------------------------
def predict_video(frames):
    if not frames:
        return {"ai_score": 50.0, "real_score": 50.0, "prediction": "UNBEKANNT"}

    freq = frequency_score(frames)
    blur = blur_score(frames)
    compression = compression_score(frames)
    
    # Frequenzanalyse hat die höchste Gewichtung (50%), da am verlässlichsten
    ai_score = (
        0.50 * freq +
        0.25 * blur +
        0.25 * compression
    )
    
    ai_score = min(max(ai_score, 0.0), 1.0)
    prediction = "AI GENERATED" if ai_score > 0.52 else "REAL"
    
    return {
        "ai_score": round(ai_score * 100, 2),
        "real_score": round((1 - ai_score) * 100, 2),
        "prediction": prediction
    }
