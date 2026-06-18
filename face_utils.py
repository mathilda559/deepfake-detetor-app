import cv2
import numpy as np
import os
import urllib.request

# Pfade für das Modell im lokalen Verzeichnis definieren
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

# Automatischer, sicherer Download der Modell-Dateien, falls noch nicht vorhanden
def ensure_model_files():
    if not os.path.exists(PROTO_PATH):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        urllib.request.urlretrieve(url, PROTO_PATH)
    if not os.path.exists(MODEL_PATH):
        url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        urllib.request.urlretrieve(url, MODEL_PATH)

try:
    ensure_model_files()
    face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
except Exception as e:
    print(f"Fehler beim Laden des Gesichtsmodells: {e}")
    face_net = None

def detect_faces(frame):
    """
    Erkennt Gesichter im Frame und gibt die Koordinaten als Liste von Boxen zurück.
    """
    if face_net is None:
        return []
        
    h, w = frame.shape[:2]
    # Blob erstellen (Optimale Größe für dieses ResNet SSD Modell ist 300x300)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # Schwellenwert leicht erhöht für weniger False Positives
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # Begrenzen auf Bildränder
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))
                
    return faces
