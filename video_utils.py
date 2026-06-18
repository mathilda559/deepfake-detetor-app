import cv2
import numpy as np
from face_utils import detect_faces

def extract_frames(video_path, max_frames=30):
    """
    Liest ein Video ein, extrahiert Frames und versucht, Gesichter auszuschneiden.
    Gibt eine Liste von Bildern (entweder Gesichter oder Vollbilder) zurück.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    face_crops = []
    
    if not cap.isOpened():
        return []

    # Frame-Anzahl ermitteln
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Falls Metadaten fehlen, lesen wir linear alle 10 Frames
        step = 10
    else:
        # Gleichmäßige Verteilung über das gesamte Video
        step = max(1, total_frames // max_frames)

    frame_idx = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % step == 0:
            # 1. Versuche Gesichter im Original-Frame zu finden
            faces = detect_faces(frame)
            
            if faces:
                # Wenn Gesichter gefunden wurden, schneide das größte Gesicht aus
                faces = sorted(faces, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
                x1, y1, x2, y2 = faces[0]
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_crop_res = cv2.resize(face_crop, (224, 224))
                    face_crops.append(face_crop_res)
            
            # Unabhängig davon speichern wir das runterskalierte Vollbild als Fallback
            full_res = cv2.resize(frame, (224, 224))
            frames.append(full_res)
            
            extracted_count += 1
            if extracted_count >= max_frames:
                break
                
        frame_idx += 1
        
    cap.release()
    
    # Priorität: Wenn wir Gesichter gefunden haben, analysieren wir NUR die Gesichter!
    if len(face_crops) >= 5:
        return face_crops
    return frames
