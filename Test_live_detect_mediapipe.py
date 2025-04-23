import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import datetime
import os

# MediaPipe Setup
mp_object_detection = mp.solutions.object_detection
mp_drawing = mp.solutions.drawing_utils

# TensorFlow Modell laden (CNN zur Qualitätsklassifikation)
model_path = "models/carrot_model_v4.h5"
model = load_model(model_path)
model(tf.zeros((1, 150, 150, 3)))
class_names = ['bad', 'good']

# Logging vorbereiten
os.makedirs("logs", exist_ok=True)
log_file = open("logs/detection_log.txt", "a")

# Objekterkennung vorbereiten
detector = mp_object_detection.ObjectDetection(model_selection=0, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Drücke 'q' zum Beenden")

def classify_crop(crop):
    crop_resized = crop.resize((150, 150))
    arr = img_to_array(crop_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0][0]
    cls = class_names[int(pred > 0.5)]
    conf = pred if cls == 'good' else 1 - pred
    return cls, conf

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamerafehler")
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    annotated = frame.copy()

    if results.detections:
        for det in results.detections:
            for cat in det.label_id:
                class_name = mp_object_detection.ObjectDetection.LABEL_MAP.get(cat, 'object')
                if 'vegetable' in class_name.lower():
                    box = det.location_data.relative_bounding_box
                    x1, y1 = int(box.xmin * w), int(box.ymin * h)
                    x2, y2 = x1 + int(box.width * w), y1 + int(box.height * h)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w), min(y2, h)

                    crop = Image.fromarray(rgb[y1:y2, x1:x2])
                    pred_class, confidence = classify_crop(crop)

                    # Logging
                    log_entry = f"{datetime.datetime.now()}: {pred_class.upper()} ({confidence:.0%})\n"
                    log_file.write(log_entry)

                    # Anzeige
                    label = f"{class_name.upper()} | {pred_class.upper()} ({confidence:.0%})"
                    color = (0, 255, 0) if pred_class == 'good' else (0, 0, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("MediaPipe + CNN Klassifikation", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()