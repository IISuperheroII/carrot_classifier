import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Modellpfad (vortrainiertes SSD-Modell auf COCO-Labels)
MODEL_PATH = "efficientdet_lite0.tflite"  # Stelle sicher, dass diese Datei im Projekt liegt

# Objektdetektor initialisieren
def create_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.5,
        max_results=5
    )
    return vision.ObjectDetector.create_from_options(options)

object_detector = create_detector()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Drücke 'q' zum Beenden")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kameraproblem.")
        break

    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = object_detector.detect(mp_image)

    annotated_frame = frame.copy()

    for det in detection_result.detections:
        cat = det.categories[0].category_name
        score = det.categories[0].score
        bbox = det.bounding_box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)

        # Nur Gemüse anzeigen, wenn Label vorhanden
        if cat.lower() in ["carrot", "vegetable", "broccoli", "cucumber"]:
            label = f"{cat.upper()} ({int(score * 100)}%)"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("MediaPipe Tasks Object Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
