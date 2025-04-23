import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.cm as cm
import os
from datetime import datetime

# Einstellungen
model_path = "models/carrot_model_v4.h5"  # Nutze nur Modelle mit Input() Layer!
class_names = ['bad', 'good']
img_width, img_height = 150, 150

# Modell laden
model = load_model(model_path)
dummy_input = tf.zeros((1, 150, 150, 3))
model(dummy_input)
print(f"Modell geladen: {model_path}")

# Vorhersagefunktion
def predict_image(image):
    img = image.resize((img_width, img_height))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    predicted_class = class_names[int(pred > 0.5)]
    confidence = 1 - pred if predicted_class == "bad" else pred
    return predicted_class, confidence, img_array

# Grad-CAM erzeugen
def generate_gradcam(image_pil, model, img_array):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    else:
        raise ValueError("Keine Conv2D-Schicht im Modell gefunden.")

    conv_layer = model.get_layer(last_conv)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        class_channel = predictions[:, class_idx]

    grads = tape.gradient(class_channel, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    img_rgb = img_array[0] * 255
    heatmap = cv2.resize(heatmap, (img_width, img_height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[:, :, :3] * 255
    superimposed_img = np.uint8(0.6 * img_rgb + 0.4 * heatmap)

    return Image.fromarray(superimposed_img.astype("uint8"))

# Funktion zur Karotten-Maske (orange Bereiche hervorheben)
def isolate_carrot_region(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV-Bereich für orange Karotten
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    inverse_mask = cv2.bitwise_not(mask)
    dark_bg = cv2.bitwise_and(background, background, mask=inverse_mask)
    combined = cv2.add(result, dark_bg)
    return combined

# Kamera starten
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
os.makedirs("snapshots", exist_ok=True)
print("Drücke 's' zum Speichern, 'p' für Grad-CAM, 'q' zum Beenden")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamerafehler.")
        break

    # Karottenbereich hervorheben
    filtered = isolate_carrot_region(frame)

    # Vorhersage vorbereiten
    image_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    pred_class, conf, img_array = predict_image(image_pil)

    label = f"{pred_class.upper()} ({conf:.0%})"
    color = (0, 255, 0) if pred_class == "good" else (0, 0, 255)
    overlay_frame = filtered.copy()
    thickness = 5
    overlay_frame = cv2.copyMakeBorder(overlay_frame, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=color)
    cv2.putText(overlay_frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Live Kamera - Karotten erkannt", overlay_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"snapshots/karotte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Bild gespeichert: {filename}")
    elif key == ord('p'):
        gradcam_img = generate_gradcam(image_pil, model, img_array)
        result = np.array(gradcam_img).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = cv2.copyMakeBorder(result, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=color)
        cv2.putText(result, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Grad-CAM Ergebnis", result)

cap.release()
cv2.destroyAllWindows()