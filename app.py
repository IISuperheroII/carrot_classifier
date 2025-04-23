import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from datetime import datetime
import re

# 🌐 Streamlit Config
st.set_page_config(page_title="Karotten-KI-Klassifikation", layout="centered")
st.title("🥕 Karotten-KI Qualitätsprüfung")

# 📂 Modelle automatisch aus /models laden
def infer_classes_from_filename(filename):
    if "3class" in filename or "unknown" in filename:
        return ['bad', 'good', 'unknown']
    else:
        return ['bad', 'good']

model_versions = {}
model_dir = "models"
for file in os.listdir(model_dir):
    if file.endswith(".h5"):
        path = os.path.join(model_dir, file)
        # Sicherstellen, dass Datei wirklich ladbar ist
        try:
            _ = load_model(path, compile=False)
            label = f"🧠 {file}"
            model_versions[label] = (path, infer_classes_from_filename(file))
        except Exception as e:
            st.warning(f"⚠️ Modell konnte nicht geladen werden: {file}\n{str(e)}")

# Kein Modell verfügbar?
if not model_versions:
    st.error("❌ Kein gültiges Modell im Ordner `models/` gefunden.")
    st.stop()

# 📌 Modellauswahl
selected_model = st.selectbox("🧠 Modell wählen", list(model_versions.keys()))
model_path, class_names = model_versions[selected_model]
model = load_model(model_path)
st.success(f"✅ Modell geladen: `{model_path}`")

# 🔍 Bildgröße für Vorverarbeitung
img_width, img_height = 150, 150

# 🧠 Vorhersagefunktion
def predict_image(image):
    img = image.resize((img_width, img_height))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if len(class_names) == 2:
        pred = model.predict(img_array, verbose=0)[0][0]
        predicted_class = class_names[int(pred > 0.5)]
        confidence = 1 - pred if predicted_class == "bad" else pred
        return predicted_class, confidence
    else:
        preds = model.predict(img_array, verbose=0)[0]
        pred_idx = np.argmax(preds)
        return class_names[pred_idx], preds[pred_idx]

# 🧭 Tabs
tab1, tab2 = st.tabs(["📁 Bild hochladen", "📷 Kamera"])

# 📁 Tab 1 – Bild Upload
with tab1:
    st.header("📁 Bild hochladen")
    uploaded_file = st.file_uploader("Wähle ein Karottenbild (.jpg, .png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Hochgeladenes Bild", use_container_width=True)

        pred_class, conf = predict_image(image)

        if pred_class == "good":
            st.success(f"✅ **GUTE Karotte erkannt** ({conf:.2%} sicher)")
        elif pred_class == "bad":
            st.error(f"❌ **SCHLECHTE Karotte erkannt** ({conf:.2%} sicher)")
        else:
            st.warning(f"⚠️ **UNBEKANNT – vermutlich keine Karotte** ({conf:.2%} sicher)")

# 📷 Tab 2 – Kamera
with tab2:
    st.header("📷 Kamera-Snapshot & Vorhersage")
    if st.button("📸 Live-Kamerabild aufnehmen"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            st.image(image, caption="📸 Livebild", use_container_width=True)

            pred_class, conf = predict_image(image)

            if pred_class == "good":
                st.success(f"✅ **GUTE Karotte erkannt** ({conf:.2%} sicher)")
            elif pred_class == "bad":
                st.error(f"❌ **SCHLECHTE Karotte erkannt** ({conf:.2%} sicher)")
            else:
                st.warning(f"⚠️ **UNBEKANNT – vermutlich keine Karotte** ({conf:.2%} sicher)")

            if st.button("💾 Snapshot speichern"):
                os.makedirs("saved_snapshots", exist_ok=True)
                filename = f"snapshot_{pred_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                Image.fromarray(frame_rgb).save(os.path.join("saved_snapshots", filename))
                st.success("📷 Snapshot gespeichert.")
        else:
            st.error("❌ Kamera konnte nicht gestartet werden.")
