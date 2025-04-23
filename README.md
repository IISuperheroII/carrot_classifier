# 🥕 KI-basierte Karottenqualitätsprüfung (Good vs. Bad)

Dieses Projekt verwendet **künstliche Intelligenz (Convolutional Neural Networks)** zur automatischen **Klassifikation von Karottenbildern** in zwei Klassen:

- ✅ **Good (gute Qualität)**
- ❌ **Bad (schlechte Qualität)**

Optional ist auch eine dritte Klasse möglich: ⚠️ **Unknown** (z. B. Plastikobjekte oder keine Karotten)

---

## 📌 Motivation

In der Landwirtschaft und Lebensmittelverarbeitung spielt die visuelle Qualitätskontrolle eine zentrale Rolle. Durch Automatisierung mit KI kann:

- die Sortierung beschleunigt werden
- menschliche Fehler reduziert werden
- Ressourcen eingespart werden (weniger Ausschuss, effizientere Verpackung)

---

## 🔧 Hauptfunktionen

- 📷 Bilderkennung per Kamera oder Bildupload
- 🧠 Echtzeit-Vorhersage mit trainiertem CNN-Modell
- 📊 Analyse mit Confusion Matrix & Fehlerbildern
- 💾 Training mit eigenen Datensätzen (z. B. von [Kaggle](https://www.kaggle.com/datasets/jocelyndumlao/good-and-bad-classification-of-fresh-carrot))
- 🌐 Streamlit-Web-App mit UI und Snapshot-Speicher

---

## 🛠️ Technologie-Stack

| Technologie      | Beschreibung                      |
|------------------|-----------------------------------|
| Python           | Hauptsprache                      |
| TensorFlow / Keras | Modelltraining und -einsatz     |
| Streamlit        | Web-App mit Bild/Kamera-Unterstützung |
| OpenCV           | Live-Kamera-Integration           |
| Matplotlib       | Visualisierungen & Fehleranalyse  |
| scikit-learn     | Confusion Matrix, Evaluation      |
| Pillow (PIL)     | Bildverarbeitung                  |

---

## 🗂️ Projektstruktur

```bash
📁 carrot_classifier/
│
├── main.py                  # Training & Modell speichern
├── app.py                   # Streamlit-Anwendung für Klassifikation
├── /models/                 # Gespeicherte .h5-Modelle
├── /carrot_dataset/         # Trainings-/Validierungsbilder (good/bad)
├── /saved_snapshots/        # Kamera-Screenshots
├── requirements.txt         # Abhängigkeiten
└── README.md                # Dieses Dokument


🚀 Anleitung zur Nutzung
1. Setup
# Virtuelle Umgebung (optional)
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt

2. Modell trainieren
python main.py

3. Web-App starten
streamlit run app.py
