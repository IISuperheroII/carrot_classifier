import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import re
from tensorflow.keras import Input
import tensorflow as tf
                                 

# Hyperparameter
img_size = (150, 150)
batch_size = 32
epochs = 20

# Datenpfade
train_path = "carrot_dataset/train"
val_path = "carrot_dataset/validation"

# Daten vorbereiten (ohne validation_split!)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path, target_size=img_size, class_mode='binary', batch_size=batch_size
)

val_data = val_datagen.flow_from_directory(
    val_path, target_size=img_size, class_mode='binary', batch_size=batch_size, shuffle=False
)

# Modell definieren mit funktionalem API für Grad-CAM-Kompatibilität
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data
)

# Ordner sicherstellen
os.makedirs("models", exist_ok=True)

# Alle existierenden Modell-Dateien mit Versionsnummer finden
existing = [f for f in os.listdir("models") if re.match(r"carrot_model_v\d+\.h5", f)]

# Versionsnummern extrahieren
versions = [int(re.search(r"v(\d+)", f).group(1)) for f in existing]
next_version = max(versions) + 1 if versions else 1

# Modell speichern mit nächster Versionsnummer
model_path = f"models/carrot_model_v{next_version}.h5"
model.save(model_path)
print(f"Modell gespeichert unter: {model_path}")

# Lernkurven plotten
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
val_data.reset()
Y_pred = model.predict(val_data, steps=len(val_data), verbose=1)
y_pred = (Y_pred > 0.5).astype(int).reshape(-1)

true_labels = val_data.classes
class_labels = list(val_data.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(true_labels, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.show()

print(classification_report(true_labels, y_pred, target_names=class_labels))

# Fehlerbeispiele anzeigen
errors = np.where(y_pred != true_labels)[0]
all_images = []
all_labels = []

for i in range(len(val_data)):
    imgs, lbls = val_data[i]
    all_images.extend(imgs)
    all_labels.extend(lbls)
    if len(all_images) >= len(y_pred):
        break

n_show = 5
for idx in errors[:n_show]:
    img = all_images[idx]
    true_lbl = class_labels[int(true_labels[idx])]
    pred_lbl = class_labels[int(y_pred[idx])]

    plt.imshow(img)
    plt.title(f"Falsch: Vorhergesagt = {pred_lbl} | Tatsächlich = {true_lbl}")
    plt.axis('off')
    plt.show()
