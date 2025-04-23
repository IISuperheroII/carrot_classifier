import os
import shutil
import random

src_folder = "carrot_dataset"
train_folder = "carrot_dataset/train"
val_folder = "carrot_dataset/validation"
split_ratio = 0.8

for cls in ["good", "bad"]:
    files = os.listdir(os.path.join(src_folder, cls))
    random.shuffle(files)
    split_point = int(len(files) * split_ratio)

    train_files = files[:split_point]
    val_files = files[split_point:]

    os.makedirs(os.path.join(train_folder, cls), exist_ok=True)
    os.makedirs(os.path.join(val_folder, cls), exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(src_folder, cls, f),
                    os.path.join(train_folder, cls, f))
    for f in val_files:
        shutil.copy(os.path.join(src_folder, cls, f),
                    os.path.join(val_folder, cls, f))

print("✅ Bilder erfolgreich aufgeteilt.")
