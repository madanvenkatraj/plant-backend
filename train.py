"""
Plant Disease Model Trainer
============================
Dataset: PlantVillage (Kaggle)
Model: MobileNetV2 (Transfer Learning)

HOW TO RUN:
-----------
1. Install dependencies:
   pip install tensorflow pillow numpy

2. Download dataset:
   - Go to: https://www.kaggle.com/datasets/emmarex/plantdisease
   - Download and extract to: backend/dataset/PlantVillage/
   - Expected structure:
       backend/dataset/PlantVillage/
           Apple___Apple_scab/
           Apple___Black_rot/
           Tomato___healthy/
           ...

3. Run training:
   cd backend
   python train.py

4. Model saved to: backend/model/plant_disease_model.h5
                   backend/model/class_indices.json
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ─── CONFIGURATION ───────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset", "PlantVillage")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "model")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  Plant AI - Model Training")
print("=" * 60)
print(f"  Dataset : {DATASET_DIR}")
print(f"  Image   : {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch   : {BATCH_SIZE}")
print(f"  Epochs  : {EPOCHS}")
print("=" * 60)

# Auto-discover dataset path inside DATASET_DIR
dataset_path = DATASET_DIR
for root, dirs, files in os.walk(DATASET_DIR):
    if len(dirs) > 10:  # The actual PlantVillage folder has ~38 class directories
        dataset_path = root
        break

if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) < 10:
    print(f"\n[ERROR] Valid dataset not found in: {DATASET_DIR}")
    print("  Please run 'python download_dataset.py' to download the Kaggle dataset.")
    exit(1)

print(f"[OK] Using dataset path: {dataset_path}")

# ─── DATA GENERATORS ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

print("\n📂 Loading training data...")
train_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print(f"\n[OK] Found {train_gen.samples} training samples")
print(f"[OK] Found {val_gen.samples} validation samples")
print(f"[OK] Number of classes: {NUM_CLASSES}")

# Save class indices
class_indices = train_gen.class_indices
indices_path = os.path.join(MODEL_DIR, "class_indices.json")
with open(indices_path, "w") as f:
    json.dump(class_indices, f, indent=2)
print(f"[OK] Class indices saved: {indices_path}")

# ─── BUILD MODEL ─────────────────────────────────────────────────
print("\n[BUILD] Building MobileNetV2 model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
# Freeze base initially
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── CALLBACKS ───────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "plant_disease_model.h5")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# ─── PHASE 1: Train top layers ────────────────────────────────────
print("\n[TRAIN] Phase 1: Training top layers...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# ─── PHASE 2: Fine-tune last 30 base layers ───────────────────────
print("\n[TRAIN] Phase 2: Fine-tuning base model...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ─── EVALUATE ────────────────────────────────────────────────────
# Final save
model.save(model_path)

print(f"\n[OK] Model saved: {model_path}")
print(f"[OK] Class indices: {indices_path}")
print("\n[DONE] Training complete! You can now start the backend server.")
print("   Run: uvicorn app.main:app --reload --port 8000")
