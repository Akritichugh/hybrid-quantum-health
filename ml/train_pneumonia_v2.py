
from pathlib import Path
import tensorflow as tf
import numpy as np


# =========================================
# PATHS AND SETTINGS
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "chest_xray"
    / "train"
)

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

NEW_MODEL_PATH = (
    MODEL_DIR
    / "pneumonia_model_v2.keras"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.20


# =========================================
# CHECK DATASET
# =========================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}"
    )

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Training folder:", TRAIN_DIR)
print("New model will be saved to:", NEW_MODEL_PATH)


# =========================================
# LOAD TRAINING DATA
# Split training images into 80% train / 20% validation
# The existing test folder is NOT used here.
# =========================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="binary",
    shuffle=True,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="binary",
    shuffle=False,
)

class_names = train_dataset.class_names

print("Classes:", class_names)

if class_names != ["NORMAL", "PNEUMONIA"]:
    raise ValueError(
        f"Unexpected class order: {class_names}"
    )


# =========================================
# CLASS WEIGHTS
# Helps account for class imbalance
# =========================================

normal_count = len(
    list((TRAIN_DIR / "NORMAL").glob("*"))
)

pneumonia_count = len(
    list((TRAIN_DIR / "PNEUMONIA").glob("*"))
)

total_count = normal_count + pneumonia_count

class_weights = {
    0: total_count / (2 * normal_count),
    1: total_count / (2 * pneumonia_count),
}

print("\nClass counts:")
print("NORMAL:", normal_count)
print("PNEUMONIA:", pneumonia_count)

print("Class weights:", class_weights)


# =========================================
# DATASET PERFORMANCE SETTINGS
# =========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# =========================================
# DATA AUGMENTATION
# =========================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ],
    name="data_augmentation",
)


# =========================================
# MOBILE NET V2 BASE MODEL
# =========================================

print("\nLoading MobileNetV2 pretrained weights...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze pretrained layers initially
base_model.trainable = False


# =========================================
# BUILD CLASSIFICATION MODEL
# =========================================

inputs = tf.keras.Input(
    shape=(160, 160, 3),
    name="xray_input",
)

# Augmentation only occurs during training
x = data_augmentation(inputs)

# MobileNetV2 preprocessing:
# Converts pixel values from [0, 255] to [-1, 1]
x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1,
    name="mobilenet_preprocessing",
)(x)

x = base_model(
    x,
    training=False,
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.30)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu",
)(x)

x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="pneumonia_probability",
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="pneumonia_mobilenet_v2",
)


# =========================================
# CALLBACKS
# =========================================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=NEW_MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# =========================================
# STAGE 1: TRAIN CLASSIFICATION HEAD
# =========================================

print("\n===== STAGE 1: TRAIN CLASSIFICATION HEAD =====")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks,
)


# =========================================
# STAGE 2: FINE-TUNE LAST MOBILE NET LAYERS
# =========================================

print("\n===== STAGE 2: FINE-TUNING =====")

base_model.trainable = True

# Freeze most layers; fine-tune only the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stability
for layer in base_model.layers:
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5,
    class_weight=class_weights,
    callbacks=callbacks,
)


# =========================================
# LOAD BEST SAVED MODEL
# =========================================

print("\nLoading best model from validation AUC...")

best_model = tf.keras.models.load_model(
    NEW_MODEL_PATH
)

print("\n===== TRAINING COMPLETE =====")
print("New model saved at:")
print(NEW_MODEL_PATH)

print("\nModel input:", best_model.input_shape)
print("Model output:", best_model.output_shape)
