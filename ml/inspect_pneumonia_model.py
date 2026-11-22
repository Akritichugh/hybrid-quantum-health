from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "pneumonia_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

print("\n===== MODEL SUMMARY =====")
model.summary()

print("\n===== MODEL INPUT =====")
print(model.input_shape)

print("\n===== MODEL OUTPUT =====")
print(model.output_shape)

print("\n===== MODEL LAYERS =====")
for layer in model.layers:
    print(
        layer.name,
        layer.__class__.__name__,
        getattr(layer, "output_shape", "Not available")
    )