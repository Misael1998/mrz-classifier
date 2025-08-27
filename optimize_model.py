import tensorflow as tf

# Load the trained model (if not already in memory)
model = tf.keras.models.load_model("models/mrz_classifier.keras")

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to disk
with open("models/mrz_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to trained_model.tflite")
