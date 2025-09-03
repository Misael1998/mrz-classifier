from models.mrz_classifier import MRZModel
from models.mrz_optimizer import MRZOptimizer
from utils.data_loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf

path = './data/training/'

target_image_size = (512, 512)
target_input_shape = target_image_size + (3,)

data_loader = DataLoader(path = path)

data = data_loader.get_data(img_size=target_image_size, normalize=False)

data.info()

classes = set(data["label"])
print(f'Number of classes {len(classes)}')

train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data["label"],
        random_state=42
        )

print("Train & Test set loaded")

X_train = np.stack(train_df["data"].values)
y_train = keras.utils.to_categorical(train_df["label"].values)

X_test = np.stack(test_df["data"].values)
y_test = keras.utils.to_categorical(test_df["label"].values)

model = MRZModel(input_shape=target_input_shape, classes=len(classes)).get_model()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16
)

model.save("models/mrzClassifier.keras")

optimizer = MRZOptimizer(model=model,x_train=X_train)

tflite_model = optimizer.optimize_model()

with open("models/mrzClassifier.tflite", "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="models/mrzClassifier.tflite")
interpreter.allocate_tensors()
for d in interpreter.get_tensor_details():
    print("type",d['dtype'], " layer:", d['name'], d['shape'])