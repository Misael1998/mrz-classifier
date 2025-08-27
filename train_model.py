from models.pre_trained_model import Model
from utils.data_loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import keras

path = './data/training/'

target_image_size = (512, 512)
target_input_shape = (512, 512,3)

data_loader = DataLoader(path = path)

data = data_loader.get_data(img_size=target_image_size)

data.info()

train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data["label"],
        random_state=42
        )

print("Train set:")
print(train_df)
print("\nTest set:")
print(test_df)

X_train = np.stack(train_df["data"].values)                         #type: ignore
y_train = keras.utils.to_categorical(train_df["label"].values)    #type: ignore

X_test = np.stack(test_df["data"].values)                           #type: ignore
y_test = keras.utils.to_categorical(test_df["label"].values)      #type: ignore

model = Model(input_shape=target_input_shape).get_model()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16  # lower for 4GB GPU
)
