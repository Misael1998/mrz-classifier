import tensorflow as tf

class Model():
    def __init__(self, input_shape = (224, 224, 3), trainable = False, classes = 5):
        self.input_shape = input_shape
        self.classes = classes
        self.trainable = trainable

        base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
                )
        base_model.trainable = self.trainable

        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.classes, activation='softmax')
            ])

        self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
        )

    def get_model(self):
        return self.model

