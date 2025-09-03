import tensorflow as tf

class MRZModel():
    def __init__(self, input_shape = (224, 224, 3), trainable = False, classes = 5):
        assert input_shape[0] % 32 == 0 and input_shape[1] % 32 == 0, \
            "Las dimensiones deben ser múltiplos de 32 por los strides=2 x5"
        inputs = tf.keras.Input(shape=input_shape)

        # Bloques conv
        x = tf.keras.layers.Conv2D(8,  (3,3), strides=2, padding="same", activation="relu")(inputs)  # /2
        x = tf.keras.layers.Conv2D(16, (3,3), strides=2, padding="same", activation="relu")(x)       # /4
        x = tf.keras.layers.Conv2D(32, (3,3), strides=2, padding="same", activation="relu")(x)       # /8
        x = tf.keras.layers.Conv2D(64, (3,3), strides=2, padding="same", activation="relu")(x)       # /16
        # Capa adicional con 128 filtros
        x = tf.keras.layers.Conv2D(128, (3,3), strides=2, padding="same", activation="relu")(x)      # /32

        # Pool global → (1,1,128)
        ph, pw = input_shape[0] // 32, input_shape[1] // 32
        x = tf.keras.layers.AveragePooling2D(pool_size=(ph, pw), padding="valid")(x)

        # Conv 1x1 a clases
        x = tf.keras.layers.Conv2D(classes, (1,1), padding="valid", use_bias=True)(x)

        # SQUEEZE espacial → logits (B, classes)
        outputs = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=[1,2]), name="squeeze_hw")(x)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def get_model(self):
        return self.model
