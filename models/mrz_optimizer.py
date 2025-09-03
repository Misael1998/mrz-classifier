import tensorflow as tf
import numpy as np

class MRZOptimizer():

    def __init__(self, model, x_train = None):
        def representative_dataset():
            for i in range(len(x_train)):  # más ejemplos = mejor calibración
                sample = x_train[i].astype(np.float32)  # rango 0..255
                sample = np.expand_dims(sample, axis=0)
                yield [sample]
        
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model)
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.converter.representative_dataset = representative_dataset
        self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        self.converter._experimental_disable_per_channel = False
        self.converter.experimental_new_quantizer = True
        #self.converter._experimental_full_int8_quantization = True

        self.converter.inference_input_type = tf.int8
        self.converter.inference_output_type = tf.int8


    def optimize_model(self):
        return self.converter.convert()

