from __init__ import Supereul
from tensorflow.contrib.layers import conv2d, relu, fully_connected, flatten
import tensorflow as tf
import numpy as np


x = np.random.randn(1000, 8, 8, 1)
y = np.random.randn(1000, 1)

hyperparameters = {
    'batch_size': 32,
}

def build_model():
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        # Placeholders
        input_ph = tf.placeholder('float32', shape=(None, 8, 8, 1), name="x")
        output_ph = tf.placeholder('float32', shape=(None, 1), name="y")

        output = conv2d(input_ph, 8, [2, 2], stride=1, padding="same")
        output = flatten(output)
        output = fully_connected(output, 1)

        return output

tf.reset_default_graph()

output = build_model()

s = Supereul(
    output, # Tensor graph
    'operation', # Operation
    [x, y], # Feed dict
    # Hyperparameter
    {
        "batch_size": 32 
    }
)

s.run()
