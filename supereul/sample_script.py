from __init__ import Supereul
from tensorflow.contrib.layers import conv2d, relu, fully_connected, flatten
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# Generate sample inputs
x = np.random.randn(1000, 8, 8, 1)
y = np.random.randn(1000, 1)

# Define hyperparameters
hyperparameters = {
    "batch_size": 32 ,
    "epochs": 100,
}

# Specify inputs for training and testing
train_feed_values = [] # Required
test_feed_values = [] # Optional, but really though

# Build default graph specified by operations
def build_model():
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        # Placeholders
        input_ph = tf.placeholder('float32', shape=(None, 8, 8, 1), name="x")
        output_ph = tf.placeholder('float32', shape=(None, 1), name="y")

        output = conv2d(input_ph, 8, [2, 2], stride=1, padding="same")
        output = flatten(output)
        output = fully_connected(output, 1)

        loss = tf.losses.mean_squared_error(output_ph, output)
        update = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss, var_list=tf.trainable_variables('model'))

        return output, loss, update

output, loss, update = build_model()

operations_and_feed_values_for_training = {
    "operations": [update, loss],
    "feed_values": [x, y]
}

operations_and_feed_values_for_testing = {
    "operations": [loss],
    "feed_values": [x[:40], y[:40]]
}


for i in range(10):
    Supereul(
        output, # Tensor graph
        operations_and_feed_values_for_training,
        operations_and_feed_values_for_testing,
        # Hyperparameters
        hyperparameters,
        {
            "test_every_n_times": 10,
            "log_every_n_times": 10,
            "save_model": True,
            "save_training_log": True
        }
    ).run()

