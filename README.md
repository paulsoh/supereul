# Supereul
Minimalistic tensorflow session wrapper for Tensorflow ML experiments

# TODOS - listed in prioritized order
* [ ] Implement file outputs for train logs and resulting trained model
* [ ] Implement Supereul().update() method so that you can change operations to run
* [ ] Implement cross validation cycles
  * Currently only train and test
* [ ] Implement early stop training

# Requirements

* Python > 3.6.x
* Tensorflow > 1.12.x


# Usage
```python

import tensorflow as tf

def build_model():
    input_placeholder = tf.placeholder('float32', shape(None, 1), name="x")
    output_placeholder = tf.placeholder('float32', shape(None, 1), name="y")

    output = fully_connected(input_placeholder, 4)
    output = fully_connected(output, 16)
    output = fully_connected(output, 4)
    output = fully_connected(output, 1)

    loss = tf.losses.mean_squared_error(output, output_placeholder)
    update = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)

    return output, loss, update

output, loss, update = build_model()

operations_and_feed_values_for_training = {
    "operations": [update, loss],
    "feed_values": [x_tr, y_tr]
}

operations_and_feed_values_for_testing = {
    "operations": [loss],
    "feed_values": [x_te, y_te]
}

configs = {
    "test_every_n_times": 10,
    "log_every_n_times": 10,
    "save_model": True,
    "save_training_log": True
}

hyperparameters = {
    "batch_size": 32 ,
    "epochs": 100,
}

for i in range(10):
    # Run multiple times
    # Change hyperparameters in between for loops
	Supereul(
	    output, # TF graph, operation
	    operations_and_feed_values_for_training,
	    operations_and_feed_values_for_testing,
	    hyperparameters,
	    configs
	).run()

```
