import tensorflow as tf

default_hyperparameters = {
    "batch_size": 32,
    "optimizer": "adam",
    "epochs": 10000,
    "learning_rate": 0.002,
}

class Supereul():
    def __init__(self, graph, operation, feed_values, hyperparamters, use_default_hyperparameters=True):
        """
        Hyperparameters
        batch_size = 
        """
        self.graph = graph
        self.operation = operation
        self.feed_values = feed_values

        # For running sessions
        self.sess = tf.Session()
        self.global_variables_initializer = tf.global_variables_initializer

        if not use_default_hyperparameters:
            self.hyperparameters = hyperparameters
        else:
            self.hyperparameters = default_hyperparameters

    def run(self, log_level='verbose'):
        epochs = self.hyperparameters.get('epochs', 10)

        self.sess.run(self.global_variables_initializer())

        for i in range(epochs):
            self._run_train()

        self._run_test()

    def _run_train(self, log_level="verbose"):
        if not self._check_feed_values_validity():
            print("Failed, check validity for feed_values")
        print("Running Training Step")

        _placeholders = [
                op for op in
                self.graph.graph.get_operations() 
                if op.type == "Placeholder"
        ]

        _placeholder_names = [
            placeholder.name
            for placeholder
            in _placeholders
        ]

        from IPython import embed; embed()

        _feed_dict = {
            _placeholders[0]: self.feed_values[0],
            _placeholders[1]: self.feed_values[1]
        }

        _results = self.sess.run([
            self.graph
        ], feed_dict=_feed_dict)
        print("Running complete")
        print(_results)

    def _run_test(self):
        pass

    def _run_prediction(self):
        pass

    def _check_feed_values_validity(self):
        # TODO: Implement validity check
        if False:
            return False
        return True
