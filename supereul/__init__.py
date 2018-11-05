import tensorflow as tf

default_hyperparameters = {
    "batch_size": 32,
    "optimizer": "adam",
    "epochs": 10,
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

        # Parse placeholder tensors
        self.placeholders = self._parse_placeholder_tensors()

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
        print("=== Fetching placeholder tensors from graph ===")

        _feed_dict = self._generate_feed_dict(
            self.placeholders,
            self.feed_values
        )

        _results = self.sess.run([
            self.graph
        ], feed_dict=_feed_dict)
        print("Running complete")
        print(_results)

    def _generate_feed_dict(self, placeholders, feed_values):
        assert len(placeholders) == len(feed_values), "Feed values and Placeholders do not have the same length!"
        return {
            key: value for
            key, value in 
            zip(placeholders, feed_values)
        }

    def _parse_placeholder_tensors(self):
        # Get tensors from self.graph
        # Get placeholder name from self.graph

        ops = tf.get_default_graph().get_operations()
        tensors = [
            o.values()[0]
            for o in ops 
            if len(o.values()) != 0
        ]

        _placeholders = [
                op.name for op in
                self.graph.graph.get_operations() 
                if op.type == "Placeholder"
        ]

        _placeholder_tensors = []
        for name in _placeholders:
            for t in tensors:
                if name in t.name:
                    _placeholder_tensors.append(t)

        return _placeholder_tensors

    def _run_test(self):
        pass

    def _run_prediction(self):
        pass

    def _check_feed_values_validity(self):
        # TODO: Implement validity check
        if False:
            return False
        return True
