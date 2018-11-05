import tensorflow as tf

default_hyperparameters = {
    "batch_size": 32,
    "optimizer": "adam",
    "epochs": 10,
    "learning_rate": 0.002,
}

class Supereul():
    def __init__(
        self,
        graph,
        operations_and_feed_values_for_training,
        operations_and_feed_values_for_testing,
        hyperparameters,
        config
    ):
        """
        Hyperparameters
        batch_size = 
        """
        self.graph = graph
        self.operations_and_feed_values_for_training = operations_and_feed_values_for_training
        self.operations_and_feed_values_for_testing = operations_and_feed_values_for_testing

        self._feed_dict_for_training = {}
        self._feed_dict_for_testing = {}

        self._operations_for_training = []
        self._operations_for_testing = []

        self.hyperparameters = hyperparameters
        self.config = config

        # Parse placeholder tensors
        self.placeholders = self._parse_placeholder_tensors()
        # For running sessions
        self.sess = tf.Session()
        self.global_variables_initializer = tf.global_variables_initializer

    def run(self, log_level='verbose'):
        epochs = self.hyperparameters.get('epochs', 10)
        test_every_n_times = self.config.get('test_every_n_times', 10)
        log_every_n_times = self.config.get('log_every_n_times', 10)

        self.sess.run(self.global_variables_initializer())
        
        print("...global variables initialized...")
        print("Hyperparameters used for current session")
        self._print_hyperparameters()

        for i in range(epochs):
            self._run_train()
            if i % test_every_n_times == 0:
                self._run_test()

        # TODO: Save model and save log

    def _print_hyperparameters(self):
        for key, value in self.hyperparameters.items():
            print("{}: {}".format(key, value))

    def _run_train(self, log_level="verbose"):
        if not self._feed_dict_for_training or not self._operations_for_training:
            operations = self.operations_and_feed_values_for_training.get('operations')
            feed_values = self.operations_and_feed_values_for_training.get('feed_values')

            self._operations_for_training = operations
            self._feed_dict_for_training = self._generate_feed_dict(
                self.placeholders,
                feed_values
            )

        _results = self.sess.run(
            self._operations_for_training, 
            feed_dict=self._feed_dict_for_training
        )


    def _run_test(self):
        if not self._feed_dict_for_testing or not self._operations_for_testing:
            operations = self.operations_and_feed_values_for_testing.get('operations')
            feed_values = self.operations_and_feed_values_for_testing.get('feed_values')

            self._operations_for_testing = operations
            self._feed_dict_for_testing = self._generate_feed_dict(
                self.placeholders,
                feed_values
            )

        print("=== Running Test ===")
        _results = self.sess.run(
            self._operations_for_testing,
            feed_dict=self._feed_dict_for_testing
        )

        print(_results)

    def _run_prediction(self):
        pass


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

    def _check_feed_values_validity(self):
        # TODO: Implement validity check
        if False:
            return False
        return True
