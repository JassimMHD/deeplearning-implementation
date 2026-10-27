import random


class Dense:
    """Fully connected layer: a = activation(W x + b).

    weights[j][i] connects input i to output neuron j.
    """

    def __init__(self, n_inputs, n_outputs, activation, rng=random):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation

        self.weights = [
            [rng.uniform(-1, 1) for _ in range(n_inputs)]
            for _ in range(n_outputs)
        ]
        self.biases = [0.0] * n_outputs

        # Cached during forward() for use in backward()
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = [
            self.activation.forward(
                sum(inputs[i] * self.weights[j][i]
                    for i in range(self.n_inputs)) + self.biases[j]
            )
            for j in range(self.n_outputs)
        ]
        return self.outputs

    def backward(self, deltas, learning_rate):
        """Apply one SGD step and return dL/d(inputs).

        deltas is dL/dz for each output neuron (gradient before activation).
        The input gradient is computed with the weights from *before* the
        update, as backprop requires.
        """
        grad_inputs = [
            sum(deltas[j] * self.weights[j][i]
                for j in range(self.n_outputs))
            for i in range(self.n_inputs)
        ]

        for j in range(self.n_outputs):
            for i in range(self.n_inputs):
                self.weights[j][i] -= learning_rate * deltas[j] * self.inputs[i]
            self.biases[j] -= learning_rate * deltas[j]

        return grad_inputs
