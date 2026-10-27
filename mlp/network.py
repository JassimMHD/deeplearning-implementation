class NeuralNetwork:
    """A stack of Dense layers trained with backpropagation."""

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(self, inputs):
        activations = list(inputs)
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def backward(self, output, target, learning_rate):
        """Backpropagate from the output layer down, updating every layer."""
        deltas = [self.loss.output_delta(output, target)]

        for index in reversed(range(len(self.layers))):
            grad_inputs = self.layers[index].backward(deltas, learning_rate)

            if index > 0:
                previous = self.layers[index - 1]
                deltas = previous.activation.backward(
                    previous.outputs, grad_inputs
                )

    def train_step(self, inputs, target, learning_rate):
        """One forward + backward pass on a single example. Returns the loss."""
        output = self.forward(inputs)[0]
        self.backward(output, target, learning_rate)
        return self.loss.loss(output, target)

    def predict_proba(self, inputs):
        return self.forward(inputs)[0]

    def predict(self, inputs, threshold=0.5):
        return int(self.predict_proba(inputs) >= threshold)
