class Trainer:
    """Runs per-example (stochastic) gradient descent over a dataset."""

    def __init__(self, network, learning_rate=1.0):
        self.network = network
        self.learning_rate = learning_rate
        self.history = []

    def fit(self, X, y, epochs, log_every=None):
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, target in zip(X, y):
                total_loss += self.network.train_step(
                    inputs, target, self.learning_rate
                )

            mean_loss = total_loss / len(X)
            self.history.append(mean_loss)

            if log_every and (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch + 1:>6}  loss: {mean_loss:.6f}")

        return self.history
