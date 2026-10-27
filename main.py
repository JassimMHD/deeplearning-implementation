"""Train a 2 -> 4 -> 1 MLP to learn XOR with hand-written backprop."""

import random

from mlp import BinaryCrossEntropy, Dense, NeuralNetwork, Sigmoid, Trainer, xor_dataset

SEED = 42
EPOCHS = 10000
LEARNING_RATE = 1.0


def build_network(rng):
    sigmoid = Sigmoid()
    return NeuralNetwork(
        layers=[
            Dense(2, 4, activation=sigmoid, rng=rng),  # hidden layer
            Dense(4, 1, activation=sigmoid, rng=rng),  # output layer
        ],
        loss=BinaryCrossEntropy(),
    )


def main():
    rng = random.Random(SEED)
    X, y = xor_dataset()

    network = build_network(rng)
    Trainer(network, learning_rate=LEARNING_RATE).fit(X, y, epochs=EPOCHS)

    print("XOR Neural Network Results:")
    for inputs in X:
        probability = network.predict_proba(inputs)
        prediction = network.predict(inputs)
        print(f"{inputs} -> {prediction} (probability: {probability:.4f})")


if __name__ == "__main__":
    main()
