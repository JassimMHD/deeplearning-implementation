# def xor_gate(a, b):
#     return (a * (1 - b)) + (b * (1 - a))


# # XOR Truth Table
# inputs = [
#     (0, 0),
#     (0, 1),
#     (1, 0),
#     (1, 1)
# ]

# for a, b in inputs:
#     output = xor_gate(a, b)
#     print(f"{a} XOR {b} = {output}")

import math
import random

random.seed(42)


def sigmoid(x):                         # Sigmoid Function
    return 1 / (1 + math.exp(-x))


X = [(0, 0), (0, 1), (1, 0), (1, 1)]
y = [0, 1, 1, 0]

                                                # Network: 2 inputs -> 4 hidden -> 1 output
w1 = [[random.uniform(-1, 1) for _ in range(2)]
      for _ in range(4)]
b1 = [0.0] * 4

w2 = [random.uniform(-1, 1) for _ in range(4)]
b2 = 0.0

learning_rate = 1.0


for epoch in range(10000):
    for inputs, target in zip(X, y):

        # Forward prop
        hidden = [
            sigmoid(sum(inputs[i] * w1[j][i]
                        for i in range(2)) + b1[j])
            for j in range(4)
        ]

        output = sigmoid(
            sum(hidden[j] * w2[j] for j in range(4)) + b2
        )
        

        # Backpropagation
        delta_out = output - target

        delta_hidden = [
            delta_out * w2[j] * hidden[j] * (1 - hidden[j])
            for j in range(4)
        ]

        # Update output weights
        for j in range(4):
            w2[j] -= learning_rate * delta_out * hidden[j]

        b2 -= learning_rate * delta_out

        # Update hidden weights
        for j in range(4):
            for i in range(2):
                w1[j][i] -= (
                    learning_rate * delta_hidden[j] * inputs[i]
                )
            b1[j] -= learning_rate * delta_hidden[j]

# Testing
print("XOR Neural Network Results:")

for inputs in X:
    hidden = [
        sigmoid(sum(inputs[i] * w1[j][i]
                    for i in range(2)) + b1[j])
        for j in range(4)
    ]

    output = sigmoid(
        sum(hidden[j] * w2[j] for j in range(4)) + b2
    )

    prediction = int(output >= 0.5)

    print(f"{inputs} -> {prediction} (probability: {output:.4f})")
