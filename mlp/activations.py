import math


class Sigmoid:
    """Sigmoid activation: s(x) = 1 / (1 + e^-x)."""

    def forward(self, x):
        return 1 / (1 + math.exp(-x))

    def backward(self, outputs, grad_outputs):
        """Turn dL/da into dL/dz using s'(z) = s(z) * (1 - s(z))."""
        return [
            grad * out * (1 - out)
            for grad, out in zip(grad_outputs, outputs)
        ]
