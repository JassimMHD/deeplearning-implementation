import math


class BinaryCrossEntropy:
    """L = -[y log(p) + (1 - y) log(1 - p)]."""

    def __init__(self, eps=1e-12):
        self.eps = eps

    def loss(self, output, target):
        p = min(max(output, self.eps), 1 - self.eps)
        return -(target * math.log(p) + (1 - target) * math.log(1 - p))

    def output_delta(self, output, target):
        """dL/dz for a sigmoid output layer.

        Sigmoid and BCE combine so their derivatives cancel nicely:
        dL/dz = p - y.
        """
        return output - target
