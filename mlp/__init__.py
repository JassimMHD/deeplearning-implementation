"""A tiny pure-Python multi-layer perceptron with hand-written backprop."""

from .activations import Sigmoid
from .data import xor_dataset
from .layers import Dense
from .losses import BinaryCrossEntropy
from .network import NeuralNetwork
from .trainer import Trainer

__all__ = [
    "Sigmoid",
    "Dense",
    "BinaryCrossEntropy",
    "NeuralNetwork",
    "Trainer",
    "xor_dataset",
]
