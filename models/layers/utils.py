from torch import nn


def get_activation(activation: str = "ReLU"):
    if activation in ["ReLU", "LeakyReLU", "Tanh", "Identity", "Sigmoid", "Softmax", "LogSoftmax"]:
        return getattr(nn, activation)()
    raise NotImplementedError(f"Activation function {activation} is not implemented.")