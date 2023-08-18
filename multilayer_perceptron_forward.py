"""A multilayer perceptron forward pass implementation in PyTorch."""

import subprocess

# Install the requirements
# subprocess.run(["pip", "install", "-r", "requirements.txt"])
from pprint import pprint
from typing import Any, Callable, Type

import numpy as np
import torch
from torch import Tensor


class MultiLayerPerceptron:
    """A multilayer perceptron forward pass implementation in PyTorch.

    Attributes:
    ----------
    input_size: int
        The size of the input layer
    hidden_layers: list[int]
        The size of the hidden layers.
    output_size: int
        The size of the output layer.
    activation_functions: list[Callable]
        The activation functions.
    activation_kwargs: list[dict[str, float]]
        The activation functions kwargs.
    gradients: list[Tensor]
        The gradients of the weights.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        output_size: int,
        activation_functions: list[str],
        activation_kwargs: list[dict[str, float]],
    ):
        self.input_size: int = input_size
        self.hidden_layers: list[int] = hidden_layers
        self.output_size: int = output_size
        self.activation_kwargs: list[dict[Any]] = activation_kwargs
        self.gradients: list[Tensor] = []
        self._init_activation_functions_(activation_functions)
        self.check_input()

    def _init_activation_functions_(self, activation_functions: list[str]):
        """It initializes the activation functions.

        Parameters
        ----------
        activation_functions: list[str]
            The activation functions.
        """

        self.activation_functions: list[Callable] = []
        for activation_function in activation_functions:
            self.activation_functions.append(getattr(type(self), activation_function))

    def check_input(self):
        """
        It checks the input of the class.
        """

        if not isinstance(self.input_size, int):
            raise TypeError("The input size must be an integer.")
        if not isinstance(self.hidden_layers, list):
            raise TypeError("The hidden layers must be a list.")
        if not isinstance(self.output_size, int):
            raise TypeError("The output size must be an integer.")
        if not isinstance(self.activation_functions, list):
            raise TypeError("The activation functions must be a list.")
        if len(self.hidden_layers) + 2 != len(self.activation_functions):
            raise ValueError(
                "The number of activation functions must be equal to the number of layers."
            )
        if len(self.activation_kwargs) != len(self.activation_functions):
            raise ValueError(
                "The number of activation kwargs must be equal to the number of activation functions."
            )

    def linear(
        self, x: Type[Tensor], W: Type[Tensor], a: Type[Tensor], b: Type[Tensor]
    ):
        """
        The lineal function of the perceptron.

        Parameters
        ----------
        x: Type[Tensor]
            The input tensor.
        W: Type[Tensor]
            The weight tensor.
        """

        # Check the dimensions of the tensors
        # assert a.shape == (self.output_size, 1) and b.shape == (self.output_size, 1)
        return a * x.mm(W) + b

    def sigmoid(self, x: Type[Tensor], W: Type[Tensor]):
        """
        The sigmoid function of the perceptron.

        Parameters
        ----------
        x: Type[Tensor]
            The input tensor.
        W: Type[Tensor]
            The weight tensor.
        """

        # Check the dimensions of the tensors
        x = x.mm(W)
        return 1 / (1 + torch.exp(-x))

    def tanh(self, x: Type[Tensor], W: Type[Tensor]):
        """
        The tanh function of the perceptron.

        Parameters
        ----------
        x: Type[Tensor]
            The input tensor.
        W: Type[Tensor]
            The weight tensor.
        """

        # Check the dimensions of the tensors
        x = x.mm(W)
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    def forward(
        self,
        x: Type[Tensor],
        weights: list[Type[Tensor]],
    ):
        """
        The feedforward function of the perceptron. It uses the tanh function
        for the input layer, the sigmoid function for the hidden layer and the
        lineal function for the output layer.

        Parameters
        ----------
        x: Type[Tensor]
            The input tensor.
            The intercept tensor.
        weights: list[Type[Tensor]]
            The list of weight tensors.

        Returns
        -------
        y_s: list[Type[Tensor]]
            The list of output tensors for each layer.
        """

        y_s = [x]
        for i, activation_function in enumerate(self.activation_functions):
            y_s.append(
                activation_function(
                    self, y_s[i], weights[i], **self.activation_kwargs[i]
                )
            )

        return y_s

    def derivative(self, activation_function: Type[Callable]):
        """
        The derivative of the activation function.

        Parameters
        ----------
        activation_function: Type[Callable]
            The activation function.

        Returns
        -------
        derivative: Type[Callable]
            The derivative of the activation function.
        """

        match activation_function.__name__:
            case self.sigmoid.__name__:
                return lambda x, W: torch.exp(-x.mm(W)) / (torch.exp(x.mm(W)) + 1) ** 2
            case self.tanh.__name__:
                return lambda x, W: 4 / (torch.exp(-x.mm(W)) + torch.exp(x.mm(W))) ** 2
            case self.linear.__name__:
                return lambda x, W, a, b: a
            case _:
                raise ValueError("The activation function is not valid.")

    def compute_local_gradient(
        self, weight: Tensor, y: Tensor, activation_function: Callable, kwargs: dict
    ):
        """Computes the local gradient

        Parameters
        ----------
        weight: Tensor
            The weight tensor.
        y: Tensor
            The output tensor.
        activation_function: Callable
            The activation function.
        kwargs: dict
            The activation function kwargs.
        """

        return self.derivative(activation_function)(y, weight, **kwargs)

    def learning_rule(
        self,
        error: Type[Tensor],
        weights: Type[Tensor],
        y_s: Type[Tensor],
        index_w: int,
        index_y: int,
    ):
        """
        The learning rule of the perceptron.

        Parameters
        ----------
        error: Type[Tensor]
            The error tensor.
        weights: list[Type[Tensor]]
            The weight tensor.
        y_s: list[Type[Tensor]]
            The list of output tensors for each layer.
        index: int
            The index of the layer.
        """

        if index_w == 0:
            return weights[index_w]
        weight = weights[index_w]
        y = y_s[index_y]
        stimulli = y_s[index_y - 1]
        # Calculate the error
        gradient: Tensor = self.compute_local_gradient(
            weight,
            stimulli,
            self.activation_functions[index_w],
            self.activation_kwargs[index_w],
        )
        if len(self.activation_functions) == index_w - 1:
            loss_func_derivative: Tensor = -(error * gradient).T * stimulli
        else:
            loss_func_derivative: Tensor = -(error * gradient).T * stimulli
        # Update the weights
        self.gradients.append(loss_func_derivative)
        weight = weight + loss_func_derivative.T
        return self.learning_rule(
            loss_func_derivative.T, weights, y_s, index_w - 1, index_y - 1
        )

    def feed_points(self, x: Type[Tensor], y_d: Type[Tensor]):
        """It calls the forward and learning rule functions for each point in the dataset.

        Parameters
        ----------
        x: Type[Tensor]
            The input tensor.
        y_d: Type[Tensor]
            The desired output tensor.
        """

        assert x.shape[1] == self.input_size
        # Create random weight tensors
        weights = []
        weights.append(torch.rand(self.input_size, self.input_size))
        for hidden_size in self.hidden_layers:
            weights.append(torch.rand(weights[-1].shape[1], hidden_size))
        weights.append(torch.rand(hidden_size, self.output_size))
        W_k = weights[-1]
        for i in range(x.shape[0]):
            x_i = x[i, :][None, :]
            yd_i = y_d[i, :][None, :]
            # Forward
            y_s = self.forward(x_i, weights)
            # Learning rule
            error = yd_i - y_s[-1]
            W_k = self.learning_rule(
                error,
                weights,
                y_s,
                len(weights) - 1,
                len(y_s) - 1,
            )
            weights[-1] = W_k
            self.gradients.append(W_k)
        return y_s


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(0)
    x = np.empty((2, 4))
    # Create the input tensor (based on the XOR example seen in class)
    x[0, :] = [0, 0, 1, 1]
    x[1, :] = [0, 1, 0, 1]
    y_d = np.empty((2, 4))
    y_d[0, :] = [0, 1, 1, 0]
    y_d[1, :] = [0, 1, 1, 0]
    y_d = torch.tensor(y_d, dtype=torch.float32).T
    x = torch.tensor(x, dtype=torch.float32).T
    # Create random intercepts and slopes for the lineal function
    a = torch.rand(1, 1)
    b = torch.rand(1, 1)
    # Test :)
    multilayer = MultiLayerPerceptron(
        2,
        [2, 2],
        2,
        ["tanh", "linear", "sigmoid", "tanh"],
        [{}, {"a": a, "b": b}, {}, {}],
    )
    # Print the output
    print("Output: ")
    pprint(multilayer.feed_points(x, y_d))
    print(" ")
    # Print the gradients
    print("Gradients: ")
    pprint(multilayer.gradients)
    print(len(multilayer.gradients))
