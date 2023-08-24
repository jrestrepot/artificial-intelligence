"""A multilayer perceptron forward pass implementation in PyTorch."""

import subprocess

# Install the requirements
subprocess.run(["pip", "install", "-r", "requirements.txt"])
from pprint import pprint
from typing import Any, Callable, Type

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
        eta: float = 1,
    ):
        self.input_size: int = input_size
        self.hidden_layers: list[int] = hidden_layers
        self.output_size: int = output_size
        self.activation_kwargs: list[dict[Any]] = activation_kwargs
        self.gradients: list[Tensor] | None = None
        self.weights: list[Tensor] = []
        self.eta = eta
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

    def _init_gradients_(self, n_epochs: int, n_points: int):
        """It initializes the gradients.

        Parameters
        ----------
        n_epochs: int
            The number of epochs.
        n_points: int
            The number of points.
        """

        self.gradients = []
        self.gradients.append(torch.zeros((n_epochs, n_points, self.input_size)))
        for hidden_layer in self.hidden_layers:
            self.gradients.append(torch.zeros((n_epochs, n_points, hidden_layer)))
        self.gradients.append(torch.zeros((n_epochs, n_points, self.output_size)))

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
        assert a.shape == b.shape
        assert a.shape[0] == W.shape[0]

        return a * W.mm(x) + b

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
        x = W.mm(x)
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
        x = W.mm(x)
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    def forward(self, x: Type[Tensor]):
        """
        The feedforward function of the perceptron. It uses the tanh function
        for the input layer, the sigmoid function for the hidden layer and the
        lineal function for the output layer.

        Parameters
        ----------
        x: Type[Tensor]
            The input tensor.
            The intercept tensor.

        Returns
        -------
        y_s: list[Type[Tensor]]
            The list of output tensors for each layer.
        """

        y_s = [x]
        for i, activation_function in enumerate(self.activation_functions):
            y_s.append(
                activation_function(
                    self, y_s[i], self.weights[i], **self.activation_kwargs[i]
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
                return lambda x, W: torch.exp(-W.mm(x)) / (torch.exp(-W.mm(x)) + 1) ** 2
            case self.tanh.__name__:
                return lambda x, W: 4 / (torch.exp(-W.mm(x)) + torch.exp(W.mm(x))) ** 2
            case self.linear.__name__:
                return lambda x, W, a, b: a
            case _:
                raise ValueError("The activation function is not valid.")

    def backpropagation(
        self, error: Type[Tensor], y_s: Type[Tensor], point: int, epoch: int
    ):
        """
        The backpropagation.

        Parameters
        ----------
        error: Type[Tensor]
            The error tensor.
        y_s: list[Type[Tensor]]
            The list of output tensors for each layer (stimuli).
        point: int
            The point index. This is used to store the gradient.
        epoch: int
            The epoch index. This is used to store the gradient.
        """

        # Compute for the output layer
        current_index = len(self.activation_functions) - 1
        stimuli = y_s[current_index]
        weight = self.weights[current_index]
        local_gradient = -error * self.derivative(
            self.activation_functions[current_index]
        )(stimuli, weight, **self.activation_kwargs[current_index])
        self.gradients[current_index][epoch, point, :] = local_gradient.T
        self.weights[current_index] = weight + self.eta * local_gradient * stimuli.T
        # if torch.isnan(self.weights[current_index]).any():
        #     print("At least one weight is NaN.")

        # Compute for the rest of the layers
        current_index -= 1
        while current_index >= 0:
            past_local_gradient = local_gradient
            past_weight = weight
            stimuli = y_s[current_index]
            weight = self.weights[current_index]
            local_gradient = self.derivative(self.activation_functions[current_index])(
                stimuli, weight, **self.activation_kwargs[current_index]
            ) * past_weight.T.mm(past_local_gradient)
            self.gradients[current_index][epoch, point, :] = local_gradient.T

            self.weights[current_index] = weight + self.eta * local_gradient * stimuli.T
            # if torch.isnan(self.weights[current_index]).any():
            #     print("At least one weight is NaN.")
            current_index -= 1

    def train(self, x: Type[Tensor], y_d: Type[Tensor], epochs: int = 5):
        """It trains the perceptron.

        Parameters
        ----------
        x: Type[Tensor]
            The input tensor.
        y_d: Type[Tensor]
            The desired output tensor.
        epochs: int
            The number of epochs.
        """

        n_columns = x.shape[1]
        n_points = x.shape[0]
        y_d_n_columns = y_d.shape[1]
        assert n_columns == self.input_size
        assert y_d_n_columns == self.output_size

        # Initialize the gradients
        if self.gradients is None:
            self._init_gradients_(epochs, n_points)

        # Initialize the weights
        self.weights.append(torch.rand(self.input_size, self.input_size))
        for hidden_size in self.hidden_layers:
            self.weights.append(torch.rand(hidden_size, self.weights[-1].shape[0]))
        self.weights.append(torch.rand(self.output_size, hidden_size))

        # Train the perceptron
        for epoch in range(epochs):
            for i in range(n_points):
                x_i = x[i, :][None, :].T
                yd_i = y_d[i, :][None, :].T
                # Forward
                y_s = self.forward(x_i)
                if i == n_points - 1:
                    index = 0
                else:
                    index = i + 1
                y_s[0] = x[index, :][None, :].T
                # Backward
                error = (yd_i - y_s[-1]).sum()
                self.backpropagation(error, y_s, i, epoch)

        # Return the trained perceptron
        return self

    def predict(self, x: Type[Tensor]):
        """It predicts the output given the input. It assumes that the model is
        already trained.

        Parameters
        ----------
        x: Type[Tensor]
            The input tensor.

        Returns
        -------
        y: Type[Tensor]
            The output tensor.
        """

        if self.gradients is None:
            raise ValueError("The model is not trained, thus it cannot predict.")

        n_columns = x.shape[1]
        n_points = x.shape[0]
        assert n_columns == self.input_size
        predictions = []
        for i in range(n_points):
            x_i = x[i, :][None, :].T
            # Forward
            y_s = self.forward(x_i)
            predictions.append(y_s[-1])
        return predictions

    def plot_gradients(self, example: str):
        """
        It plots the gradients for each layer.

        Parameters:
        -----------

        example: str
            The name of the example
        """

        assert self.gradients is not None

        for i, layer in enumerate(self.gradients):
            fig = go.Figure()
            # Reshape the layer so that it is easier to plot
            reshaped_layer = layer.reshape(layer.shape[0] * layer.shape[1], -1)
            for j in range(reshaped_layer.shape[1]):
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(reshaped_layer.shape[0]),
                        y=reshaped_layer[:, j],
                        mode="lines",
                        name=f"Gradient {j + 1}",
                    )
                )
            fig.update_layout(
                title=f"Gradients for layer {i + 1}, example {example}",
                xaxis_title="Interations",
                yaxis_title="Gradient",
            )

            # Save figure into a html
            fig.write_html(f"figures/fig_layer_{i+1}_({example}).html")
