"""A multilayer perceptron forward pass implementation in PyTorch."""
from typing import Any, Callable, Type

import numpy as np
import plotly.graph_objects as go
import torch
from torch import Tensor, tensor


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
        The gradients of the weights.+
    energies: list[float]
        The energies of the errors.
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
        self.energies: list[float] = []
        self.weights: list[Tensor] | None = None
        self.eta = eta
        self.derivatives = []
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

    def _init_weights_(self):
        """It initializes the weights."""

        self.weights = []
        self.weights.append(torch.rand(self.input_size, self.input_size))
        for hidden_size in self.hidden_layers:
            self.weights.append(torch.rand(hidden_size, self.weights[-1].shape[0]))
        self.weights.append(torch.rand(self.output_size, hidden_size))

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
        self,
        stimulli: Type[Tensor],
        weight: Type[Tensor],
        a: Type[Tensor],
        b: Type[Tensor],
    ):
        """
        The lineal function of the perceptron.

        Parameters
        ----------
        stimulli: Type[Tensor]
            The input tensor.
        weight: Type[Tensor]
            The weight tensor.
        """

        # Check the dimensions of the tensors
        assert a.shape == b.shape
        assert a.shape[0] == weight.shape[0]

        return a * weight.mm(stimulli) + b

    def sigmoid(self, stimulli: Type[Tensor], weight: Type[Tensor]):
        """
        The sigmoid function of the perceptron.

        Parameters
        ----------
        stimulli: Type[Tensor]
            The input tensor.
        weight: Type[Tensor]
            The weight tensor.
        """

        # Check the dimensions of the tensors
        local_induced_field = weight.mm(stimulli)
        return torch.sigmoid(local_induced_field)

    def tanh(self, stimulli: Type[Tensor], weight: Type[Tensor]):
        """
        The tanh function of the perceptron.

        Parameters
        ----------
        stimulli: Type[Tensor]
            The input tensor.
        weight: Type[Tensor]
            The weight tensor.
        """

        # Check the dimensions of the tensors
        local_induced_field = weight.mm(stimulli)

        return torch.tanh(local_induced_field)

    def forward(self, x_input: Type[Tensor]):
        """
        The feedforward function of the perceptron. It uses the tanh function
        for the input layer, the sigmoid function for the hidden layer and the
        lineal function for the output layer.

        Parameters
        ----------
        x_input: Type[Tensor]
            The input tensor.
            The intercept tensor.

        Returns
        -------
        y_s: list[Type[Tensor]]
            The list of output tensors for each layer.
        """

        y_s = [x_input]
        self.derivatives = []

        for i, activation_function in enumerate(self.activation_functions):
            y_s.append(
                activation_function(
                    self, y_s[i], self.weights[i], **self.activation_kwargs[i]
                )
            )
            self.derivatives.append(
                self.derivative(activation_function)(
                    y_s[i], self.weights[i], **self.activation_kwargs[i]
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
                return lambda x, W: activation_function(self, x, W) * (
                    1 - activation_function(self, x, W)
                )
            case self.tanh.__name__:
                return lambda x, W: 1 - activation_function(self, x, W) ** 2
            case self.linear.__name__:
                return lambda x, W, a, b: a
            case _:
                raise ValueError("The activation function is not valid.")

    def energy(self, error: Type[Tensor]):
        """
        The energy function.

        Parameters
        ----------
        error: Type[Tensor]
            The error tensor.
        """

        return 0.5 * (torch.square(error)).sum()

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
        local_gradient = error * self.derivatives[current_index]
        self.gradients[current_index][epoch, point, :] = local_gradient.T
        self.weights[current_index] = weight + self.eta * local_gradient * stimuli.T
        # Compute for the rest of the layers
        current_index -= 1
        while current_index >= 0:
            past_local_gradient = local_gradient
            past_weight = weight
            stimuli = y_s[current_index]
            weight = self.weights[current_index]
            local_gradient = self.derivatives[current_index] * past_weight.T.mm(
                past_local_gradient
            )
            self.gradients[current_index][epoch, point, :] = local_gradient.T

            self.weights[current_index] = weight + self.eta * local_gradient * stimuli.T
            current_index -= 1

    def sequential_train(
        self,
        x_input: Type[Tensor],
        y_d: Type[Tensor],
        max_epochs: int = 50,
        tolerance: float = 1e-3,
    ):
        """It trains the perceptron.

        Parameters
        ----------
        x_input: Type[Tensor]
            The input tensor.
        y_d: Type[Tensor]
            The desired output tensor.
        max_epochs: int
            The max number of epochs.
        tolerance: float
            The tolerance for the change in the error.
        """

        n_columns = x_input.shape[1]
        n_points = x_input.shape[0]
        y_d_n_columns = y_d.shape[1]

        # Check dimensions
        assert n_columns == self.input_size
        assert y_d_n_columns == self.output_size

        print("Training the perceptron...")

        # Initialize the gradients
        if self.gradients is None:
            self._init_gradients_(max_epochs, n_points)

        # Initialize the weights
        if self.weights is None:
            self._init_weights_()

        # Train the perceptron
        epoch = 0
        # past_mean_energy = np.inf
        while epoch < max_epochs:
            energies = []
            for i in range(n_points):
                x_i = x_input[i, :][None, :].T
                yd_i = y_d[i, :][None, :].T
                # Forward
                y_s = self.forward(x_i)
                # Backward
                error = (yd_i - y_s[-1]).sum()
                energies.append(self.energy(error))
                self.backpropagation(error, y_s, i, epoch)
            # Stop iterating if the error is not changing
            mean_energy = np.mean(energies)
            self.energies.append(mean_energy)
            if mean_energy < tolerance:
                print("The error tolerance has been reached. Stopping the training...")
                self.gradients = [
                    self.gradients[i][: epoch + 1, :, :]
                    for i in range(len(self.gradients))
                ]
                break
            epoch += 1

        # Return the trained perceptron
        return self

    def batch_train(
        self,
        x_input: Type[Tensor],
        y_d: Type[Tensor],
        max_epochs: int = 1000,
        tolerance: float = 1e-3,
    ):
        """It trains the perceptron in batch mode.

        Parameters
        ----------
        x_input: Type[Tensor]
            The input tensor.
        y_d: Type[Tensor]
            The desired output tensor.
        max_epochs: int
            The max number of epochs.
        tolerance: float
            The tolerance for the change in the error.
        """

        n_columns = x_input.shape[1]
        n_points = x_input.shape[0]
        y_d_n_columns = y_d.shape[1]

        # Check dimensions
        assert n_columns == self.input_size
        assert y_d_n_columns == self.output_size

        print("Training the perceptron...")

        # Initialize the gradients
        if self.gradients is None:
            self._init_gradients_(max_epochs, n_points)

        # Initialize the weights
        if self.weights is None:
            self._init_weights_()

        # Train the perceptron
        epoch = 0
        while epoch < max_epochs:
            energies = np.zeros((n_points, 1))
            errors = np.zeros((n_points, 1))
            stimuli = [None] * n_points

            for i in range(n_points):
                x_i = x_input[i, :][None, :].T
                yd_i = y_d[i, :][None, :].T
                # Forward
                y_s = self.forward(x_i)
                stimuli[i] = y_s
                error = (yd_i - y_s[-1]).sum()
                errors[i] = error
                energies[i] = self.energy(error)

            max_error_index = np.argmax(errors)
            if not isinstance(max_error_index, np.int64):
                max_error_index = max_error_index[0]
            # Get stimuli for the max error
            stimuli_max_error = stimuli[max_error_index]
            mean_energy = np.mean(energies)
            mean_error = np.mean(errors)
            self.energies.append(mean_energy)
            # Backward
            self.backpropagation(mean_error, stimuli_max_error, i, epoch)
            # Stop iterating if the error is not changing
            if mean_energy < tolerance:
                print("The error tolerance has been reached. Stopping the training...")
                self.gradients = [
                    self.gradients[i][: epoch + 1, :, :]
                    for i in range(len(self.gradients))
                ]
                break
            epoch += 1

        # Return the trained perceptron
        return self

    def predict(self, x_input: Type[Tensor]) -> list[float]:
        """It predicts the output given the input. It assumes that the model is
        already trained.

        Parameters
        ----------
        x_input: Type[Tensor]
            The input tensor.

        Returns
        -------
        y: Type[Tensor]
            The output tensor.
        """

        if self.gradients is None:
            raise ValueError("The model is not trained, thus it cannot predict.")

        print("Predicting...")

        n_columns = x_input.shape[1]
        n_points = x_input.shape[0]
        assert n_columns == self.input_size
        predictions = []
        for i in range(n_points):
            x_i = x_input[i, :][None, :].T
            # Forward
            y_s = self.forward(x_i)
            predictions.append(y_s[-1])
        return tensor(predictions)

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
                        x=np.arange(reshaped_layer.shape[0] + 1),
                        y=reshaped_layer[:, j],
                        mode="lines",
                        name=f"Gradient {j + 1}",
                    )
                )
            fig.update_layout(
                title=f"Gradients for layer {i + 1}, example {example}",
                xaxis_title="Iterations",
                yaxis_title="Gradient",
            )

            # Save figure into a html
            fig.write_html(f"figures/fig_layer_{i+1}_({example}).html")

    def plot_mean_gradients(self, example: str):
        """
        Plot the mean gradients per epoch for each layer.

        Parameters:
        -----------
        example: str
            The name of the example
        """

        assert self.gradients is not None

        fig = go.Figure()
        mean_gradients = []
        for layer in self.gradients:
            # Get mean gradients per epoch
            mean_gradients.append(np.array(layer.mean(axis=1).mean(axis=1)))
        mean_gradients = np.array(mean_gradients).mean(axis=0)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(mean_gradients) + 1),
                y=mean_gradients,
                mode="lines",
                name="Mean Gradient",
            )
        )

        fig.update_layout(
            title=f"Mean Gradients, example {example}",
            xaxis_title="Epochs",
            yaxis_title="Gradient",
        )

        # Save figure into a html
        fig.write_html(f"figures/mean_gradients_({example}).html")

    def plot_energies(self, example: str):
        """
        It plots the energies for each layer.

        Parameters:
        -----------

        example: str
            The name of the example
        """

        assert self.energies is not None

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(self.energies) + 1),
                y=self.energies,
                mode="lines",
                name="Mean Instantaneous Energy of the Error per Epoch",
            )
        )
        fig.update_layout(
            title=f"Energies for example {example}",
            xaxis_title="Epochs",
            yaxis_title="Instantaneous Energy of the Error",
        )

        # Save figure into a html
        fig.write_html(f"figures/energies_({example}).html")

    def plot_predictions(self, predictions: Tensor, real: Tensor, example: str) -> None:
        """
        It plots the predictions vs the real values.

        Parameters:
        -----------
        predictions: Tensor
            The predictions
        real: Tensor
            The real values
        example: str
            The name of the example
        """

        real = real.flatten()
        predictions = predictions.flatten()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(real) + 1),
                y=predictions,
                mode="lines",
                name="Predictions",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(real) + 1),
                y=real,
                mode="lines",
                name="Real",
            )
        )
        fig.update_layout(
            title="Real vs Predicted",
        )

        # Save figure into a html
        fig.write_html(f"figures/output_({example}).html")
