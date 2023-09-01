"""The file that runs the code."""

from pprint import pprint

import numpy as np
import plotly.graph_objects as go
import torch

from multilayer_perceptron import MultiLayerPerceptron
from utils import format_input, normalize_to_hypercube, read_mat_file

if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Read the input data
    data = read_mat_file("data/datosIA.mat")
    data = normalize_to_hypercube(data)
    y_d = data.loc[:, "X"]
    x = data.loc[:, ["OD", "S"]]
    y_d = format_input(y_d)
    x = format_input(x)

    a1 = torch.rand(1, 1)
    b1 = torch.rand(1, 1)

    multilayer = MultiLayerPerceptron(
        2,
        [2],
        1,
        ["sigmoid", "sigmoid", "linear"],
        [{}, {}, {"a": a1, "b": b1}],
        eta=1,
    )

    # Train the perceptron
    multilayer.train(x, y_d, epochs=50)
    # Plot the predictions
    predictions = multilayer.predict(x)
    multilayer.plot_predictions(predictions, y_d, "normalized")

    multilayer.plot_gradients("normalized_smaller_network")

    # Print the gradients
    print("Gradients: ")
    for i, layer in enumerate(multilayer.gradients):
        print("Layer", i + 1)
        pprint(layer)

    # Create random intercepts and slopes for the lineal function
    # a1 = torch.rand(2, 1)
    # b1 = torch.rand(2, 1)

    # a2 = torch.ones(4, 1)
    # b2 = torch.zeros(4, 1)

    # a3 = torch.tensor(np.array([1, 1, 1, 1]))
    # a3 = a3.reshape(4, 1)
    # b3 = torch.zeros(4, 1)
    # "a": a1, "b": b1
