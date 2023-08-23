"""The file that runs the code."""

from pprint import pprint

import torch
from multilayer_perceptron import MultiLayerPerceptron
from utils import format_input, read_mat_file, read_txt

if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(0)

    # Create the input tensor (based on the XOR example seen in class)
    data = read_txt("data/DATOS.txt")
    y_d = data.iloc[:, -1]
    x = data.iloc[:, :-1]
    y_d = format_input(y_d)
    x = format_input(x)

    # Create random intercepts and slopes for the lineal function
    a = torch.rand(1, 1)
    b = torch.rand(1, 1)

    # Test :)
    multilayer = MultiLayerPerceptron(
        2,
        [2, 5, 3],
        1,
        ["tanh", "linear", "tanh", "sigmoid", "sigmoid"],
        [{}, {"a": a, "b": b}, {}, {}, {}],
        eta=1,
    )

    # Train the perceptron
    multilayer.train(x, y_d, epochs=1)

    # Print the gradients
    print("Gradients: ")
    for i, layer in enumerate(multilayer.gradients):
        print("Layer", i + 1)
        pprint(layer)

    multilayer.plot_gradients()
