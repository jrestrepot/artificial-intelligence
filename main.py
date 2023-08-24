"""The file that runs the code."""

from pprint import pprint

import torch

from multilayer_perceptron import MultiLayerPerceptron
from utils import format_input, normalize_to_hypercube, read_mat_file

if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # # 1) Run the non-normalized example

    # # Read the input data
    # data = read_mat_file("data/datosIA.mat")
    # y_d = data.loc[:, "X"]
    # x = data.loc[:, ["OD", "S"]]
    # y_d = format_input(y_d)
    # x = format_input(x)

    # # Create random intercepts and slopes for the lineal function
    # a1 = torch.rand(2, 1)
    # b1 = torch.rand(2, 1)
    # # Test :)
    # multilayer = MultiLayerPerceptron(
    #     2,
    #     [2, 2],
    #     1,
    #     ["tanh", "linear", "tanh", "sigmoid"],
    #     [{}, {"a": a1, "b": b1}, {}, {}],
    #     eta=1,
    # )

    # # Train the perceptron
    # multilayer.train(x, y_d, epochs=5)

    # # Print the gradients
    # print("Gradients: ")
    # for i, layer in enumerate(multilayer.gradients):
    #     print("Layer", i + 1)
    #     pprint(layer)

    # multilayer.plot_gradients("not_normalized")

    # 2) Run the normalized example

    # Read the input data
    data = read_mat_file("data/datosIA.mat")
    data = normalize_to_hypercube(data)
    y_d = data.loc[:, "X"]
    x = data.loc[:, ["OD", "S"]]
    y_d = format_input(y_d)
    x = format_input(x)

    # Create random intercepts and slopes for the lineal function
    a1 = torch.rand(2, 1)
    b1 = torch.rand(2, 1)
    # Test :)
    multilayer = MultiLayerPerceptron(
        2,
        [2],
        1,
        ["tanh", "linear", "sigmoid"],
        [{}, {"a": a1, "b": b1}, {}],
        eta=1,
    )

    # Train the perceptron
    multilayer.train(x, y_d, epochs=5)

    # Print the gradients
    print("Gradients: ")
    for i, layer in enumerate(multilayer.gradients):
        print("Layer", i + 1)
        pprint(layer)

    multilayer.plot_gradients("normalized_smaller_network")

    # # 3) Run the non-normalized example with eta = 0.5

    # # Read the input data
    # data = read_mat_file("data/datosIA.mat")
    # y_d = data.loc[:, "X"]
    # x = data.loc[:, ["OD", "S"]]
    # y_d = format_input(y_d)
    # x = format_input(x)

    # # Create random intercepts and slopes for the lineal function
    # a1 = torch.rand(2, 1)
    # b1 = torch.rand(2, 1)
    # # Test :)
    # multilayer = MultiLayerPerceptron(
    #     2,
    #     [2, 2],
    #     1,
    #     ["tanh", "linear", "tanh", "sigmoid"],
    #     [{}, {"a": a1, "b": b1}, {}, {}],
    #     eta=0.5,
    # )

    # # Train the perceptron
    # multilayer.train(x, y_d, epochs=5)

    # # Print the gradients
    # print("Gradients: ")
    # for i, layer in enumerate(multilayer.gradients):
    #     print("Layer", i + 1)
    #     pprint(layer)

    # multilayer.plot_gradients("not_normalized_eta0.5")

    # # 4) Run the normalized example with eta = 0.5

    # # Read the input data
    # data = read_mat_file("data/datosIA.mat")
    # data = normalize_to_hypercube(data)
    # y_d = data.loc[:, "X"]
    # x = data.loc[:, ["OD", "S"]]
    # y_d = format_input(y_d)
    # x = format_input(x)

    # # Create random intercepts and slopes for the lineal function
    # a1 = torch.rand(2, 1)
    # b1 = torch.rand(2, 1)
    # # Test :)
    # multilayer = MultiLayerPerceptron(
    #     2,
    #     [2, 2],
    #     1,
    #     ["tanh", "linear", "tanh", "sigmoid"],
    #     [{}, {"a": a1, "b": b1}, {}, {}],
    #     eta=0.5,
    # )

    # # Train the perceptron
    # multilayer.train(x, y_d, epochs=5)

    # # Print the gradients
    # print("Gradients: ")
    # for i, layer in enumerate(multilayer.gradients):
    #     print("Layer", i + 1)
    #     pprint(layer)

    # multilayer.plot_gradients("normalized_eta0.5")
