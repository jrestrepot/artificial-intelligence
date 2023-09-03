"""The file that runs the code."""
import subprocess

# Install the requirements
subprocess.run(["pip", "install", "-r", "requirements.txt"])


from pprint import pprint

import torch

from multilayer_perceptron import MultiLayerPerceptron
from utils import (
    format_input,
    normalize_to_hypercube,
    plot_train_test_val_split,
    read_mat_file,
    train_test_val_split,
)

if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Read the input data
    data = read_mat_file("data/datosIA.mat")
    # Preprocess the data
    data = normalize_to_hypercube(data)
    # Split the data
    train, test, val = train_test_val_split(data)

    # Get the training data
    y_train = train.loc[:, "X"]
    x_train = train.loc[:, ["OD", "S"]]
    y_train = format_input(y_train)
    x_train = format_input(x_train)

    # Get the test data
    y_test = format_input(test.loc[:, "X"])
    x_test = format_input(test.loc[:, ["OD", "S"]])
    y_test = format_input(y_test)
    x_test = format_input(x_test)

    # Get the validation data
    y_val = format_input(val.loc[:, "X"])
    x_val = format_input(val.loc[:, ["OD", "S"]])
    y_val = format_input(y_val)
    x_val = format_input(x_val)

    # Plot the data splits
    plot_train_test_val_split(train, test, val)

    # Train the perceptrons
    energies = {}
    for num_hidden_layers in range(1, 4):
        for num_neurons in range(1, 6):
            for lerning_rate in [0.3, 0.5, 0.9]:
                hidden_layers = [num_neurons] * num_hidden_layers
                activation_functions = ["sigmoid"] * (num_hidden_layers + 2)
                activation_kwargs = [{}] * (num_hidden_layers + 2)
                multilayer = MultiLayerPerceptron(
                    2,
                    hidden_layers,
                    1,
                    activation_functions,
                    activation_kwargs,
                    eta=lerning_rate,
                )

                # Train the perceptron
                multilayer.sequential_train(x_train, y_train)
                # Test on the test set
                test_predictions = multilayer.predict(x_test)
                test_predictions = torch.reshape(test_predictions, (-1, 1))
                test_energy = multilayer.energy(test_predictions - y_test)
                multilayer.plot_predictions(
                    test_predictions,
                    y_test,
                    f"test_{num_hidden_layers}_{num_neurons}_{lerning_rate}",
                )
                multilayer.plot_energies(
                    f"test_{num_hidden_layers}_{num_neurons}_{lerning_rate}"
                )

                multilayer.plot_mean_gradients(
                    f"test_{num_hidden_layers}_{num_neurons}_{lerning_rate}"
                )
                # Test on the validation set, this mimics production
                val_predictions = multilayer.predict(x_val)
                val_predictions = torch.reshape(val_predictions, (-1, 1))
                val_energy = multilayer.energy(val_predictions - y_val)
                multilayer.plot_predictions(
                    val_predictions,
                    y_val,
                    f"val_{num_hidden_layers}_{num_neurons}_{lerning_rate}",
                )
                # Save the total energy
                energies[f"{num_hidden_layers}_{num_neurons}_{lerning_rate}"] = (
                    test_energy + val_energy
                )

    sorted_energies = sorted(energies, key=energies.get)
    min_energies = [sorted_energies[0], sorted_energies[1], sorted_energies[2]]
    print("Architectures with the minimum energies:", min_energies)
    max_energies = [sorted_energies[-1], sorted_energies[-2], sorted_energies[-3]]
    print("Architectures with the maximum energies:", max_energies)
    median_index = len(energies) // 2
    median_energies = [
        sorted_energies[median_index - 1],
        sorted_energies[median_index],
        sorted_energies[median_index + 1],
    ]
    print("Architectures with the median energy:", median_energies)
    pprint(energies)
