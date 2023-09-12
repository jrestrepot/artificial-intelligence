"""The file that runs the code."""
import subprocess

# Install the requirements
subprocess.run(["pip", "install", "-r", "requirements.txt"])
import torch

from multilayer_perceptron import MultiLayerPerceptron
from utils import (
    format_input,
    normalize_to_hypercube,
    plot_train_test_val_split,
    read_mat_file,
    train_test_val_split,
)

# Install the requirements
# subprocess.run(["pip", "install", "-r", "requirements.txt"])


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Read the input data
    data = read_mat_file("data/datosIA.mat")
    # Preprocess the data
    data = normalize_to_hypercube(data)
    # Split the data
    train, test, val = train_test_val_split(data)

    # Get the training data, x and y are the same
    y_train = train.loc[:, ["OD", "S", "X"]]
    x_train = train.loc[:, ["OD", "S", "X"]]
    y_train = format_input(y_train)
    x_train = format_input(x_train)

    # Get the test data
    y_test = format_input(test.loc[:, ["OD", "S", "X"]])
    x_test = format_input(test.loc[:, ["OD", "S", "X"]])
    y_test = format_input(y_test)
    x_test = format_input(x_test)

    # Get the validation data
    y_val = format_input(val.loc[:, ["OD", "S", "X"]])
    x_val = format_input(val.loc[:, ["OD", "S", "X"]])
    y_val = format_input(y_val)
    x_val = format_input(x_val)

    # Plot the data splits
    plot_train_test_val_split(train, test, val)

    # Train the autoencoder
    hidden_layers = [4]
    activation_functions = ["tanh", "tanh"]
    activation_kwargs = [{}, {}]
    multilayer = MultiLayerPerceptron(
        3,
        hidden_layers,
        3,
        activation_functions,
        activation_kwargs,
        eta=0.3,
    )

    multilayer.sequential_train(x_train, y_train, max_epochs=50, tolerance=1e-10)
    multilayer.plot_mean_gradients("encoder_expand")
    multilayer.plot_energies("encoder_expand")
    predictions = multilayer.predict(x_test)
    multilayer.plot_predictions(predictions, y_test, "encoder_expand_test")
    predictions_val = multilayer.predict(x_val)
    multilayer.plot_predictions(predictions_val, y_val, "encoder_expand_val")

    # Train the autoencoder from PyTorch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    print("Training Pytorch's autoencoder for comparison...")

    # Define the model
    class Autoencoder(nn.Module):
        """Autoencoder model."""

        def __init__(self, input_size, hidden_size):
            """Initialize the model."""
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, input_size),
                nn.Tanh(),
            )

        def forward(self, x):
            """Forward pass."""
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    # Define the model
    model = Autoencoder(3, 4)
    # Define the loss function
    criterion = nn.MSELoss()
    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    # Define the data loaders
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # Train the model
    for epoch in range(50):
        for x, y in train_loader:
            model.forward(x)
            loss = criterion(model.forward(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Test the model
    predictions = predictions = torch.empty((len(x_val), len(x_val[0])))
    for i, (x, y) in enumerate(val_loader):
        predictions[i] = model.forward(x)
    multilayer.plot_predictions(y_val, predictions.detach(), "Pytorch")
