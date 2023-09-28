import subprocess

# Install the requirements
subprocess.run(["pip", "install", "-r", "requirements.txt"])

from lenet_5 import model
from utils import plot_confusion_matrix, process_data

if __name__ == "__main__":
    # Define file paths/ names
    train_filename = {
        "images": "CNN/data/train-images.idx3-ubyte",
        "labels": "CNN/data/train-labels.idx1-ubyte",
    }
    val_filename = {
        "images": "CNN/data/t10k-images.idx3-ubyte",
        "labels": "CNN/data/t10k-labels.idx1-ubyte",
    }
    # Process and read the data
    train_x, train_y, test_x, test_y, valid_x, valid_y = process_data(
        "CNN/data/numbers.csv", train_filename, val_filename
    )

    # Train the model and get the predictions
    predictions, accuracy = model(
        train_x,
        train_y,
        valid_x,
        valid_y,
        test_x,
        test_y,
        learning_rate=0.001,
        batch_size=128,
        num_epochs=10,
    )

    # Plot a confusion matrix
    plot_confusion_matrix(predictions, test_y, "figures/confusion_matrix.html")
