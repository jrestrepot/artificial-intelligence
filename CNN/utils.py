"""A module to process the MNIST data."""

import idx2numpy
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix


def process_data(
    test_path: str, train_path: dict[str, str], validation_path: dict[str, str]
):
    """It reads and processes the MNIST data so that the lenet 5 algorithm can
    use it.

    Parameters
    ----------
    test_path: str
        The path to the test data.
    train_path: dict[str, str]
        The path to the training data.
    validation_path: dict[str, str]
        The path to the validation data.

    Returns
    -------
    train_x: pd.DataFrame
        The training data.
    train_y: pd.DataFrame
        The training labels.
    valid_x: pd.DataFrame
        The validation data.
    valid_y: pd.DataFrame
        The validation labels.
    test_x: pd.DataFrame
        The test data.
    test_y: pd.DataFrame
        The test labels.
    """

    # Read the training and testing data
    test = pd.read_csv(test_path)

    # Drop the first column as it is just the index
    test = test.iloc[:, 1:]
    test_x = test.iloc[:, :-1]
    test_y = test.iloc[:, -1]
    test_y = pd.to_numeric(test_y, errors="coerce")

    # Drop all rows with NaN values
    test_x = test_x.loc[-test_y.isnull()]
    test_y.dropna(inplace=True)

    # Process our test data so the pixels are in the correct format
    test_x = 1 - test_x

    # Read MNIST data
    train = process_pixels(read_idx_file(train_path))
    validation = process_pixels(read_idx_file(validation_path))

    # Split train and validation sets
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    valid_x, valid_y = validation.iloc[:, :-1], validation.iloc[:, -1]

    print("Training set size: {}".format(train_x.shape))
    print("Validation set size: {}".format(valid_x.shape))
    print("Test set size: {}".format(test_x.shape))

    # Normalise all columns
    # The network will converge faster with normalized values.
    train_x = train_x.apply(lambda x: x / 255)
    valid_x = valid_x.apply(lambda x: x / 255)

    min_test_pixel = test_x.min().min()
    max_test_pixel = test_x.max().max()
    test_x = test_x.apply(
        lambda x: x - min_test_pixel / (max_test_pixel - min_test_pixel)
    )

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def process_pixels(data: tuple[np.ndarray, np.ndarray]) -> pd.DataFrame:
    """Converts a tuple of 2D arrays and labels into a dataframe.

    Parameters
    ----------
    data: tuple[np.ndarray, np.ndarray]
        The data.

    Returns
    -------
    data: pd.DataFrame
        The transformed data.

    """

    labels = [str(label) for label in data[1]]
    labels = pd.DataFrame(labels)
    pixels = data[0]
    # Flatten the pixels
    pixels = pixels.reshape(pixels.shape[0], -1)
    # Add the pixels and labels to a dataframe
    data = pd.DataFrame(pixels)
    return pd.concat([data, labels], axis=1)


def read_idx_file(filenames: str) -> np.ndarray:
    """It reads a idx file and returns a numpy array.

    Parameters
    ----------
    filenames: dict
        The files path.

    Returns
    -------
    data: np.ndarray
        The data.
    """
    # Use idx2numpy to read the IDX file into a NumPy array
    images = filenames["images"]
    labels = filenames["labels"]
    mnist_data = idx2numpy.convert_from_file(images)
    mnist_labels = idx2numpy.convert_from_file(labels)
    return mnist_data, mnist_labels


def plot_confusion_matrix(predictions: np.ndarray, test_y: np.ndarray, name: str):
    """It plots the confusion matrix.

    Parameters
    ----------
    predictions: np.ndarray
        The predictions.
    test_y: np.ndarray
        The test labels.
    name: str
        The name of the file.
    """

    test_y = test_y.astype(int)
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(test_y, predictions)
    conf_matrix_percentage = (
        conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    )
    fig = px.imshow(
        conf_matrix_percentage,
        x=[str(i) for i in range(10)],
        y=[str(i) for i in range(10)],
        text_auto=True,
    )
    fig.update_layout(title_text="Confusion matrix", title_x=0.5)
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="True")
    fig.write_html(f"figures/MNIST_confusion_matrix.html")
