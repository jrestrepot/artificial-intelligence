"""It contains the functions to read data files."""

import struct as st

import idx2numpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.io as sio
import torch
from torch import Tensor


def format_input(data: pd.DataFrame | np.ndarray, transpose: bool = False) -> Tensor:
    """It formats the input data.

    Parameters
    ----------
    data: pd.DataFrame | np.ndarray
        The data.

    Returns
    -------
    x: Tensor
        The input tensor.
    transpose: bool
        Whether to transpose the input data.
    """

    if isinstance(data, pd.DataFrame):
        data = torch.tensor(data.values, dtype=torch.float32)
    if isinstance(data, pd.Series):
        data = torch.tensor(data.values, dtype=torch.float32)
        data = data.resize(data.shape[0], 1)
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    if transpose:
        data = data.T
    return data


def read_mat_file(file_path: str) -> pd.DataFrame:
    """It reads a mat file and returns a DataFrame.

    Parameters
    ----------
    file_path: str
        The file path.

    Returns
    -------
    data: pd.DataFrame
        The data.
    """

    mat = sio.loadmat(file_path)
    data = pd.DataFrame()
    for i in list(mat.keys() - ["__header__", "__version__", "__globals__"]):
        data[i] = mat[i].reshape(-1)
    return data


def read_txt(data_path: str, sep: str = ",") -> pd.DataFrame:
    """It reads a txt file and returns a DataFrame.

    Parameters
    ----------
    data_path: str
        The file path.
    sep: str
        The separator.

    Returns
    -------
    data: pd.DataFrame
        The data.
    """

    data = pd.read_csv(data_path, sep=sep, header=None)
    return data


def normalize_to_hypercube(data: pd.DataFrame) -> pd.DataFrame:
    """It normalizes the data to the hypercube [0, 1].

    Parameters
    ----------
    data: pd.DataFrame
        The data.

    Returns
    -------
    normalized_data: pd.DataFrame
        The normalized data.
    """

    # Normalize each column
    normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    return normalized_data


def train_test_val_split(data: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    It splits the data into train, test and validation sets using the uniform
    distribution.

    Parameters
    ----------
    data: pd.DataFrame
        The data.

    Returns
    -------
    train: pd.DataFrame
        The training set.
    """

    # Set the seed for reproducibility
    np.random.seed(42)

    indexes = np.arange(len(data))
    # Sample the indexes (np.random.choice uses the uniform distribution)
    train = np.random.choice(indexes, size=round(0.6 * len(data)), replace=False)
    test = np.random.choice(indexes, size=round(0.2 * len(data)), replace=False)
    val = np.random.choice(indexes, size=round(0.2 * len(data)), replace=False)

    # Sort the indexes
    train.sort()
    test.sort()
    val.sort()
    return data.iloc[train, :], data.iloc[test, :], data.iloc[val, :]


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


def plot_train_test_val_split(
    train_set: pd.DataFrame, test_set: pd.DataFrame, val_set: pd.DataFrame
) -> None:
    """It plots the train, test and validation sets in different colors.

    Parameters
    ----------
    train_set: pd.DataFrame
        The training set.
    test_set: pd.DataFrame
        The test set.
    val_set: pd.DataFrame
        The validation set.
    """

    fig = go.Figure()

    # Reindex the dataframes so that they're easier to plot
    max_index = np.max([train_set.index[-1], test_set.index[-1], val_set.index[-1]])
    x_index = np.arange(0, max_index + 1)
    reindex_train_set = train_set.reindex(x_index, fill_value=np.nan)
    reindex_test_set = test_set.reindex(x_index, fill_value=np.nan)
    reindex_val_set = val_set.reindex(x_index, fill_value=np.nan)

    # Plot each column
    for column in train_set.columns:
        fig.add_trace(
            go.Scatter(
                x=x_index,
                y=reindex_train_set.loc[:, column],
                mode="markers",
                marker=dict(size=3, opacity=0.5),
                name=f"{column}, Training Set",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_index,
                y=reindex_test_set.loc[:, column],
                mode="markers",
                marker=dict(size=3, opacity=0.5),
                name=f"{column}, Test Set",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_index,
                y=reindex_val_set.loc[:, column],
                mode="markers",
                marker=dict(size=3, opacity=0.5),
                name=f"{column}, Validation Set",
            )
        )

    fig.update_layout(
        title=f"Data split",
    )

    # Save figure into a html
    fig.write_html(f"figures/train_test_val_split.html")
