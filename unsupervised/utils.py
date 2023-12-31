import itertools
from multiprocessing import Pool, cpu_count
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px


def read_data(file_path: str) -> pd.DataFrame:
    """Reads the data from a file. It considers numerous file types: csv, txt,
    xlsx, xls, and json.

    Parameters
    ----------
    file_path: str
        The file path.

    Returns
    -------
    pd.DataFrame
        The data.
    """

    if file_path.endswith(".csv") or file_path.endswith(".txt"):
        data = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        data = pd.read_excel(file_path)
    elif file_path.endswith(".json"):
        data = pd.read_json(file_path)
    else:
        raise ValueError(f"File type not supported: {file_path}")
    return data


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """It processes the DataFrame.

    Parameters
    ----------
    data: pd.DataFrame
        The data.

    Returns
    -------
    np.ndarray
        The processed data.
    """

    # One hot encode
    data = pd.get_dummies(data)
    # Drop the nans
    data = data.dropna()
    data = data.values
    # Normalise with min max
    data = (data - data.min()) / (data.max() - data.min())
    return data


def pairwise_distance(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    distance: Callable,
    distance_kwargs: dict = None,
) -> np.ndarray:
    """
     Compute the pairwise distance between two matrices.

    Parameters
    ----------
    matrix_a: np.ndarray
        The first matrix.
    matrix_b: np.ndarray
        The second matrix.
    """

    assert matrix_a.shape[1] == matrix_b.shape[1]

    if distance_kwargs is None:
        distance_kwargs = {}

    # Compute the pairwise distance between two matrices
    distance_matrix = np.zeros((matrix_a.shape[0], matrix_b.shape[0]))
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_b.shape[0]):
            distance_matrix[i, j] = distance(
                vector_a=matrix_a[i], vector_b=matrix_b[j], **distance_kwargs
            )
    return distance_matrix


def plot_distance_matrix(distances: np.ndarray, name: str) -> None:
    """
    It plots the distance matrix.

    Parameters
    ----------
    distances: np.ndarray
        The distance matrix.
    """

    fig = px.imshow(distances)
    # Save to html
    fig.write_html(f"figures/distance_matrix_{name}.html")


def create_nd_grid_list(indices):
    return np.array(indices)


def create_nd_grid(num_partitions: int, dimension: int):
    """
    Create an n-dimensional grid with the specified shape.

    Args:
    dimension: int
        The dimension of the grid.
    num_partitions: int
        The number of partitions in each dimension.

    Returns:
    list: An n-dimensional grid represented as a list of tuples.
    """

    try:
        cpus = cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default

    shape = [num_partitions + 1] * dimension

    print(f"Using {cpus} processes to create grid")
    # Use multiprocessing to parallelize grid creation
    with Pool(processes=cpus) as pool:
        results = pool.map(
            create_nd_grid_list,
            itertools.product(*[range(dim_size) for dim_size in shape]),
        )
    return np.array(results) / num_partitions
