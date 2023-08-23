"""It contains the functions to read data files."""

import numpy as np
import pandas as pd
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
