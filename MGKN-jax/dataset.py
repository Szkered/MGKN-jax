import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from scipy.ndimage import gaussian_filter
from typing import Tuple, Any


def load_file(file_path: str) -> Tuple[Any, bool]:
  is_matlab_format = False
  try:
    data = scipy.io.loadmat(file_path)
    is_matlab_format = True
  except Exception:
    data = h5py.File(file_path)
  return data, is_matlab_format


def read_field(data, field: str, num: int, r: int, is_matlab_format: bool):
  x = data[field]

  if not is_matlab_format:
    x = x[()]
    x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

  x = x.astype(np.float32)  # TODO check this

  return x[:num, ::r, ::r].reshape(num, -1)


class Normalizer:

  def __init__(self, x, pointwise: bool = False, eps: float = 1e-5):
    if pointwise:
      self.mean = np.mean(x, 0)
      self.std = np.std(x, 0)
    else:
      self.mean = np.mean(x)
      self.std = np.std(x)
    self.eps = eps
    self.normalized = self.normalize(x)

  def normalize(self, x):
    x = (x - self.mean) / (self.std + self.eps)
    return x

  def unnormalize(self, x, sample_idx=None):
    if sample_idx is None:
      std = self.std + self.eps  # n
      mean = self.mean
    else:
      if len(self.mean.shape) == len(sample_idx[0].shape):
        std = self.std[sample_idx] + self.eps  # batch*n
        mean = self.mean[sample_idx]
      if len(self.mean.shape) > len(sample_idx[0].shape):
        std = self.std[:, sample_idx] + self.eps  # T*batch*n
        mean = self.mean[:, sample_idx]

    # x is in shape of batch*n or T*batch*n
    x = (x * std) + mean
    return x


class ParametricEllipticalPDE:

  def __init__(
    self,
    train_path: str,
    test_path: str,
    n_train: int = 100,
    n_test: int = 100,
    r: int = 1
  ):
    in_fields = ["coeff", "Kcoeff", "Kcoeff_x", "Kcoeff_y"]
    train_data, is_matlab_format = load_file(train_path)
    self.train_in = {
      k: Normalizer(read_field(train_data, k, n_train, r, is_matlab_format))
      for k in in_fields
    }
    self.train_out = Normalizer(
      read_field(train_data, "sol", n_train, r, is_matlab_format),
      pointwise=True
    )
    test_data = load_file(test_path)
    self.test_in = {
      k: Normalizer(read_field(test_data, k, n_test, r, is_matlab_format))
      for k in in_fields
    }
    self.test_out = Normalizer(
      read_field(test_data, "sol", n_test, r, is_matlab_format), pointwise=True
    )
