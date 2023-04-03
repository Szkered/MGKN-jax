from typing import Any, Dict, List, Optional, Tuple, Any
import jraph

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import scipy.io
from dataclasses import dataclass, fields, asdict

from MGKN_jax.config import DataConfig, TrainConfig
from MGKN_jax.types import Array


def load_file(file_path: str) -> Tuple[Any, bool]:
  is_matlab_format = False
  try:
    data = scipy.io.loadmat(file_path)
    is_matlab_format = True
  except Exception:
    data = h5py.File(file_path)
  return data, is_matlab_format


def read_field(data, field: str, num: int, res: int, is_matlab_format: bool):
  """
  Args:
    r: resolution
  """
  x = data[field]

  if not is_matlab_format:
    x = x[()]
    x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

  x = x.astype(np.float32)  # TODO check this

  return x[:num, ::res, ::res].reshape(num, -1)


def calc_multilevel_connectivity(
  sub_mesh_sizes: List[int],
  grid_samples: List[Array],
  inner_radii: List[float],
  inter_radii: List[float],
):
  """
  Args:
    grid_samples: [(mesh_size, n_dim) for each mesh size]
  """
  pairwise_dist = lambda a, b: jnp.linalg.norm(a - b[:, None], axis=-1)

  # edges within each level
  inner_dists = [pairwise_dist(g, g) for g in grid_samples]
  inner_edge_index = [
    jnp.vstack(jnp.where(dist <= r)) + m
    for dist, r, m in zip(inner_dists, inner_radii, sub_mesh_sizes)
  ]  # (2, num_edges)

  # edges between levels
  inter_dists = [
    pairwise_dist(grid_samples[l], grid_samples[l + 1])
    for l in range(len(grid_samples) - 1)
  ]
  # NOTE: to get the up edges simply flip the down edges
  inter_edge_index = [
    jnp.vstack(jnp.where(dist <= r)) + jnp.array([[0], [m]])
    for dist, r, m in zip(inter_dists, inter_radii, sub_mesh_sizes)
  ]  # (2, num_edges)

  return inner_edge_index, inter_edge_index


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


@dataclass
class ParametricEllipticalPDEInput:
  coeff: Normalizer
  """a in Darcy flow"""

  Kcoeff: Normalizer
  """a_smooth"""

  Kcoeff_x: Normalizer
  """a_gradx"""

  Kcoeff_y: Normalizer
  """a_grady"""


class ParametricEllipticalPDE:
  """
  Load Parametric Elliptical PDE data on a grid.

  The equation is the 2d Darcy Flow

  .. math::
  \begin{alignat*}{3}
    -\nabla \cdot (a(x) \nabla u(x)) &= f(x), \quad    &&x\in (0,1)^2 \\
    u(x)&=0, \quad   &&x\in \partial(0,1)^2
  \end{alignat*}

  where $a(x)$ is the parameter/coefficient of the equation, and we want to learn
  the mapping from $a(x)$ to $u(x)$.

  The dataset consist of parameter $a(x)$ sampled from the distribution
  .. math::
  \(
  \psi_\# \mathcal{N}(0, (-\Delta + 9I)^{-2})
  \)

  coeff has shape (num_sample, num_grid_points)
  """

  in_fields = [
    "coeff",  # a
    "Kcoeff",  # a_smooth
    "Kcoeff_x",  # a_gradx
    "Kcoeff_y",  # a_grady
  ]

  def __init__(self, cfg: DataConfig):
    self.cfg = cfg
    train_data, is_matlab_format = load_file(cfg.train_path)

    self.train_in = ParametricEllipticalPDEInput(
      **{
        f.name: Normalizer(
          read_field(
            train_data, f.name, cfg.n_train, cfg.res, is_matlab_format
          )
        ) for f in fields(ParametricEllipticalPDEInput)
      }
    )

    self.train_out = Normalizer(
      read_field(train_data, "sol", cfg.n_train, cfg.res, is_matlab_format),
      pointwise=True
    )
    test_data, _ = load_file(cfg.test_path)
    self.test_in = ParametricEllipticalPDEInput(
      **{
        f.name: Normalizer(
          read_field(test_data, f.name, cfg.n_test, cfg.res, is_matlab_format)
        ) for f in fields(ParametricEllipticalPDEInput)
      }
    )

    self.test_out = Normalizer(
      read_field(test_data, "sol", cfg.n_test, cfg.res, is_matlab_format),
      pointwise=True
    )

    self.multi_mesh = RandomMultiMeshGenerator(cfg)

  def make_data_gen(self, cfg: TrainConfig):
    rng = jax.random.PRNGKey(cfg.rng_seed)
    key, rng = jax.random.split(rng)

    for _ in range(cfg.epochs):

      for i in range(self.cfg.n_train):
        node_data = [v.normalized[i, :] for v in asdict(self.train_in).values()]
        edge_data = self.train_in.coeff.normalized[i, :]
        for _ in range(self.cfg.n_samples_per_train_data):
          yield self.multi_mesh.sample(
            key, node_data=node_data, edge_data=edge_data
          )


class RandomMultiMeshGenerator:
  """Generate multi-level mesh for multi-level graph representation."""

  def __init__(self, cfg: DataConfig):
    """
    Args:
      domain_boundary: (d, 2) where d is the dimension of the real space
      mesh_size: original mesh size
    """
    self.cfg = cfg

    sub_mesh_sizes = jnp.array(cfg.mesh_cfg.sub_mesh_sizes)
    self.sub_mesh_sizes = sub_mesh_sizes
    self.total_mesh_size = sub_mesh_sizes.sum()
    self.level = len(sub_mesh_sizes)
    self.n_dim = len(cfg.domain_boundary)

    assert len(cfg.mesh_size) == self.n_dim

    if self.n_dim == 1:
      self.n_grid_pts = cfg.mesh_size[0]
      self.grid = jnp.linspace(
        cfg.domain_boundary[0][0], cfg.domain_boundary[0][1], self.n_grid_pts
      ).reshape((self.n_grid_pts, 1))
    else:
      self.n_grid_pts = np.prod(cfg.mesh_size)
      grids = [
        jnp.linspace(lb, up, m)
        for (lb, up), m in zip(cfg.domain_boundary, cfg.mesh_size)
      ]
      self.grid = jnp.vstack([xx.ravel() for xx in jnp.meshgrid(*grids)]).T

  def sample(self, key, node_data: List[Array], edge_data: Array = None):
    """sample non-overlapping multi level/resolution graph from the loaded grid."""
    perm = jax.random.permutation(key, self.n_grid_pts)
    sampled_indices = jnp.split(perm, jnp.cumsum(self.sub_mesh_sizes))[:-1]
    sampled_indices_all = perm[:self.total_mesh_size]

    # node features
    grid_samples = [self.grid[idx] for idx in sampled_indices]
    grid_sample_all = self.grid[sampled_indices_all]

    nodes = jnp.concatenate(
      [grid_sample_all] +
      [d[sampled_indices_all][..., None] for d in node_data],
      axis=-1
    )

    # calculate connectivity
    inner_edge_index, inter_edge_index = calc_multilevel_connectivity(
      self.sub_mesh_sizes, grid_samples, self.cfg.inner_radii,
      self.cfg.inter_radii
    )

    # edge attributes, which is grid_point+aux_edge_data
    inner_edge_attr = [
      self.grid[e_idx.T].reshape(-1, 2 * self.n_dim)
      for e_idx in inner_edge_index
    ]
    if edge_data is not None:
      inner_edge_attr = [
        jnp.concatenate([e_attr, edge_data[e_idx.T]], axis=-1)
        for e_idx, e_attr in zip(inner_edge_index, inner_edge_attr)
      ]

    inter_edge_attr = [
      self.grid[e_idx.T].reshape(-1, 2 * self.n_dim)
      for e_idx in inter_edge_index
    ]
    if edge_data is not None:
      inter_edge_attr = [
        jnp.concatenate([e_attr, edge_data[e_idx.T]], axis=-1)
        for e_idx, e_attr in zip(inter_edge_index, inter_edge_attr)
      ]

    # combine edges from all level into a single tensor
    n_inner_edges = [e.shape[-1] for e in inner_edge_index]
    n_inter_edges = [e.shape[-1] for e in inter_edge_index]

    # (2, n_edges)
    inner_edge_index = jnp.hstack(inner_edge_index)
    inter_edge_index = jnp.hstack(inter_edge_index)
    edge_index = jnp.hstack([inner_edge_index, inter_edge_index])

    # (n_edges, n_attr)
    inner_edge_attr = jnp.vstack(inner_edge_attr)
    inter_edge_attr = jnp.vstack(inter_edge_attr)
    edges = jnp.vstack([inner_edge_attr, inter_edge_attr])

    globals = dict(n_inner_edges=n_inner_edges, n_inter_edges=n_inter_edges)

    gt = jraph.GraphsTuple(
      nodes=nodes,
      edges=edges,
      senders=edge_index[0],
      receivers=edge_index[1],
      n_node=len(nodes),
      n_edge=len(edges),
      globals=globals,
    )

    return gt
