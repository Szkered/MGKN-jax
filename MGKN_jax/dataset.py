from dataclasses import asdict, dataclass, fields
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

import h5py
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import scipy.io

from MGKN_jax.config import DataConfig, MultiMeshConfig, TrainConfig

Array = np.ndarray


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
  """Compute connectivity between nodes within each resolution and
  between resolutions using Euclidean distance between nodes.

  Args:
    sub_mesh_sizes: mesh size for each resolution level
    grid_samples: [(mesh_size, n_dim) for each mesh size]
    inner_radii: create an edge between a pair of nodes within the same
      level, if the distance between them is less than this radii
    inter_radii: create an edge between a pair of nodes within the same
      level, if the distance between them is less than this radii

  Returns:
    inner_edge_index, inter_edge_index in (senders, receivers) format
  """
  pairwise_dist = lambda a, b: np.linalg.norm(a - b[:, None], axis=-1)

  # edges within each level
  inner_dists = [pairwise_dist(g, g) for g in grid_samples]

  offset = np.concatenate([np.array([0]), sub_mesh_sizes.cumsum()])

  inner_edge_index = [
    np.vstack(np.where(dist <= r)) + m
    for dist, r, m in zip(inner_dists, inner_radii, offset)
  ]  # (2, num_edges)

  # edges between levels
  inter_dists = [
    pairwise_dist(grid_samples[l], grid_samples[l + 1]).T
    for l in range(len(grid_samples) - 1)
  ]
  # We computes the down edges in (senders, receivers) format
  # to get the up edges simply flip the down edges
  inter_edge_index = [
    np.vstack(np.where(dist <= r)) + np.array([[offset[i]], [offset[i + 1]]])
    for i, (dist, r) in enumerate(zip(inter_dists, inter_radii))
  ]  # (2, num_edges)

  return inner_edge_index, inter_edge_index


class Normalizer:

  def __init__(self, x, pointwise: bool = False, eps: float = 1e-5):
    self.pointwise = pointwise
    if pointwise:  # normalize across datasets
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

    node_data = {
      "inputs": [v.normalized for v in asdict(self.train_in).values()],
      "outputs": self.train_out.normalized
    }
    edge_data = self.train_in.coeff.normalized
    self.multi_mesh = RandomMultiMeshGenerator(
      cfg, node_data, edge_data, self.train_out
    )

    self.n_data_per_epoch = self.cfg.n_train * self.cfg.n_samples_per_train_data
    if cfg.static_grids:
      rng = jax.random.PRNGKey(cfg.rng_seed)
      self.samples = []
      for data_idx in tqdm(range(self.cfg.n_train)):
        for _ in range(self.cfg.n_samples_per_train_data):
          key, rng = jax.random.split(rng)
          sample = self.multi_mesh.sample(key, data_idx)
          self.samples.append(sample)

  def get_init_data(self) -> jraph.GraphsTuple:
    return self.multi_mesh.sample(jax.random.PRNGKey(137), 0)

  def make_data_gen(self, cfg: TrainConfig):
    rng = jax.random.PRNGKey(cfg.rng_seed)

    if self.cfg.static_grids:
      for _ in range(cfg.epochs):
        key, rng = jax.random.split(rng)
        data_idx_perms = jax.random.permutation(key, self.n_data_per_epoch)
        for data_idx in data_idx_perms:
          yield self.samples[data_idx]

    else:
      for _ in range(cfg.epochs):
        key, rng = jax.random.split(rng)
        data_idx_perms = jax.random.permutation(key, self.cfg.n_train)
        # for data_idx in range(self.cfg.n_train):
        for data_idx in data_idx_perms:
          for _ in range(self.cfg.n_samples_per_train_data):
            key, rng = jax.random.split(rng)
            yield self.multi_mesh.sample(key, data_idx)


class RandomMultiMeshGenerator:
  """Generate multi-level mesh for multi-level graph representation."""

  def __init__(
    self, cfg: DataConfig, node_data: Dict[str, List[Array]], edge_data: Array,
    train_out
  ):
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
    self.train_out = train_out

    assert len(cfg.mesh_size) == self.n_dim

    if self.n_dim == 1:
      self.n_grid_pts = cfg.mesh_size[0]
      self.grid = np.linspace(
        cfg.domain_boundary[0][0], cfg.domain_boundary[0][1], self.n_grid_pts
      ).reshape((self.n_grid_pts, 1))
    else:
      self.n_grid_pts = np.prod(cfg.mesh_size)
      grids = [
        np.linspace(lb, up, m)
        for (lb, up), m in zip(cfg.domain_boundary, cfg.mesh_size)
      ]
      self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    self.node_data = node_data
    self.edge_data = edge_data

  def sample(self, key, data_idx: int) -> jraph.GraphsTuple:
    """sample non-overlapping multi level/resolution graph from the loaded grid."""
    # sample nodes
    perm = jax.random.permutation(key, self.n_grid_pts)
    sampled_indices = np.split(perm, np.cumsum(self.sub_mesh_sizes))[:-1]
    sampled_indices_all = perm[:self.total_mesh_size]

    # node features: grid_points + all features
    grid_samples = [self.grid[idx] for idx in sampled_indices]
    grid_sample_all = self.grid[sampled_indices_all]

    inputs = np.concatenate(
      [grid_sample_all] + [
        d[data_idx, sampled_indices_all][..., None]
        for d in self.node_data["inputs"]
      ],
      axis=-1
    )
    outputs = self.node_data["outputs"][data_idx, sampled_indices[0]]
    nodes = dict(inputs=inputs, outputs=outputs)

    edge_sample = self.edge_data[data_idx, sampled_indices_all]

    # calculate connectivity
    inner_edge_index, inter_edge_index = calc_multilevel_connectivity(
      self.sub_mesh_sizes, grid_samples, self.cfg.mesh_cfg.inner_radii,
      self.cfg.mesh_cfg.inter_radii
    )

    # edge attributes, which is grid_point + aux_edge_data
    inner_edge_attr = [
      grid_sample_all[e_idx.T].reshape(-1, 2 * self.n_dim)
      for e_idx in inner_edge_index
    ]
    inner_edge_attr = [
      np.concatenate([e_attr, edge_sample[e_idx.T]], axis=-1)
      for e_idx, e_attr in zip(inner_edge_index, inner_edge_attr)
    ]

    inter_edge_attr = [
      grid_sample_all[e_idx.T].reshape(-1, 2 * self.n_dim)
      for e_idx in inter_edge_index
    ]
    inter_edge_attr = [
      np.concatenate([e_attr, edge_sample[e_idx.T]], axis=-1)
      for e_idx, e_attr in zip(inter_edge_index, inter_edge_attr)
    ]

    senders = dict(
      inter=[i[0] for i in inter_edge_index],
      inner=[i[0] for i in inner_edge_index]
    )
    receivers = dict(
      inter=[i[1] for i in inter_edge_index],
      inner=[i[1] for i in inner_edge_index]
    )

    edges = dict(
      inter=inter_edge_attr,
      inner=inner_edge_attr,
    )

    # count nodes for each level
    n_inner_nodes = np.array(self.sub_mesh_sizes)
    n_inter_nodes = n_inner_nodes[:-1] + n_inner_nodes[1:]
    n_node = dict(
      inter=np.split(n_inter_nodes, self.level - 1),
      inner=np.split(n_inner_nodes, self.level)
    )

    # combine edges from all level into a single tensor
    n_inner_edges = np.array([e.shape[-1] for e in inner_edge_index])
    n_inter_edges = np.array([e.shape[-1] for e in inter_edge_index])
    n_edge = dict(
      inter=np.split(n_inter_edges, self.level - 1),
      inner=np.split(n_inner_edges, self.level)
    )

    gt = jraph.GraphsTuple(
      nodes=nodes,
      edges=edges,
      senders=senders,
      receivers=receivers,
      n_node=n_node,
      n_edge=n_edge,
      globals=(
        self.train_out.mean[sampled_indices[0]],
        self.train_out.std[sampled_indices[0]],
      ),
    )

    return gt
