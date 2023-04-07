import itertools
import random
from dataclasses import asdict, dataclass, fields
from typing import (
  Any, Dict, Iterable, Iterator, List, NamedTuple, Tuple, TypeVar
)

import h5py
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import scipy.io
from absl import logging
from tqdm import tqdm

from MGKN_jax.config import DataConfig, TrainConfig

Array = np.ndarray
_T = TypeVar('_T')


class Data(NamedTuple):
  """Container for the training data."""
  input: List[jraph.GraphsTuple]
  label: Array


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


def _repeat_and_shuffle(iterable: Iterable[_T], *,
                        buffer_size: int) -> Iterator[_T]:
  """Infinitely repeats, caches, and shuffles data from `iterable`."""
  ds = itertools.cycle(iterable)
  buffer = [next(ds) for _ in range(buffer_size)]
  random.shuffle(buffer)
  for item in ds:
    idx = random.randint(0, buffer_size - 1)  # Inclusive.
    result = buffer[idx]
    buffer[idx] = item
    yield result


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

  def __init__(self, cfg: DataConfig, rng):
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
      logging.info("creating multilevel mesh...")
      self.data = []
      for data_idx in tqdm(range(self.cfg.n_train)):
        for _ in range(self.cfg.n_samples_per_train_data):
          key, rng = jax.random.split(rng)
          self.data.append(self.multi_mesh.sample(key, data_idx))

  def make_data_gen(self, cfg: TrainConfig):
    ds = _repeat_and_shuffle(
      self.data, buffer_size=cfg.batch_size * cfg.num_shuffle_batches
    )
    while True:
      yield next(ds)


def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding."""
  y = 2
  while y < x:
    y *= 2
  return y


def pad_graph_to_nearest_power_of_two(
  graphs_tuple: jraph.GraphsTuple, edge_only: bool = True
) -> jraph.GraphsTuple:
  """Pads a batched `GraphsTuple` to the nearest power of two.

  Jax will re-jit your graphnet every time a new graph shape is encountered.
  In the limit, this means a new compilation every training step, which
  will result in *extremely* slow training. To prevent this, pad each
  batch of graphs to the nearest power of two. Since jax maintains a cache
  of compiled programs, the compilation cost is amortized.

  For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
  would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)
  And since padding is accomplished using `jraph.pad_with_graphs`, an extra
  graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs

  Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).
    edge_only: If true only pad edges. Use this when nodes are already
      of the same size
  Returns:
    A graphs_tuple batched to the nearest power of two.
  """
  # Add 1 since we need at least one padding node for pad_with_graphs.
  if edge_only:
    pad_nodes_to = jnp.sum(graphs_tuple.n_node) + 1
  else:
    pad_nodes_to = _nearest_bigger_power_of_two(
      jnp.sum(graphs_tuple.n_node)
    ) + 1
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
  return jraph.pad_with_graphs(
    graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
  )


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

  def sample(self, key, data_idx: int) -> Tuple[List[jraph.GraphsTuple], Array]:
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

    edge_sample = self.edge_data[data_idx, sampled_indices_all]

    # calculate connectivity
    inner_edge_indices, inter_edge_indices = calc_multilevel_connectivity(
      self.sub_mesh_sizes, grid_samples, self.cfg.mesh_cfg.inner_radii,
      self.cfg.mesh_cfg.inter_radii
    )

    # edge attributes, which is grid_point + aux_edge_data
    inner_edge_attrs = [
      grid_sample_all[e_idx.T].reshape(-1, 2 * self.n_dim)
      for e_idx in inner_edge_indices
    ]
    inner_edge_attrs = [
      np.concatenate([e_attr, edge_sample[e_idx.T]], axis=-1)
      for e_idx, e_attr in zip(inner_edge_indices, inner_edge_attrs)
    ]

    inter_edge_attrs = [
      grid_sample_all[e_idx.T].reshape(-1, 2 * self.n_dim)
      for e_idx in inter_edge_indices
    ]
    inter_edge_attrs = [
      np.concatenate([e_attr, edge_sample[e_idx.T]], axis=-1)
      for e_idx, e_attr in zip(inter_edge_indices, inter_edge_attrs)
    ]

    edge_attrs = inner_edge_attrs + inter_edge_attrs
    edge_indices = inner_edge_indices + inter_edge_indices

    stats = np.vstack(
      [
        self.train_out.mean[sampled_indices[0]],
        self.train_out.std[sampled_indices[0]]
      ]
    )

    graphs = []
    for i, (edges, edge_index) in enumerate(zip(edge_attrs, edge_indices)):
      nodes = inputs if i == 0 else np.zeros_like(inputs)
      n_node = np.array([len(nodes)])
      n_edge = np.array([len(edges)])
      senders = edge_index[0]
      receivers = edge_index[1]
      globals = stats if i == 0 else np.zeros_like(stats)
      gt = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=globals,  # not used
      )
      gt = pad_graph_to_nearest_power_of_two(gt)
      graphs.append(gt)

    return Data(input=graphs, label=outputs)
