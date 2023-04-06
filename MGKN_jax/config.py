from typing import Iterable, Literal, Tuple, Union

from ml_collections import ConfigDict
from pydantic.dataclasses import dataclass

config = dict(validate_assignment=True)


@dataclass(config=config)
class MultiMeshConfig:

  sub_mesh_sizes: Tuple = (2400, 1600, 400, 100, 25)
  """mesh sizes for each resolution level, ordering from finest to coarsest"""

  inner_radii = [
    0.5 / 8 * 1.41,
    0.5 / 8,
    0.5 / 4,
    0.5 / 2,
    0.5,
  ]
  inter_radii = [
    0.5 / 8 * 1.1,
    0.5 / 8 * 1.41,
    0.5 / 4 * 1.41,
    0.5 / 2 * 1.41,
  ]
  """Radii for constructing the multilevel mesh"""


@dataclass(config=config)
class MultiMeshConfigS1:
  sub_mesh_sizes: Tuple = (400, 100, 25)
  """mesh sizes for each resolution level, ordering from finest to coarsest"""

  inner_radii = [
    0.5 / 4,
    0.5 / 2,
    0.5,
  ]
  inter_radii = [
    0.5 / 4 * 1.41,
    0.5 / 2 * 1.41,
  ]
  """Radii for constructing the multilevel mesh"""


@dataclass(config=config)
class MultiMeshConfigS2:
  sub_mesh_sizes: Tuple = (100, 25)
  """mesh sizes for each resolution level, ordering from finest to coarsest"""

  inner_radii = [
    0.5 / 2,
    0.5,
  ]
  inter_radii = [
    0.5 / 2 * 1.41,
  ]
  """Radii for constructing the multilevel mesh"""


MeshConfig = Union[MultiMeshConfig, MultiMeshConfigS1, MultiMeshConfigS2]


@dataclass(config=config)
class DataConfig:
  train_path: str = 'data/piececonst_r241_N1024_smooth1.mat'

  test_path: str = 'data/piececonst_r241_N1024_smooth2.mat'

  n_train: int = 100

  n_test: int = 100

  n_samples_per_train_data: int = 1

  res: int = 1
  """resolution of grid"""

  domain_boundary = ((0, 1), (0, 1))
  """boundary of domain"""

  grid_size_per_dim: int = 241
  """grid size per dimension. Assuming square domain"""

  mesh_size = [(grid_size_per_dim - 1) // res + 1] * len(domain_boundary)
  """total number of grid points"""

  mesh_cfg: MeshConfig = MultiMeshConfigS2()


@dataclass(config=config)
class MLPConfig:
  activation: Literal["tanh", "silu", "elu", "relu"] = "relu"

  init_bias_scale: float = 0.0

  init_weights_scale: Literal["fan_in", "fan_out", "fan_avg"] = "fan_avg"

  init_weights_distribution: Literal["normal", "truncated_normal",
                                     "uniform"] = "uniform"

  residual: bool = False

  linear_out: bool = True

  output_bias: bool = True


@dataclass(config=config)
class NNConvConfig:
  width: int = 64
  """size of hidden node embedding"""

  aggr: Literal["mean", "sum"] = "mean"
  """message aggregation method"""


@dataclass(config=config)
class MGKNConfig:
  finest_ker_width: int = 256
  """size of node embedding of the finest resolution"""

  depth: int = 4
  """number of v-cycles"""

  ker_in: int = 6
  """size of input node features"""

  ker_out: int = 1
  """size of output node features"""

  mesh_cfg: MeshConfig = MultiMeshConfigS2()

  mlp_cfg: MLPConfig = MLPConfig()

  nnconv_cfg: NNConvConfig = NNConvConfig()


@dataclass(config=config)
class TrainConfig:
  data_cfg: DataConfig = DataConfig()

  epochs: int = 200

  learning_rate: float = 0.1 / data_cfg.n_train

  lr_decay: bool = True

  l2_reg: float = 5e-4

  scheduler_step: int = 10 * data_cfg.n_train
  """scheduler step in epochs"""

  scheduler_gamma: float = 0.8
  """decay rate"""

  rng_seed: int = 137

  batch_size: int = 1
  """batching not supported yet"""

  optimizer: Literal["adam", "sgd"] = "adam"


def get_config() -> ConfigDict:
  cfg = ConfigDict()
  cfg.mlp_cfg = MLPConfig()
  cfg.mgkn_cfg = MGKNConfig()
  cfg.train_cfg = TrainConfig()
  return cfg
