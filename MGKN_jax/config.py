from typing import Iterable, Literal, Tuple, Union, Optional, List

from ml_collections import ConfigDict
from pydantic.dataclasses import dataclass

config = dict(validate_assignment=True)


@dataclass(config=config)
class MLPConfig:
  activation: Literal["tanh", "silu", "elu", "relu"] = "relu"

  init_bias_scale: float = 0.0

  init_weights_scale: Literal["fan_in", "fan_out", "fan_avg"] = "fan_avg"

  init_weights_distribution: Literal["normal", "truncated_normal",
                                     "uniform"] = "uniform"

  residual: bool = False

  linear_out: bool = True

  output_bias: bool = False


@dataclass(config=config)
class NNConvConfig:
  width: int = 64

  aggr: Literal["mean"] = "mean"
  """message aggregation method"""


@dataclass(config=config)
class MeshConfig:

  sub_mesh_sizes: Tuple = (2400, 1600, 400, 100, 25)
  """mesh sizes for each resolution level, ordering from finest to coarsest"""


@dataclass(config=config)
class MGKNConfig:
  ker_width: int = 256

  depth: int = 4

  ker_in: int = 6

  in_width: int = 6
  """node_features"""

  out_width: int = 1

  mesh_cfg: MeshConfig = MeshConfig()

  mlp_cfg: MLPConfig = MLPConfig()

  nnconv_cfg: NNConvConfig = NNConvConfig()


# @dataclass(config=config)
# class StandardOptimizerConfig:
#   name: Literal["adam", "rmsprop_momentum", "sgd"] = "adam"

#   learning_rate: float = 1e-3

#   lr_schedule: Union[InverseLRScheduleConfig,
#                      ConstantLRSchedule] = InverseLRScheduleConfig()
#   """Schedule for the learning rate decay"""

#   scaled_modules: Optional[List[str]] = None
#   """List of parameters for which the learning rate is being scaled."""

#   scale_lr = 1.0
#   """Factor which to apply to the learning rates of specified modules"""

# @dataclass(config=config)
# class OptimizationConfig:

#   optimizer: StandardOptimizerConfig = StandardOptimizerConfig()


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
  mesh_size = [int(((241 - 1) / res) + 1)] * 2
  inner_radii = [0.5 / 8 * 1.41, 0.5 / 8, 0.5 / 4, 0.5 / 2, 0.5]
  inter_radii = [0.5 / 8 * 1.1, 0.5 / 8 * 1.41, 0.5 / 4 * 1.41, 0.5 / 2 * 1.41]
  """NOTE: this is hardcoded for the dataset"""

  mesh_cfg: MeshConfig = MeshConfig()


@dataclass(config=config)
class TrainConfig:
  data_cfg: DataConfig = DataConfig()

  epochs: int = 200

  learning_rate: float = 0.1 / data_cfg.n_train

  scheduler_step: int = 10

  scheduler_gamma: float = 0.8

  rng_seed: int = 137


def get_config() -> ConfigDict:
  cfg = ConfigDict()
  cfg.mlp_cfg = MLPConfig()
  cfg.mgkn_cfg = MGKNConfig()
  cfg.train_cfg = TrainConfig()
  return cfg
