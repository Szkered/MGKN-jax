from typing import Callable, Iterable, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import numpy as jnp

from MGKN_jax.config import MLPConfig, NNConvConfig, MGKNConfig


class MLP(hk.Module):

  def __init__(
    self,
    output_sizes: Iterable[int],
    config: MLPConfig,
    name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.config = config
    self.output_sizes = output_sizes

    self.activation = dict(
      tanh=jnp.tanh, silu=jax.nn.silu, elu=jax.nn.elu, relu=jax.nn.relu
    )[config.activation]
    self.init_w = hk.initializers.VarianceScaling(
      1.0, config.init_weights_scale, config.init_weights_distribution
    )
    self.init_b = hk.initializers.TruncatedNormal(config.init_bias_scale)

  def __call__(self, x):
    for i, output_size in enumerate(self.output_sizes):
      is_output_layer = i == (len(self.output_sizes) - 1)
      y = hk.Linear(
        output_size, self.config.output_bias or not is_output_layer,
        self.init_w, self.init_b, f"linear_{i}"
      )(
        x
      )
      if not (is_output_layer and self.config.linear_out):
        y = self.activation(y)
      if self.config.residual and (x.shape == y.shape):
        x = (y + x) / np.sqrt(2.0)
      else:
        x = y
    return x


class NNConv(hk.Module):

  def __init__(
    self,
    nn: Callable,
    config: NNConvConfig,
    name: Optional[str] = None,
    **kwargs
  ):
    super().__init__(name=name)
    self.nn = nn
    self.config = config

  def __call__(self, x, edge_index, edge_attr=None, size=None):
    """
    Args:
      x: (num_nodes, N)
      edge_index: (2, num_edges)
      edge_attr: (num_edges, E)
      size: [num_nodes, num_nodes]
    """
    # calc message
    weight = self.nn(edge_attr)  # (num_edges, E')
    weight = weight.reshape(-1, self.config.width, self.config.width)
    # i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
    x_j = x[edge_index[1]][:, None]  # (num_edges, 1, E)
    msgs = jnp.matmul(x_j, weight).squeeze(1)  # # (num_edges, OC)

    # TODO: aggregate (neighbours)
    if self.config.aggr == "mean":
      msg = jnp.mean(msgs)
    else:
      raise NotImplementedError

    updated_x = msg

    return updated_x


class MGKN(hk.Module):

  def __init__(
    self,
    config: MGKNConfig,
    name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.config = config
    self.points_total = np.sum(config.points)

    self.ker_widths = [
      self.config.ker_width // (2**(l + 1)) for l in range(self.config.level)
    ]

  def __call__(self, data):
    edge_index_down, edge_attr_down, range_down = data.edge_index_down, data.edge_attr_down, data.edge_index_down_range
    edge_index_mid, edge_attr_mid, range_mid = data.edge_index_mid, data.edge_attr_mid, data.edge_index_range
    edge_index_up, edge_attr_up, range_up = data.edge_index_up, data.edge_attr_up, data.edge_index_up_range

    width = self.config.nnconv_config.width

    x = MLP([width], self.config.mlp_config)(data.x)

    # DOWNWARD: K12, K23, K34 ...
    for l in range(self.config.level):
      kernel_l = MLP(
        [self.config.ker_in, self.ker_widths[l], width**2],
        self.config.mlp_config
      )
      x = x + NNConv(
        kernel_l, self.config.nnconv_config, name=f"K{l+1}{l+2}"
      )(
        x, edge_index_down[:, range_down[l, 0]:range_down[l, 1]],
        edge_attr_down[range_down[l, 0]:range_down[l, 1], :]
      )
      x = jax.nn.relu(x)

    # UPWARD: (K11, K21), (K22, K32), (K33, K43) ...
    for l in reversed(range(self.config.level)):
      # K11, K22, K33, ...
      kernel_l_ii = MLP(
        [self.config.ker_in, self.ker_widths[l], self.ker_widths[l], width**2],
        self.config.mlp_config
      )
      x = x + NNConv(
        kernel_l_ii, self.config.nnconv_config, name=f"K{l+1}{l+1}"
      )(
        x, edge_index_mid[:, range_mid[l, 0]:range_mid[l, 1]],
        edge_attr_mid[range_mid[l, 0]:range_mid[l, 1], :]
      )
      x = jax.nn.relu(x)

      if l > 0:  # from previous (coarser) level: K21, K32, K43, ...
        kernel_l_ji = MLP(
          [self.config.ker_in, self.ker_widths[l], width**2],
          self.config.mlp_config
        )
        x = x + NNConv(
          kernel_l_ji, self.config.nnconv_config, name=f"K{l+2}{l+1}"
        )(
          x, edge_index_up[:, range_up[l - 1, 0]:range_up[l - 1, 1]],
          edge_attr_up[range_up[l - 1, 0]:range_up[l - 1, 1], :]
        )

        x = jax.nn.relu(x)

    x = MLP([self.config.ker_width], self.config.mlp_config)(
      x[:self.config.points[0]]
    )
    x = jax.nn.relu(x)
    x = MLP([1], self.config.mlp_config)(x)
    return x
