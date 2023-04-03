from typing import Callable, Iterable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from jax import numpy as jnp

from MGKN_jax.config import MGKNConfig, MLPConfig, NNConvConfig


class MLP(hk.Module):

  def __init__(
    self,
    output_sizes: Iterable[int],
    cfg: MLPConfig,
    name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.cfg = cfg
    self.output_sizes = output_sizes

    self.activation = dict(
      tanh=jnp.tanh, silu=jax.nn.silu, elu=jax.nn.elu, relu=jax.nn.relu
    )[cfg.activation]
    self.init_w = hk.initializers.VarianceScaling(
      1.0, cfg.init_weights_scale, cfg.init_weights_distribution
    )
    self.init_b = hk.initializers.TruncatedNormal(cfg.init_bias_scale)

  def __call__(self, x):
    for i, output_size in enumerate(self.output_sizes):
      is_output_layer = i == (len(self.output_sizes) - 1)
      y = hk.Linear(
        output_size, self.cfg.output_bias or not is_output_layer, self.init_w,
        self.init_b, f"linear_{i}"
      )(
        x
      )
      if not (is_output_layer and self.cfg.linear_out):
        y = self.activation(y)
      if self.cfg.residual and (x.shape == y.shape):
        x = (y + x) / np.sqrt(2.0)
      else:
        x = y
    return x


class NNConv(hk.Module):

  def __init__(
    self,
    nn: Callable,
    cfg: NNConvConfig,
    name: Optional[str] = None,
    **kwargs
  ):
    super().__init__(name=name)
    self.nn = nn
    self.cfg = cfg

  def __call__(self, x, senders, receivers, edge_attr):
    """
    Args:
      x: (num_nodes, N)
      edge_index: (2, num_edges)
      edge_attr: (num_edges, E)
      size: [num_nodes, num_nodes]
    """
    # calculate message
    weight = self.nn(edge_attr)  # (num_edges, E')
    weight = weight.reshape(-1, self.cfg.width, self.cfg.width)
    # i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
    x_j = x[senders][:, None]  # (num_edges, 1, E)
    msgs = jnp.matmul(x_j, weight).squeeze(1)  # (num_edges, OC)

    # aggregate neighbours
    if self.cfg.aggr == "mean":
      msg = jraph.segment_mean(msgs, receivers)
    elif self.cfg.aggr == "sum":
      msg = jraph.segment_sum(msgs, receivers)
    else:
      raise NotImplementedError

    updated_x = msg

    return updated_x


class MGKN(hk.Module):

  def __init__(
    self,
    cfg: MGKNConfig,
    name: Optional[str] = "MGKN",
  ):
    super().__init__(name=name)
    self.cfg = cfg
    self.points_total = np.sum(cfg.mesh_cfg.sub_mesh_sizes)
    self.level = len(cfg.mesh_cfg.sub_mesh_sizes)
    self.finest_mesh_size = cfg.mesh_cfg.sub_mesh_sizes[0]

    self.ker_widths = [
      self.cfg.ker_width // (2**(l + 1)) for l in range(self.level)
    ]

  def __call__(self, data: jraph.GraphsTuple):
    n_inner_edges_total = sum(data.globals.n_inner_edges)
    n_inter_edges_total = sum(data.globals.n_inter_edges)
    n_edges = n_inner_edges_total + n_inter_edges_total
    breakpoint()

    edge_index_down, edge_attr_down, range_down = data.edge_index_down, data.edge_attr_down, data.edge_index_down_range
    edge_index_mid, edge_attr_mid, range_mid = data.edge_index_mid, data.edge_attr_mid, data.edge_index_range
    edge_index_up, edge_attr_up, range_up = data.edge_index_up, data.edge_attr_up, data.edge_index_up_range

    width = self.cfg.nnconv_cfg.width

    x = MLP([width], self.cfg.mlp_cfg)(data.x)

    # DOWNWARD: K12, K23, K34 ...
    for l in range(self.level):
      kernel_l = MLP(
        [self.cfg.ker_in, self.ker_widths[l], width**2], self.cfg.mlp_cfg
      )
      x = x + NNConv(
        kernel_l, self.cfg.nnconv_cfg, name=f"K{l+1}{l+2}"
      )(
        x, edge_index_down[:, range_down[l, 0]:range_down[l, 1]],
        edge_attr_down[range_down[l, 0]:range_down[l, 1], :]
      )
      x = jax.nn.relu(x)

    # UPWARD: (K11, K21), (K22, K32), (K33, K43) ...
    for l in reversed(range(self.level)):
      # K11, K22, K33, ...
      kernel_l_ii = MLP(
        [self.cfg.ker_in, self.ker_widths[l], self.ker_widths[l], width**2],
        self.cfg.mlp_cfg
      )
      x = x + NNConv(
        kernel_l_ii, self.cfg.nnconv_cfg, name=f"K{l+1}{l+1}"
      )(
        x, edge_index_mid[:, range_mid[l, 0]:range_mid[l, 1]],
        edge_attr_mid[range_mid[l, 0]:range_mid[l, 1], :]
      )
      x = jax.nn.relu(x)

      if l > 0:  # from previous (coarser) level: K21, K32, K43, ...
        kernel_l_ji = MLP(
          [self.cfg.ker_in, self.ker_widths[l], width**2], self.cfg.mlp_cfg
        )
        x = x + NNConv(
          kernel_l_ji, self.cfg.nnconv_cfg, name=f"K{l+2}{l+1}"
        )(
          x, edge_index_up[:, range_up[l - 1, 0]:range_up[l - 1, 1]],
          edge_attr_up[range_up[l - 1, 0]:range_up[l - 1, 1], :]
        )

        x = jax.nn.relu(x)

    x = MLP([self.cfg.ker_width], self.cfg.mlp_cfg)(x[:self.finest_mesh_size])
    x = jax.nn.relu(x)
    x = MLP([1], self.cfg.mlp_cfg)(x)
    return x

  @hk.experimental.name_like("__call__")
  def agg(self, data):
    """placeholder"""
    return 1

  def init_for_multitransform(self):
    return self.__call__, (self.__call__, self.agg)
