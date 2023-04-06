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
      1.0, cfg.init_weights_mode, cfg.init_weights_distribution
    )
    self.init_b = hk.initializers.VarianceScaling(
      1.0, cfg.init_bias_mode, cfg.init_bias_distribution
    )

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
    n_segs = x.shape[0]

    # calculate message
    weight = self.nn(edge_attr)  # (num_edges, out_chan)
    weight = weight.reshape(-1, self.cfg.width, self.cfg.width)
    x_j = x[senders][:, None]  # (num_edges, 1, in_chan)
    msgs = jnp.matmul(x_j, weight).squeeze(1)  # (num_edges, out_chan)

    # aggregate neighbours
    if self.cfg.aggr == "mean":
      msg = jraph.segment_mean(msgs, receivers, n_segs)
    elif self.cfg.aggr == "sum":
      msg = jraph.segment_sum(msgs, receivers, n_segs)

    # NOTE: Original impl: root_weight=False, bias=False
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
      self.cfg.finest_ker_width // (2**(l + 1)) for l in range(self.level)
    ]

  def __call__(self, data: jraph.GraphsTuple):
    """
    """
    width = self.cfg.nnconv_cfg.width

    x = MLP([width], self.cfg.mlp_cfg, name="first")(data.nodes["inputs"])

    for d in range(self.cfg.depth):
      # DOWNWARD: K12, K23, K34 ...
      for l in range(self.level - 1):
        kernel_l = MLP(
          [self.ker_widths[l], width**2],
          self.cfg.mlp_cfg,
          name=f"K{l+1}{l+2}_{d}"
        )
        x = x + NNConv(kernel_l, self.cfg.nnconv_cfg)(
          x, data.senders['inter'][l], data.receivers['inter'][l],
          data.edges['inter'][l]
        )
        x = jax.nn.relu(x)

      # UPWARD: (K55), (K44, K54), (K33, K43) ...
      for l in reversed(range(self.level)):
        # K55, K44, K33, ...
        kernel_l_ii = MLP(
          [self.ker_widths[l], self.ker_widths[l], width**2],
          self.cfg.mlp_cfg,
          name=f"K{l+1}{l+1}_{d}"
        )
        x = x + NNConv(kernel_l_ii, self.cfg.nnconv_cfg)(
          x, data.senders['inner'][l], data.receivers['inner'][l],
          data.edges['inner'][l]
        )
        x = jax.nn.relu(x)

        if l < self.level - 1:  # from previous (coarser) level: K54, K43, K32, ...
          kernel_l_ji = MLP(
            [self.ker_widths[l], width**2],
            self.cfg.mlp_cfg,
            name=f"K{l+2}{l+1}_{d}"
          )
          # NOTE: To get up edge, flip the down edge
          # (x0,y0), (x1,y1), (a0,a1) -> (x1,y1), (x0,y0), (a1,a0)
          swap_end_pt = np.array([2, 3, 0, 1, 6, 4])
          x = x + NNConv(kernel_l_ji, self.cfg.nnconv_cfg)(
            x, data.receivers['inter'][l], data.senders['inter'][l],
            data.edges['inter'][l][:, swap_end_pt]
          )
          x = jax.nn.relu(x)

    x = MLP(
      [self.cfg.finest_ker_width, self.cfg.ker_out],
      self.cfg.mlp_cfg,
      name="final"
    )(
      x[:self.finest_mesh_size]
    )
    return x
