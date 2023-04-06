from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
import tensorboardX as tb
from absl import logging
from ml_collections import ConfigDict
from tqdm import tqdm

from MGKN_jax.config import TrainConfig
from MGKN_jax.dataset import ParametricEllipticalPDE
from MGKN_jax.model import MGKN


class TrainingState(NamedTuple):
  """Container for the training state."""
  params: hk.Params
  opt_state: optax.OptState
  rng: jax.Array
  step: jax.Array


def get_optimizer(cfg: TrainConfig):
  optimizer_kwargs = dict(learning_rate=cfg.learning_rate)
  if cfg.lr_decay:
    scheduler = optax.exponential_decay(
      cfg.learning_rate,
      cfg.scheduler_step,
      cfg.scheduler_gamma,
      staircase=True
    )
    optimizer_kwargs["learning_rate"] = scheduler

  if cfg.optimizer == 'sgd':
    optimizer = optax.sgd(**optimizer_kwargs)
  elif cfg.optimizer == 'adam':
    optimizer = optax.adam(**optimizer_kwargs)

  return optimizer


def train(cfg: ConfigDict):

  def forward(data: jraph.GraphsTuple):
    model = MGKN(cfg.mgkn_cfg)
    return model(data)

  optimizer = get_optimizer(cfg.train_cfg)

  @hk.transform
  def loss_fn(data):
    y_pred = forward(data)  # (n_grid_pts, 1)
    y_pred = jnp.squeeze(y_pred)
    y = data.nodes["outputs"]  # (n_grid_pts)
    mean, std = data.globals  # (n_grid_pts)
    unnormalize = lambda y: (y * (std + 1e-5)) + mean
    y_pred_unnorm = unnormalize(y_pred)
    y_unnorm = unnormalize(y)
    diff_norm = jnp.linalg.norm(y_pred_unnorm - y_unnorm, axis=-1)
    y_norm = jnp.linalg.norm(y_unnorm, axis=-1)
    loss = jnp.sum(diff_norm / y_norm)
    mse = jnp.mean(jnp.square(y_pred - y))
    return loss, mse

  def add_l2_regularization(grads, params, l2_reg_weight):
    l2_grads = jax.tree_util.tree_map(lambda p: l2_reg_weight * p, params)
    return jax.tree_util.tree_map(lambda g, l2g: g + l2g, grads, l2_grads)

  @jax.jit
  def update(state: TrainingState, data: jraph.GraphsTuple):
    rng, new_rng = jax.random.split(state.rng)
    loss_and_grad_fn = jax.value_and_grad(loss_fn.apply, has_aux=True)
    (loss, mse), grad = loss_and_grad_fn(state.params, rng, data)
    grad = add_l2_regularization(grad, state.params, cfg.train_cfg.l2_reg)
    updates, new_opt_state = optimizer.update(grad, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    new_state = TrainingState(
      params=new_params,
      opt_state=new_opt_state,
      rng=new_rng,
      step=state.step + 1,
    )
    metrics = {'step': state.step, 'loss': loss, 'mse': mse}
    return new_state, metrics

  @jax.jit
  def init(rng, data) -> TrainingState:
    rng, init_rng = jax.random.split(rng)
    initial_params = loss_fn.init(init_rng, data)
    initial_opt_state = optimizer.init(initial_params)
    return TrainingState(
      params=initial_params,
      opt_state=initial_opt_state,
      rng=rng,
      step=jnp.array(0),
    )

  # init data
  dataset = ParametricEllipticalPDE(cfg.train_cfg.data_cfg)
  data_gen = dataset.make_data_gen(cfg.train_cfg)
  data = next(data_gen)

  rng = jax.random.PRNGKey(cfg.train_cfg.rng_seed)
  state = init(rng, data)

  writer = tb.SummaryWriter("logs")

  n_data_per_epoch = (
    dataset.cfg.n_train * dataset.cfg.n_samples_per_train_data
  )
  # go through one random multilevel graph at a time
  for epoch in range(cfg.train_cfg.epochs):
    train_mse = 0.0
    train_l2 = 0.0
    for _ in tqdm(range(n_data_per_epoch)):
      data = next(data_gen)
      state, metrics = update(state, data)
      train_l2 += metrics['loss']
      train_mse += metrics['mse']

    train_mse /= n_data_per_epoch
    train_l2 /= n_data_per_epoch
    logging.info(f"{epoch}| mse: {train_mse:.6f}, l2: {train_l2:.6f}")
    writer.add_scalar("l2", train_l2, epoch)
    writer.add_scalar("mse", train_mse, epoch)
