import haiku as hk
import jax
import jax.numpy as jnp
import optax
from absl import logging
from ml_collections import ConfigDict

from MGKN_jax.config import TrainConfig
from MGKN_jax.dataset import ParametricEllipticalPDE
from MGKN_jax.model import MGKN
import tensorboardX as tb


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
  # init data
  dataset = ParametricEllipticalPDE(cfg.train_cfg.data_cfg)
  data_init = dataset.get_init_data()

  # NOTE: currently we can't batch graphs, need to modify jraph or write custom batch
  # data_init2 = dataset.get_init_data()
  # batch = jraph.batch([data_init, data_init2])
  # breakpoint()

  # init model
  model = hk.transform(lambda x: MGKN(cfg.mgkn_cfg)(x))
  rng = jax.random.PRNGKey(cfg.train_cfg.rng_seed)
  params = model.init(rng, data_init)

  # check input shapes
  # logging.info(data.nodes['inputs'].shape)
  # logging.info(data_init.n_edge)
  # breakpoint()

  # # viz
  # z = jax.xla_computation(model.apply)(params, None, data_init)
  # with open("t.dot", "w") as f:
  #   f.write(z.as_hlo_dot_graph())

  optimizer = get_optimizer(cfg.train_cfg)
  opt_state = optimizer.init(params)

  def add_l2_regularization(grads, params, l2_reg_weight):
    l2_grads = jax.tree_util.tree_map(lambda p: l2_reg_weight * p, params)
    return jax.tree_util.tree_map(lambda g, l2g: g + l2g, grads, l2_grads)

  def loss_fn(params, data):
    y_pred = model.apply(params, None, data)  # (n_grid_pts, 1)
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
    return loss, (mse, y_pred)

  @jax.jit
  def update(params, opt_state, data):
    loss = lambda params: loss_fn(params, data)
    (loss_val, mse), grad = jax.value_and_grad(loss, has_aux=True)(params)
    grad = add_l2_regularization(grad, params, cfg.train_cfg.l2_reg)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val, mse

  writer = tb.SummaryWriter("logs")

  data_gen = dataset.make_data_gen(cfg.train_cfg)
  n_data_per_epoch = (
    dataset.cfg.n_train * dataset.cfg.n_samples_per_train_data
  )
  # go through one random multilevel graph at a time
  for epoch in range(cfg.train_cfg.epochs):
    train_mse = 0.0
    train_l2 = 0.0
    for _ in range(n_data_per_epoch):
      data = next(data_gen)
      params, opt_state, train_l2_i, aux = update(params, opt_state, data)
      train_mse_i, y_pred = aux
      train_mse += train_mse_i
      train_l2 += train_l2_i

    train_mse /= n_data_per_epoch
    train_l2 /= n_data_per_epoch
    logging.info(f"{epoch}| mse: {train_mse:.6f}, l2: {train_l2:.6f}")
    writer.add_scalar("l2", train_l2, epoch)
    writer.add_scalar("mse", train_mse, epoch)
