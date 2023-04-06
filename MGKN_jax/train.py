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
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val, mse

  writer = tb.SummaryWriter("logs")

  data_gen = dataset.make_data_gen(cfg.train_cfg)
  n_train = dataset.cfg.n_train
  # go through one random multilevel graph at a time
  train_mse = 0.0
  train_l2 = 0.0
  for step, data in enumerate(data_gen):
    epoch, train_idx = divmod(step, n_train)
    params, opt_state, train_l2_i, aux = update(params, opt_state, data)
    train_mse_i, y_pred = aux
    train_mse += train_mse_i
    train_l2 += train_l2_i
    if train_idx == n_train - 1:  # end of epoch
      logging.info(
        f"{epoch}:{step}| mse: {train_mse/n_train:.4f}, l2: {train_l2/n_train:.4f}"
      )
      writer.add_scalar("l2", train_l2 / n_train, step)
      writer.add_scalar("mse", train_mse / n_train, step)
      train_mse = 0.0
      train_l2 = 0.0
