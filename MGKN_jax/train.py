import haiku as hk
import jax
import jax.numpy as jnp
import optax
from ml_collections import ConfigDict
from MGKN_jax.config import TrainConfig

from MGKN_jax.dataset import ParametricEllipticalPDE
from MGKN_jax.model import MGKN


def get_optimizer(cfg: TrainConfig):
  optimizer_kwargs = dict(learning_rate=cfg.learning_rate)
  if cfg.lr_decay:
    total_steps = cfg.data_cfg.n_train / cfg.batch_size * cfg.epochs
    scheduler = optax.piecewise_constant_schedule(
      init_value=cfg.learning_rate,
      boundaries_and_scales={
        int(total_steps * 0.5): 0.5,
        int(total_steps * 0.75): 0.5
      }
    )
    optimizer_kwargs["learning_rate"] = scheduler

  optimizer = cfg.optimizer(**optimizer_kwargs)
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
  model = hk.transform(MGKN(cfg.mgkn_cfg))
  rng = jax.random.PRNGKey(cfg.train_cfg.rng_seed)
  params = model.init(rng, data_init)

  optimizer = get_optimizer(cfg.train_cfg)
  opt_state = optimizer.init(params)

  def loss_fn(params, data):
    y_pred = model.apply(params, None, data)
    y = data.nodes["outputs"]
    y_pred_unnorm = dataset.train_out.unnormalize(
      y_pred, sample_idx=data.globals
    )
    y_unnorm = dataset.train_out.unnormalize(y, sample_idx=data.globals)
    diff_norm = jnp.linalg.norm(y_pred_unnorm - y_unnorm, axis=-1)
    y_norm = jnp.linalg.norm(y_unnorm, axis=-1)
    loss = jnp.sum(diff_norm / y_norm)
    return loss

  @jax.jit
  def update(params, opt_state, data):
    loss = lambda params: loss_fn(params, data)
    loss_val, grad = jax.value_and_grad(loss)(params)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

  data_gen = dataset.make_data_gen(cfg.train_cfg)
  # go through one random multilevel graph at a time
  for data in data_gen:
    params, opt_state, loss_val = update(params, opt_state, data)
    logging.info(f"loss: {loss_val}")
