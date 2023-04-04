import haiku as hk
import jax
from ml_collections import ConfigDict

from MGKN_jax.dataset import ParametricEllipticalPDE
from MGKN_jax.model import MGKN


def train(cfg: ConfigDict):
  # init data
  dataset = ParametricEllipticalPDE(cfg.train_cfg.data_cfg)
  data_gen = dataset.make_data_gen(cfg.train_cfg)

  data_init = next(data_gen)

  # NOTE: currently we can't batch graphs, need to modify jraph or write custom batch
  # data_init2 = next(data_gen)
  # batch = jraph.batch([data_init, data_init2])
  # breakpoint()

  # init model
  model = hk.multi_transform(
    lambda: MGKN(cfg.mgkn_cfg).init_for_multitransform()
  )
  rng = jax.random.PRNGKey(cfg.train_cfg.rng_seed)
  params = model.init(rng, data_init)

  breakpoint()
