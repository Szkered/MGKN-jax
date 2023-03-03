import haiku as hk
import jax
import jax.numpy as jnp
from absl import app, flags
from ml_collections.config_flags import config_flags

from MGKN_jax.dataset import ParametricEllipticalPDE
from MGKN_jax.model import MGKN

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(name="config", default="MGKN_jax/config.py")


def main(_):
  cfg = FLAGS.config  # type: ConfigDict

  # init data
  dataset = ParametricEllipticalPDE(cfg.train_cfg.data_cfg)

  # init model
  model = hk.multi_transform(lambda: MGKN(cfg).init_for_multitransform())
  rng = jax.random.PRNGKey(cfg.train_cfg.rng_seed)
  fake_x = jnp.ones_like()  #  TODO
  params = model.init(rng, fake_x)


if __name__ == "__main__":
  app.run(main)