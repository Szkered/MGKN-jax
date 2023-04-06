import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorboardX as tb
from absl import app, flags, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

from MGKN_jax.config import TrainConfig
from MGKN_jax.dataset import ParametricEllipticalPDE
from MGKN_jax.model import MGKN
from MGKN_jax.train import train

FLAGS = flags.FLAGS
flags.DEFINE_enum("run", "train", ["train", "test"], "which routine to run")
config_flags.DEFINE_config_file(name="config", default="MGKN_jax/config.py")


def main(_):
  cfg = FLAGS.config  # type: ConfigDict

  # dataset = ParametricEllipticalPDE(cfg.train_cfg.data_cfg)
  # rng = jax.random.PRNGKey(137)
  # key, rng = jax.random.split(rng)
  # with jax.profiler.trace("./jax-trace", create_perfetto_link=True):
  #   for i in range(5):
  #     with jax.profiler.TraceAnnotation(f"step {i}"):
  #       print(i)
  #       key, rng = jax.random.split(rng)
  #       sample = dataset.multi_mesh.sample(key, i)

  # return

  if FLAGS.run == "train":
    train(cfg)


if __name__ == "__main__":
  app.run(main)
