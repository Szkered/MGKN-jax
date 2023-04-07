from absl import app, flags
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

from MGKN_jax.train import train

FLAGS = flags.FLAGS
flags.DEFINE_enum("run", "train", ["train", "test"], "which routine to run")
flags.DEFINE_bool("use_tb", False, "which to log to tensorboard")
config_flags.DEFINE_config_file(name="config", default="MGKN_jax/config.py")


def main(_):
  cfg = FLAGS.config  # type: ConfigDict

  if FLAGS.run == "train":
    train(cfg, FLAGS.use_tb)


if __name__ == "__main__":
  app.run(main)
