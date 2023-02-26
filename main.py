from absl import app, flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(name="config", default="MGKN_jax/config.py")


def main(_):
  cfg = FLAGS.config  # type: ConfigDict
  breakpoint()


if __name__ == "__main__":
  app.run(main)
