from absl import app
from ml_collections.config_flags import DEFINE_config_file
import jax
from safetensors.flax import load_file, save_file
from flax.traverse_util import flatten_dict

from pkg.train_lib import train_model
from pkg.types import TokenizedDataset

jax.config.update("jax_log_compiles", True)
jax.config.update("jax_debug_nans", True)
CONFIG = DEFINE_config_file("config", "src/config.py")


def main(*args):
    config = CONFIG.value
    print("-" * 50)
    print(f"{jax.devices() = }")
    print(f"{config = }")
    print("-" * 50)

    # fmt: off
    tensors = load_file("data/dataset.safetensors")
    dataset = TokenizedDataset.from_ndarray(tensors["training"], tensors["validation"])  # noqa
    params = train_model(dataset, config)
    params_flat = flatten_dict(params, sep="/")
    save_file(params_flat, "data/model.safetensors") # noqa
    # fmt: on


if __name__ == "__main__":
    app.run(main)
