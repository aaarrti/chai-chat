import jax
import pytest
import jax.numpy as jnp
from typing import Callable

from src.pkg.modeling import FNetModel
from src.pkg.types import InputTuple, TokenBatch

vocab_size = 8192
max_len = 40
embed_dim = 256
latent_dim = 512


@pytest.fixture(scope="session", autouse=True)
def model() -> Callable:
    dummy_value = jnp.ones([1, max_len], dtype=int)
    dummy_input = InputTuple(
        encoder_inputs=TokenBatch(dummy_value.copy(), dummy_value.copy()),
        decoder_inputs=TokenBatch(dummy_value.copy(), dummy_value.copy()),
        outputs=None,  # noqa
    )
    with jax.disable_jit():
        model = FNetModel(
            max_len=max_len,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
        )
        params = model.init(jax.random.PRNGKey(0), dummy_input)["params"]
        return lambda x: (model.apply({"params": params}, x))


@pytest.fixture(scope="session", autouse=True)
def x_batch():
    dummy_value = jnp.ones([32, max_len], dtype=int)
    dummy_input = InputTuple(
        encoder_inputs=TokenBatch(dummy_value.copy(), dummy_value.copy()),
        decoder_inputs=TokenBatch(dummy_value.copy(), dummy_value.copy()),
        outputs=None,  # noqa
    )
    return dummy_input


def test_model_no_jit(model: Callable, x_batch: InputTuple, capsys):
    with jax.disable_jit():
        predictions, logits = model(x_batch)
        assert predictions.shape == (32, 40)
        assert logits.shape == (32, 40, 8192)


def test_model_jitted(model: Callable, x_batch: InputTuple, capsys):
    predictions, logits = model(x_batch)
    assert predictions.shape == (32, 40)
    assert logits.shape == (32, 40, 8192)
