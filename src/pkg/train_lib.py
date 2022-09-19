from __future__ import annotations

import numpy as np
from jax.random import PRNGKey
import jax
from flax.training.train_state import TrainState
from ml_collections.config_dict import ConfigDict
import jax.numpy as jnp
import optax
from typing import Tuple, Dict
from optax import GradientTransformation
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from tqdm import tqdm

from .modeling import build_model
from .types import InputTuple, TokenBatch, TokenizedDataset


optimizers = {"adam": optax.adam, "yogi": optax.yogi}


def create_optimizer(config: ConfigDict) -> GradientTransformation:
    if config.optimizer.learnable:
        raise NotImplementedError()
    return optimizers[config.optimizer.type](config.optimizer.learning_rate)


def create_train_state(rng: PRNGKey, config: ConfigDict) -> TrainState:
    model = build_model(config.fnet)

    dummy_value = jnp.ones([1, config.fnet.max_len], dtype=int)
    dummy_input = InputTuple(
        encoder_inputs=TokenBatch(dummy_value.copy(), dummy_value.copy()),
        decoder_inputs=TokenBatch(dummy_value.copy(), dummy_value.copy()),
        outputs=None,  # noqa
    )
    params = model.init(rng, dummy_input)["params"]
    tx = create_optimizer(config)
    apply_func = jax.jit(model.apply)
    return TrainState.create(apply_fn=apply_func, params=params, tx=tx)


@jax.jit
def compute_metrics(
    logits: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray
) -> jnp.float32:
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = loss * mask

    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    loss = jnp.mean(loss)
    return loss, accuracy


@jax.jit
def train_step(
    state: TrainState, batch: InputTuple
) -> Tuple[TrainState, jnp.float32, jnp.float32]:
    @jax.jit
    def loss_fn(params: FrozenDict) -> jnp.float32:
        predictions, logits = state.apply_fn({"params": params}, batch)
        expected = batch.outputs
        loss_val, accuracy_val = compute_metrics(
            logits, expected.token_ids, expected.attention_mask
        )
        return loss_val, accuracy_val

    (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = jax.jit(state.apply_gradients)(grads=grads)
    return new_state, loss, accuracy


@jax.jit
def eval_step(state: TrainState, batch: InputTuple) -> Tuple[jnp.float32, jnp.float32]:
    predictions, logits = state.apply_fn({"params": state.params}, batch)
    expected = batch.outputs
    return compute_metrics(logits, expected.token_ids, expected.attention_mask)


def train_model(dataset: TokenizedDataset, config: ConfigDict) -> Dict:
    """
    :param dataset:
    :param config:
    :return: Model's params, recorded metrics
    """
    prng = jax.random.PRNGKey(config.prng.key)
    train_state = create_train_state(prng, config)

    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(1, config.training.epochs + 1):

        for batch in (pbar := tqdm(dataset.training)):
            train_state, loss, accuracy = train_step(train_state, batch)
            train_losses.append(loss)
            train_accuracies.append(accuracy)

            pbar.set_description(
                f"Epoch={epoch},"
                f"Loss={np.mean(train_losses):.3f},"
                f"Accuracy={np.mean(train_accuracies):.3f},"
                f"Validation Loss={np.mean(validation_losses):.3f},"
                f"Validation Accuracy={np.mean(validation_accuracies):.3f}"
            )

        for batch in dataset.validation:
            loss, acc = eval_step(train_state, batch)
            validation_losses.append(loss)
            validation_accuracies.append(acc)

            pbar.set_description(
                f"Epoch={epoch},"
                f"Loss={np.mean(train_losses):.3f},"
                f"Accuracy={np.mean(train_accuracies):.3f},"
                f"Validation Loss={np.mean(validation_losses):.3f},"
                f"Validation Accuracy={np.mean(validation_accuracies):.3f}"
            )

    state_dict = to_state_dict(train_state.params)
    return state_dict
