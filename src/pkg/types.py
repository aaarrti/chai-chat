from __future__ import annotations

from typing import NamedTuple, List, Tuple

import numpy as np
import jax.numpy as jnp


class TokenBatch(NamedTuple):
    """
    token_ids: shape=[batch_size, max_lem], dtype=int
    attention_mask: shape=[batch_size, max_lem], dtype=int
    """

    token_ids: np.ndarray | jnp.ndarray
    attention_mask: np.ndarray | jnp.ndarray

    @staticmethod
    def from_ndarray(arr: np.ndarray) -> TokenBatch:
        return TokenBatch(token_ids=arr[0], attention_mask=arr[1])

    def to_ndarray(self) -> np.ndarray:
        return np.stack(
            [
                self.token_ids,
                self.attention_mask,
            ]
        )


class InputTuple(NamedTuple):
    encoder_inputs: TokenBatch
    decoder_inputs: TokenBatch
    outputs: TokenBatch

    @staticmethod
    def from_ndarray(arr: np.ndarray) -> InputTuple:
        return InputTuple(
            encoder_inputs=TokenBatch.from_ndarray(arr[0]),
            decoder_inputs=TokenBatch.from_ndarray(arr[1]),
            outputs=TokenBatch.from_ndarray(arr[2]),
        )

    def to_ndarray(self) -> np.ndarray:
        return np.stack(
            [
                self.encoder_inputs.to_ndarray(),
                self.decoder_inputs.to_ndarray(),
                self.outputs.to_ndarray(),
            ]
        )


class TokenizedDataset(NamedTuple):
    training: List[InputTuple]
    validation: List[InputTuple]

    def to_ndarray(self) -> Tuple[np.ndarray, np.ndarray]:
        training = np.stack([i.to_ndarray() for i in self.training])
        validation = np.stack([i.to_ndarray() for i in self.validation])
        return training, validation

    @staticmethod
    def from_ndarray(arr1: np.ndarray, arr2: np.ndarray) -> TokenizedDataset:
        training = [InputTuple.from_ndarray(i) for i in arr1]
        validation = [InputTuple.from_ndarray(i) for i in arr2]
        return TokenizedDataset(training, validation)
