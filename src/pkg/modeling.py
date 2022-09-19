from __future__ import annotations

import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
from typing import Optional, Tuple
from jax.numpy.fft import fft2

import numpy as np
from ml_collections.config_dict import ConfigDict

from .types import InputTuple


def jax_range(upper_bound: int) -> jnp.ndarray:
    return jnp.linspace(0, upper_bound - 1, upper_bound).astype(int)


class FFN(nn.Module):
    dim1: int
    dim2: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.dim1)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim2)(x)
        return x


class FourierAttention(nn.Module):
    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        fft = lax.real(fft2(inputs.astype(jnp.complex64)))
        if attention_mask is not None:
            fft = jax.vmap(lambda a, b: a * b, in_axes=[-1, None], out_axes=-1)(
                fft, attention_mask
            )
        return fft


class FNetEncoder(nn.Module):
    dense_dim: int
    embed_dim: int

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        fft = FourierAttention()(inputs, attention_mask)
        proj_input = nn.LayerNorm()(inputs + fft)
        proj_output = FFN(dim1=self.dense_dim, dim2=self.embed_dim)(proj_input)
        return nn.LayerNorm()(proj_input + proj_output)


class FNetDecoder(nn.Module):
    embed_dim: int
    latent_dim: int

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        encoder_outputs,
        self_attention_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        self_fft = FourierAttention()(inputs, self_attention_mask)

        out_1 = nn.LayerNorm()(inputs + self_fft)

        fft_1 = FourierAttention()(out_1)
        fft_2 = FourierAttention()(encoder_outputs)

        cross_fft = fft_1 * fft_2

        out_2 = nn.LayerNorm()(out_1 + cross_fft)

        proj_output = FFN(dim1=self.latent_dim, dim2=self.embed_dim)(out_2)
        return nn.LayerNorm()(out_2 + proj_output)


class PositionalEmbedding(nn.Module):
    sequence_length: int
    vocab_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray | np.ndarray) -> jnp.ndarray:
        length = inputs.shape[-1]
        positions = jax_range(length)
        embedded_tokens = nn.Embed(self.vocab_size, self.embed_dim)(inputs)
        embedded_positions = nn.Embed(self.sequence_length, self.embed_dim)(positions)
        return embedded_tokens + embedded_positions


class FNetModel(nn.Module):
    max_len: int
    vocab_size: int
    embed_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, inputs: InputTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Inputs is a dict, e.g.,
            {
            "encoder_inputs": {
                "token_ids": np.ndarray, with shape[64,40], dtype=int,
                "attention_mask": np.ndarray, with shape[64,40], dtype=int,
                }
            "decoder_inputs": {
                "token_ids": np.ndarray, with shape[64,40], dtype=int,
                "attention_mask": np.ndarray, with shape[64,40], dtype=int,
                }
            }
        When 64 is a batch size, and 40 is maximal sequence length
        """
        encoder_inputs = inputs.encoder_inputs
        decoder_inputs = inputs.decoder_inputs

        x_encoder = PositionalEmbedding(
            sequence_length=self.max_len,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )(encoder_inputs.token_ids)
        encoder_outputs = FNetEncoder(
            embed_dim=self.embed_dim, dense_dim=self.latent_dim
        )(x_encoder, encoder_inputs.attention_mask)

        x_decoder = PositionalEmbedding(
            sequence_length=self.max_len,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )(decoder_inputs.token_ids)
        decoder_output = FNetDecoder(
            embed_dim=self.embed_dim,
            latent_dim=self.latent_dim,
        )(x_decoder, encoder_outputs, decoder_inputs.attention_mask)

        decoder_output = nn.Dropout(0.1, deterministic=True)(decoder_output)
        decoder_outputs = nn.Dense(self.vocab_size)(decoder_output)
        predictions = jnp.argmax(decoder_outputs, axis=-1)
        return predictions, decoder_outputs


def build_model(config: ConfigDict) -> nn.Module:
    model = nn.jit(FNetModel)(
        max_len=config.max_len,
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        latent_dim=config.latent_dim,
    )
    return model
