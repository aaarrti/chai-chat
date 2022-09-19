from __future__ import annotations
from tokenizers import Tokenizer
import numpy as np
from typing import Tuple, Callable
from .types import InputTuple, TokenBatch


def sample_sentence(
    *,
    prompt: str,
    tokenizer: Tokenizer,
    predict_func: Callable[[InputTuple], Tuple[np.ndarray, np.ndarray]],
    max_len: int,
):
    def tokenize(x: str) -> Tuple[np.ndarray, np.ndarray]:
        ids = tokenizer.encode(x)
        return np.expand_dims(ids.ids, 0), np.expand_dims(ids.attention_mask, 0)

    vocab = tokenizer.get_vocab()

    # Mapping the input sentence to tokens and adding start and end tokens
    input_ids, input_mask = tokenize("[CLS] " + prompt + " [SEP]")
    # Initializing the initial sentence consisting of only the start token.
    target_ids, target_mask = tokenize("[CLS]")

    sampled_sentence = []

    for i in range(max_len):
        # Get the predictions
        input_tuple = InputTuple(
            encoder_inputs=TokenBatch(input_ids, input_mask),
            decoder_inputs=TokenBatch(target_ids, target_mask),
            outputs=None,  # noqa
        )
        predicted_tokens, _ = predict_func(input_tuple)
        # Calculating the token with maximum probability and getting the corresponding word
        sampled_token_index = predicted_tokens[0, i]

        # If sampled token is the end token then stop generating and return the sentence
        if sampled_token_index == vocab["[SEP]"]:
            break
        sampled_sentence.append(sampled_token_index)
        target_ids[0, i] = sampled_token_index
        target_mask[0, i] = 1

    return tokenizer.decode(sampled_sentence)
