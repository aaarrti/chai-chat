from __future__ import annotations
from functools import partial
import json
from typing import List, Tuple, Callable

from tokenizers import Tokenizer, normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

from ml_collections.config_flags import DEFINE_config_file
from ml_collections.config_dict import ConfigDict
from absl import app

from safetensors.flax import save_file
import numpy as np


from src.pkg.types import TokenizedDataset, InputTuple, TokenBatch

CONFIG = DEFINE_config_file("config", "src/config.py")


def flatten(llist: List[List]) -> List:
    return [item for sublist in llist for item in sublist]


def load_dataset() -> List[str]:
    with open("data/raw_dataset.json") as file:
        raw_data = json.load(file)["messages"]

    text_entities = [i["text_entities"] for i in raw_data]
    text_entities = list(filter(lambda i: len(i) > 0, text_entities))
    text_entities = flatten(text_entities)
    text_entities = list(filter(lambda i: i["type"] == "plain", text_entities))
    messages = [i["text"] for i in text_entities]
    return messages


def question_answer_split(text: List[str]) -> Tuple[List[str], List[str]]:
    questions = []
    answers = []
    for i in range(0, len(text) - 1, 2):
        questions.append(text[i])
        answers.append(text[i + 1])
    return questions, answers


def train_tokenizer(text: List[str], config: ConfigDict) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))  # noqa
    # fmt: off
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])  # noqa
    # fmt: on
    tokenizer.pre_tokenizer = Whitespace()  # noqa

    trainer = WordPieceTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"],
        vocab_size=config.fnet.vocab_size,
    )  # noqa
    tokenizer.enable_truncation(max_length=config.fnet.max_len)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=config.fnet.max_len)
    tokenizer.train_from_iterator(text, trainer)
    tokenizer.save("data/tokenizer.json")
    return tokenizer


def training_validation_split(text: List[str]) -> Tuple[List[str], List[str]]:
    training = text[: int(0.9 * len(text))]
    validation = text[len(training) :]
    return training, validation


def tokenize(text: List[str], tokenizer: Tokenizer) -> TokenBatch:
    tokens = tokenizer.encode_batch(text)
    ids = [i.ids for i in tokens]
    mask = [i.attention_mask for i in tokens]
    return TokenBatch(np.asarray(ids), np.asarray(mask))


def build_input_tuple(
    questions: List[str],
    answers: List[str],
    tokenize_func: Callable[[List[str]], TokenBatch],
) -> InputTuple:
    inputs = tokenize_func(questions)
    outputs = tokenize_func(answers)

    # One extra padding token to the right to match the output shape
    output_tokens = np.pad(outputs.token_ids, [[0, 1]], constant_values=0)
    output_mask = np.pad(outputs.attention_mask, [[0, 1]], constant_values=0)

    return InputTuple(
        encoder_inputs=inputs,
        decoder_inputs=TokenBatch(output_tokens[:-1, :-1], output_mask[:-1, :-1]),
        outputs=TokenBatch(output_tokens[1:, 1:], output_mask[1:, 1:]),
    )


def make_dataset(
    text: List[str], tokenize_func: Callable[[List[str]], TokenBatch], batch_size: int
) -> List[InputTuple]:
    questions, answers = question_answer_split(text)

    questions = questions[: len(questions) // batch_size * batch_size]
    answers = questions[: len(answers) // batch_size * batch_size]

    questions = np.asarray(questions).reshape((-1, batch_size))
    answers = np.asarray(answers).reshape((-1, batch_size))

    data = []
    for q, a in zip(questions, answers):
        it = build_input_tuple(q, a, tokenize_func)
        data.append(it)

    return data


def main(*args):
    config = CONFIG.value
    text_ds = load_dataset()
    tokenizer = train_tokenizer(text_ds, config)
    text_ds = [f"[CLS] {i} [SEP]" for i in text_ds]
    training_text, validation_text = training_validation_split(text_ds)

    batch_size = config.training.batch_size

    make_dataset_func = partial(
        make_dataset,
        batch_size=batch_size,
        tokenize_func=partial(tokenize, tokenizer=tokenizer),
    )

    dataset = TokenizedDataset(
        training=make_dataset_func(training_text),
        validation=make_dataset_func(validation_text),
    )

    training_tensors, validation_tensors = dataset.to_ndarray()
    tensors = {"training": training_tensors, "validation": validation_tensors}
    save_file(tensors, "data/dataset.safetensors")  # noqa


if __name__ == "__main__":
    app.run(main)
