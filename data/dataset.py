import string

import tensorflow as tf
import json
from absl import app, flags
import re

FLAGS = flags.FLAGS

flags.DEFINE_string("task", default=None, required=True, help="Task to execute")
flags.DEFINE_integer("vocab_size", default=20000, help="Only consider the top N words")
flags.DEFINE_integer("maxlen", default=50, help="Max sequence size")
flags.DEFINE_integer("sbs", default=128, help="Shuffle buffer size")


def clean_text(sentence: str) -> str:
    sentence = sentence.lower()
    sentence = re.sub(r"n\'t|n’t", " not", sentence)
    sentence = re.sub(r"\'re|’re", " are", sentence)
    sentence = re.sub(r"\'s|’s", " is", sentence)
    sentence = re.sub(r"\'d|’d", " would", sentence)
    sentence = re.sub(r"\'ll|’ll", " will", sentence)
    sentence = re.sub(r"\'t|’t", " not", sentence)
    sentence = re.sub(r"\'ve|’ve", " have", sentence)
    sentence = re.sub(r"\'m|’m", " am", sentence)
    return sentence


def build_vocab():
    tv = tf.keras.layers.TextVectorization(
        max_tokens=FLAGS.vocab_size - 1,
        output_mode="int",
        output_sequence_length=FLAGS.maxlen + 1,
    )

    ds = tf.data.Dataset.load("text")
    tv.adapt(ds)

    vocab = tv.get_vocabulary()
    tf.io.write_file("vocab.txt", "\n".join(vocab))


def build_dataset():
    with open("raw.json") as f:
        raw_data = json.load(f)
    raw_data = raw_data["messages"]
    text_data = [i["text"] for i in raw_data if isinstance(i["text"], str)]
    text_data = list(map(clean_text, filter(lambda i: len(i) > 1, text_data)))
    ds = tf.data.Dataset.from_tensor_slices(
        tf.constant(text_data, dtype=tf.string), name="chai_messages"
    ).shuffle(FLAGS.sbs)
    ds.save("./text")


task_map = {"vocab": build_vocab, "dataset": build_dataset}


def main(_):
    task = task_map[FLAGS.task]
    with tf.device("CPU"):
        task()


if __name__ == "__main__":
    app.run(main)
