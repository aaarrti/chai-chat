import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # noqa
import numpy as np
import os
import re
import string
import random
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("task", default=None, required=True, help="Task to execute")
flags.DEFINE_integer("vocab_size", default=20000, help="Only consider the top N words")
flags.DEFINE_integer("maxlen", default=50, help="Max sequence size")
flags.DEFINE_integer("sbs", default=128, help="Shuffle buffer size")
flags.DEFINE_integer("embed_dim", default=256, help="Embedding size for each token")
flags.DEFINE_integer("num_heads", default=2, help="Number of attention heads")
flags.DEFINE_integer("feed_forward_dim", default=256, help="Hidden layer size in feed forward network inside transformer")
flags.DEFINE_integer("batch_size", default=128, help="Dataset batch size")


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def create_model() -> tf.keras.Model:
    inputs = layers.Input(shape=(FLAGS.maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(FLAGS.maxlen, FLAGS.vocab_size, FLAGS.embed_dim)
    x = embedding_layer(inputs) # noqa
    transformer_block = TransformerBlock(FLAGS.embed_dim, FLAGS.num_heads, FLAGS.feed_forward_dim)
    x = transformer_block(x) # noqa
    outputs = layers.Dense(FLAGS.vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile("adam", loss=[loss_fn, None])  # No loss and optimization based on word embeddings from transformer block
    return model


class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(self,
                 max_tokens,
                 start_tokens,
                 index_to_word,
                 top_k=10,
                 print_every=1
                 ):
        super().__init__()
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = FLAGS.maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:FLAGS.maxlen]
                sample_index = FLAGS.maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")


def load_ds() -> tf.data.Dataset:
    tv = tf.keras.layers.TextVectorization(
        max_tokens=FLAGS.vocab_size - 1,
        output_mode="int",
        output_sequence_length=FLAGS.maxlen + 1,
        vocabulary="../data/vocab.txt"
    )

    def prepare_lm_inputs_labels(text):
        """
        Shift word sequences by 1 position so that the target for position (i) is
        word at position (i+1). The model will use all words up till position (i)
        to predict the next word.
        """
        text = tf.expand_dims(text, -1)
        tokenized_sentences = tv(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y


    text_ds = tf.data.Dataset.load("../data/text")
    text_ds = text_ds.batch(FLAGS.batch_size)
    text_ds = text_ds.map(prepare_lm_inputs_labels)
    text_ds = text_ds.prefetch(tf.data.AUTOTUNE)
    return text_ds


def plot_model():
    model = create_model()
    tf.keras.utils.plot_model(model)


def train_model():
    model = create_model()
    text_ds = load_ds()
    text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)

    model.fit(text_ds, verbose=2, epochs=25, callbacks=[text_gen_callback])


task_map = {
    "plot": plot_model,
    "train": train_model
}


def main(_):
    task = task_map[FLAGS.task]
    task()


if __name__ == '__main__':
    app.run(main)