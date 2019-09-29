import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras import regularizers, constraints, initializers


def get_data(test_pct, samplepct=1):
    with open("./data/small_vocab_en", "r") as f:
        en_text = [x.rstrip() for x in f.readlines()]
        en_text = np.array(en_text)
    with open("./data/small_vocab_fr", "r") as f:
        fr_text = [x.rstrip() for x in f.readlines()]
        fr_text = [
            "sos " + sent[:-1] + " eos ."
            if sent.endswith(".")
            else "sos " + sent + " eos ."
            for sent in fr_text
        ]
        fr_text = np.array(fr_text)

    indexes = np.arange(len(en_text))
    np.random.shuffle(indexes)

    sample_size = int(np.ceil(len(indexes) * samplepct))
    indexes = indexes[:sample_size]

    train_size = int(np.floor(len(indexes) * (1 - test_pct)))

    train_indexes = indexes[:train_size]
    test_indexes = indexes[train_size:]

    train_en_text = en_text[train_indexes]
    train_fr_text = fr_text[train_indexes]

    test_en_text = en_text[test_indexes]
    test_fr_text = fr_text[test_indexes]

    return train_en_text, train_fr_text, test_en_text, test_fr_text


def sent2seq(tokenizer, sents, reverse=False, pad_length=None, pad_type="post"):
    encoded_text = tokenizer.texts_to_sequences(sents)
    preprocessed_text = keras.preprocessing.sequence.pad_sequences(
        encoded_text, padding=pad_type, maxlen=pad_length
    )
    if reverse:
        preprocessed_text = np.flip(preprocessed_text, axis=1)
    return preprocessed_text


def plot_attention(inputs, attention_weights, en_id2word, fr_id2word):
    outputs = []
    attentions = []
    for out_index, weights in attention_weights:
        if out_index != 0:
            outputs.append(out_index)
            attentions.append(weights.reshape(-1))
    attentions = np.transpose(np.array(attentions))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(attentions)

    ax.set_xticks(np.arange(attentions.shape[1]))
    ax.set_yticks(np.arange(attentions.shape[0]))

    ax.set_xticklabels(
        [fr_id2word[inp].decode("utf-8") if inp != 0 else "<Res>" for inp in outputs]
    )
    ax.set_yticklabels(
        [en_id2word[inp] if inp != 0 else "<Res>" for inp in inputs.ravel()]
    )

    ax.tick_params(labelsize=10)
    ax.tick_params(axis="x", labelrotation=90)

    plt.show()
