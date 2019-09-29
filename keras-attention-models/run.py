import random

import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from layers import LuongAttentionDecoder
from models import create_attentive_models
from utils import get_data, sent2seq, plot_attention


def train():
    en_train, fr_train, en_test, fr_test = get_data(0.2, samplepct=1)

    en_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="UNK")
    en_tokenizer.fit_on_texts(en_train)

    fr_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="UNK")
    fr_tokenizer.fit_on_texts(fr_train)

    X_train = sent2seq(en_tokenizer, en_train, pad_type="pre", pad_length=20)
    y_train = sent2seq(fr_tokenizer, fr_train, pad_type="post", pad_length=20)
    X_test = sent2seq(en_tokenizer, en_test, pad_type="pre", pad_length=20)
    y_test = sent2seq(fr_tokenizer, fr_test, pad_type="post", pad_length=20)

    en_vsize = max(en_tokenizer.index_word.keys()) + 1
    fr_vsize = max(fr_tokenizer.index_word.keys()) + 1

    X_train_onehot = keras.utils.to_categorical(X_train, num_classes=en_vsize)
    y_train_onehot = keras.utils.to_categorical(y_train, num_classes=fr_vsize)
    X_test_onehot = keras.utils.to_categorical(X_test, num_classes=en_vsize)
    y_test_onehot = keras.utils.to_categorical(y_test, num_classes=fr_vsize)

    model, encoder_model, decoder_model = create_attentive_models(
        rnn_hidden_dim=128,
        encoder_timesteps=20,
        decoder_timesteps=20,
        encoder_input_dim=en_vsize,
        decoder_input_dim=fr_vsize,
    )

    # training
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", restore_best_weights=True, patience=5
    )
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath="loop_best_train.hdf5", monitor="val_loss", save_best_only=True
    )
    context_train = np.zeros((X_train_onehot.shape[0], 128))
    context_test = np.zeros((X_test_onehot.shape[0], 128))
    model.fit(
        [X_train_onehot, y_train_onehot[:, :-1, :], context_train],
        y_train_onehot[:, 1:, :],
        validation_data=(
            [X_test_onehot, y_test_onehot[:, :-1, :], context_test],
            y_test_onehot[:, 1:, :],
        ),
        callbacks=[model_checkpoint_cb, early_stopping_cb],
        batch_size=64,
        epochs=10,
    )

    # inference
    encoder_out, encoder_state_h, encoder_state_c = encoder_model.predict(
        X_test_onehot, batch_size=64
    )
    decoder_states = [encoder_state_h, encoder_state_c]

    target_seq = np.empty((X_test_onehot.shape[0], 1, 1))
    target_seq.fill(fr_tokenizer.word_index["sos"])
    target_seq = keras.utils.to_categorical(target_seq, num_classes=fr_vsize)

    last_context = np.zeros((X_test_onehot.shape[0], 128))

    decoded_seq = np.empty((X_test_onehot.shape[0], 20))

    for i in range(20):
        decoder_pred, last_context, attn, h, c = decoder_model.predict(
            [encoder_out, target_seq, last_context] + decoder_states
        )

        decoder_states = [h, c]

        pred = np.argmax(decoder_pred, axis=-1)
        decoded_seq[:, i] = pred.ravel()

        pred = np.expand_dims(pred, axis=2)
        target_seq = keras.utils.to_categorical(pred, num_classes=fr_vsize)

    decoded_seq = [
        " ".join([fr_tokenizer.index_word[idx] for idx in seq if idx != 0])
        for seq in decoded_seq.tolist()
    ]

    for i in np.random.choice(range(1, len(decoded_seq)), 10):
        print(
            "\nen: {}\nfr act: {}\nfr pred: {}\n".format(
                en_test[i], fr_test[i], decoded_seq[i]
            )
        )


def infer():
    en_train, fr_train, en_test, fr_test = get_data(0.2, samplepct=1)

    en_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="UNK")
    en_tokenizer.fit_on_texts(en_train)

    fr_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="UNK")
    fr_tokenizer.fit_on_texts(fr_train)

    X_test = sent2seq(en_tokenizer, en_test, pad_type="pre", pad_length=20)
    y_test = sent2seq(fr_tokenizer, fr_test, pad_type="post", pad_length=20)

    en_vsize = max(en_tokenizer.index_word.keys()) + 1
    fr_vsize = max(fr_tokenizer.index_word.keys()) + 1

    X_test_onehot = keras.utils.to_categorical(X_test, num_classes=en_vsize)
    y_test_onehot = keras.utils.to_categorical(y_test, num_classes=fr_vsize)

    model, encoder_model, decoder_model = create_model(
        rnn_hidden_dim=128,
        encoder_timesteps=20,
        decoder_timesteps=20,
        encoder_input_dim=en_vsize,
        decoder_input_dim=fr_vsize,
    )
    model.load_weights("loop_best_train.hdf5")

    encoder_out, encoder_state_h, encoder_state_c = encoder_model.predict(
        X_test_onehot, batch_size=64
    )
    decoder_states = [encoder_state_h, encoder_state_c]

    target_seq = np.empty((X_test_onehot.shape[0], 1, 1))
    target_seq.fill(fr_tokenizer.word_index["sos"])
    target_seq = keras.utils.to_categorical(target_seq, num_classes=fr_vsize)

    last_context = np.zeros((X_test_onehot.shape[0], 128))

    decoded_seq = np.empty((X_test_onehot.shape[0], 20))

    sample_index = 12367
    attention_results = []
    for i in range(20):
        decoder_pred, last_context, attn, h, c = decoder_model.predict(
            [encoder_out, target_seq, last_context] + decoder_states
        )

        decoder_states = [h, c]

        pred = np.argmax(decoder_pred, axis=-1)
        decoded_seq[:, i] = pred.ravel()

        attention_results.append((pred[sample_index, 0], attn[sample_index]))

        pred = np.expand_dims(pred, axis=2)
        target_seq = keras.utils.to_categorical(pred, num_classes=fr_vsize)

    decoded_seq = [
        " ".join([fr_tokenizer.index_word[idx] for idx in seq if idx != 0])
        for seq in decoded_seq.tolist()
    ]

    for i in np.random.choice(range(1, len(decoded_seq)), 10):
        print(
            "\n{} en: {}\nfr act: {}\nfr pred: {}\n".format(
                i, en_test[i], fr_test[i], decoded_seq[i]
            )
        )

    plot_attention(
        X_test[sample_index],
        attention_results,
        en_tokenizer.index_word,
        fr_tokenizer.index_word,
    )


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(session)

    keras.utils.generic_utils.get_custom_objects().update(
        {"LuongAttentionDecoder": LuongAttentionDecoder}
    )

    train()
