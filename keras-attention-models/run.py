import os
import random
import argparse
import pickle

import keras
import keras.backend as K
import tensorflow as tf
import numpy as np

from models import create_input_feeding_attention_models
from utils import get_data, sent2seq, plot_attention


def fit_tokenizers(X_train, y_train):
    x_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="UNK")
    x_tokenizer.fit_on_texts(X_train)

    y_tokenizer = keras.preprocessing.text.Tokenizer(oov_token="UNK")
    y_tokenizer.fit_on_texts(y_train)

    return x_tokenizer, y_tokenizer


def train(args):
    # load data
    source_train, target_train, source_test, target_test = get_data(
        args.source_data,
        args.target_data,
        test_size=args.test_size,
        samplepct=args.samplepct,
    )

    # create and save tokenizers
    x_tokenizer, y_tokenizer = fit_tokenizers(source_train, target_train)
    with open(
        os.path.join(args.dest_path, "{}_xtok.pkl".format(args.prefix)), "wb"
    ) as f:
        pickle.dump(x_tokenizer, f)
    with open(
        os.path.join(args.dest_path, "{}_ytok.pkl".format(args.prefix)), "wb"
    ) as f:
        pickle.dump(y_tokenizer, f)

    # pad sequences
    X_train = sent2seq(x_tokenizer, source_train, pad_type="pre", pad_length=20)
    y_train = sent2seq(y_tokenizer, target_train, pad_type="post", pad_length=20)
    X_test = sent2seq(x_tokenizer, source_test, pad_type="pre", pad_length=20)
    y_test = sent2seq(y_tokenizer, target_test, pad_type="post", pad_length=20)

    # create model
    X_vsize = max(x_tokenizer.index_word.keys()) + 1
    y_vsize = max(y_tokenizer.index_word.keys()) + 1
    model, encoder_model, decoder_model = create_input_feeding_attention_models(
        rnn_hidden_dim=128,
        encoder_timesteps=20,
        decoder_timesteps=20,
        encoder_input_dim=X_vsize,
        decoder_input_dim=y_vsize,
        attention=args.model_type,
    )

    # start training
    X_train_onehot = keras.utils.to_categorical(X_train, num_classes=X_vsize)
    y_train_onehot = keras.utils.to_categorical(y_train, num_classes=y_vsize)
    X_test_onehot = keras.utils.to_categorical(X_test, num_classes=X_vsize)
    y_test_onehot = keras.utils.to_categorical(y_test, num_classes=y_vsize)

    model.compile(optimizer="adam", loss="categorical_crossentropy")
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", restore_best_weights=True, patience=10
    )
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.dest_path, "{}.hdf5".format(args.prefix)),
        monitor="val_loss",
        save_best_only=True,
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
        epochs=20,
    )

    # inference loop
    encoder_out, encoder_state_h, encoder_state_c = encoder_model.predict(
        X_test_onehot, batch_size=64
    )
    decoder_states = [encoder_state_h, encoder_state_c]

    target_seq = np.empty((X_test_onehot.shape[0], 1, 1))
    target_seq.fill(y_tokenizer.word_index["sos"])
    target_seq = keras.utils.to_categorical(target_seq, num_classes=y_vsize)

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
        target_seq = keras.utils.to_categorical(pred, num_classes=y_vsize)

    decoded_seq = [
        " ".join([y_tokenizer.index_word[idx] for idx in seq if idx != 0])
        for seq in decoded_seq.tolist()
    ]

    for i in np.random.choice(range(1, len(decoded_seq)), 10):
        print(
            "\nen: {}\nfr act: {}\nfr pred: {}\n".format(
                source_test[i], target_test[i], decoded_seq[i]
            )
        )


def infer(args):
    # load data
    source_train, target_train, source_test, target_test = get_data(
        args.source_data,
        args.target_data,
        test_size=args.test_size,
        samplepct=args.samplepct,
    )

    # load tokenizers
    with open(
        os.path.join(args.dest_path, "{}_xtok.pkl".format(args.prefix)), "rb"
    ) as f:
        x_tokenizer = pickle.load(f)
    with open(
        os.path.join(args.dest_path, "{}_ytok.pkl".format(args.prefix)), "rb"
    ) as f:
        y_tokenizer = pickle.load(f)

    # preprocess data
    X_test = sent2seq(x_tokenizer, source_test, pad_type="pre", pad_length=20)

    X_vsize = max(x_tokenizer.index_word.keys()) + 1
    y_vsize = max(y_tokenizer.index_word.keys()) + 1

    X_test_onehot = keras.utils.to_categorical(X_test, num_classes=X_vsize)

    # load models
    model, encoder_model, decoder_model = create_input_feeding_attention_models(
        rnn_hidden_dim=128,
        encoder_timesteps=20,
        decoder_timesteps=20,
        encoder_input_dim=X_vsize,
        decoder_input_dim=y_vsize,
        attention=args.model_type,
    )
    model.load_weights(os.path.join(args.dest_path, "{}.hdf5".format(args.prefix)))

    # infer
    encoder_out, encoder_state_h, encoder_state_c = encoder_model.predict(
        X_test_onehot, batch_size=64
    )
    decoder_states = [encoder_state_h, encoder_state_c]

    target_seq = np.empty((X_test_onehot.shape[0], 1, 1))
    target_seq.fill(y_tokenizer.word_index["sos"])
    target_seq = keras.utils.to_categorical(target_seq, num_classes=y_vsize)

    last_context = np.zeros((X_test_onehot.shape[0], 128))

    decoded_seq = np.empty((X_test_onehot.shape[0], 20))

    infer_index = args.infer_index
    if infer_index is None:
        random_state = np.random.RandomState()
        infer_index = random_state.randint(low=0, high=X_test_onehot.shape[0])
    attention_results = []
    for i in range(20):
        decoder_pred, last_context, attn, h, c = decoder_model.predict(
            [encoder_out, target_seq, last_context] + decoder_states
        )

        decoder_states = [h, c]

        pred = np.argmax(decoder_pred, axis=-1)
        decoded_seq[:, i] = pred.ravel()

        attention_results.append((pred[infer_index, 0], attn[infer_index]))

        pred = np.expand_dims(pred, axis=2)
        target_seq = keras.utils.to_categorical(pred, num_classes=y_vsize)

    decoded_seq = [
        " ".join([y_tokenizer.index_word[idx] for idx in seq if idx != 0])
        for seq in decoded_seq.tolist()
    ]

    for i in np.random.choice(range(1, len(decoded_seq)), 10):
        print(
            "\n{} en: {}\nfr act: {}\nfr pred: {}\n".format(
                i, source_test[i], target_test[i], decoded_seq[i]
            )
        )

    plot_attention(
        X_test[infer_index],
        attention_results,
        x_tokenizer.index_word,
        y_tokenizer.index_word,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test seq2seq attention models")
    parser.add_argument(
        "--source_data", default="./data/small_vocab_en", help="source data"
    )
    parser.add_argument(
        "--target_data", default="./data/small_vocab_fr", help="target data"
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        choices=["infer", "train"],
        help="train or infer",
    )
    parser.add_argument(
        "--model_type",
        default="luong_general",
        type=str,
        choices=["luong_dot", "luong_general", "luong_concat", "bahdanau"],
        help="Attention implementation",
    )
    parser.add_argument(
        "--samplepct", type=float, default=1.0, help="Size of a random sample to use"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Size of the test set"
    )
    parser.add_argument(
        "--dest_path", type=str, default="./bin", help="Trained model destination path"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="model",
        help="Prefix of all models (tokenizers and keras model itsefl)",
    )
    parser.add_argument(
        "--infer_index", type=int, help="Specific index to sample and infer on"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(session)

    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
