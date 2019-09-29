import tensorflow as tf
import keras
from keras.layers.wrappers import Wrapper
from keras import regularizers, constraints, initializers
from keras import backend as K
from keras.engine import InputSpec


class LuongAttentionDecoder(Wrapper):
    def __init__(self, layer, attn_type="dot", do_fc=False, **kwargs):
        self.supports_masking = True
        self.attn_type = attn_type
        self.do_fc = do_fc

        self.initializer = initializers.get("glorot_uniform")
        self.regularizer = regularizers.get(None)
        self.constraint = constraints.get(None)

        super(LuongAttentionDecoder, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=x) for x in input_shape]

        decoder_inp_shape, encoder_out_shape, context_shape, _, _ = input_shape
        dim = decoder_inp_shape[-1] + encoder_out_shape[-1]

        self.encoder_dim = encoder_out_shape[-1]
        self.decoder_dim = self.layer.units
        self.context_dim = context_shape[-1]
        self.time_steps = encoder_out_shape[1]

        self.layer.build(input_shape=(decoder_inp_shape[0], decoder_inp_shape[1], dim))

        if self.attn_type == "general":
            self.W_a = self.add_weight(
                name="W_a",
                shape=(self.encoder_dim, self.encoder_dim),
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
            )
        elif self.attn_type == "concat":
            self.W_a = self.add_weight(
                name="W_a",
                shape=(self.encoder_dim + self.decoder_dim, self.encoder_dim),
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
            )
            self.V_a = self.add_weight(
                name="V_a",
                shape=(self.encoder_dim, 1),
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
            )

        if self.do_fc:
            self.W_c = self.add_weight(
                name="W_c",
                shape=(self.decoder_dim + self.context_dim, self.decoder_dim),
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
            )

        super(LuongAttentionDecoder, self).build(input_shape)

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list) and len(inputs) >= 3

        decoder_inputs = inputs[0]
        encoder_outputs = inputs[1]
        context = inputs[2]
        encoder_states = inputs[3:]

        def dot_score(encoder_outs, decoder_out):
            # (1, dec_dim) . (time_steps, enc_dim)
            # dot attention
            # score shape: (batch_size, 1, time_steps)
            enc_transposed = tf.transpose(encoder_outs, perm=[0, 2, 1])
            score = tf.matmul(tf.expand_dims(decoder_out, axis=1), enc_transposed)
            return score

        def general_score(encoder_outs, decoder_out):
            W_a_dot_hs = K.dot(encoder_outs, self.W_a)
            W_a_dot_hs = tf.transpose(W_a_dot_hs, perm=[0, 2, 1])
            score = tf.matmul(tf.expand_dims(decoder_out, axis=1), W_a_dot_hs)
            return score

        def concat_score(encoder_outs, decoder_out):
            h_t = tf.expand_dims(decoder_out, 1)
            h_t = tf.tile(h_t, [1, self.time_steps, 1])
            ht_hs = tf.concat([h_t, encoder_outs], 2)

            W_a_dot_ht_hs = K.dot(ht_hs, self.W_a)
            W_a_dot_ht_hs = K.tanh(W_a_dot_ht_hs)
            score = K.dot(W_a_dot_ht_hs, self.V_a)
            score = K.squeeze(score, axis=-1)
            score = tf.expand_dims(score, 1)
            return score

        def step(x, states):
            last_context = states[0]
            states = states[1:]

            # encoder_outputs shape: (batch_size, time_steps, enc_dim)
            # decoder_input (x) shape: (batch_size, input_dim)
            # c_t shape: (batch_size, enc_dim)
            inp = K.concatenate([x, last_context])
            # decoder_out shape: (batch_size, dec_dim)
            # decoder states shape [(batch_size, dec_dim), (batch_size, dec_dim)]
            decoder_out, decoder_states = self.layer.cell.call(inp, states)

            if self.attn_type == "dot":
                score = dot_score(encoder_outputs, decoder_out)
            elif self.attn_type == "general":
                score = general_score(encoder_outputs, decoder_out)
            elif self.attn_type == "concat":
                score = concat_score(encoder_outputs, decoder_out)
            else:
                raise ValueError("Unknown score function {}".format(self.attn_type))

            # alignment vector a_t shape: (batch_size,1,  time_steps)
            energy = tf.nn.softmax(score)

            # context vector c_t is the avg sum of encoder out, shape: (batch_size, 1, enc_dim)
            context = tf.matmul(energy, encoder_outputs)

            # context shape: (batch_size, enc_dim)
            context = tf.squeeze(context, axis=1)
            # energy shape: (batch_size, time_steps)
            energy = tf.squeeze(energy, axis=1)

            if self.do_fc:
                decoder_out = tf.concat([decoder_out, context], 1)
                decoder_out = K.dot(decoder_out, self.W_c)
                decoder_out = K.tanh(decoder_out)
                context = decoder_out

            output = tf.concat([decoder_out, context, energy], 1)
            return output, [context] + decoder_states

        last_output, outputs, states = K.rnn(
            step,
            decoder_inputs,
            initial_states=[context] + encoder_states,
            go_backwards=self.layer.go_backwards,
            mask=mask,
            unroll=self.layer.unroll,
        )

        decoder_outputs = outputs[:, :, : self.decoder_dim + self.context_dim]
        energy_outputs = outputs[:, :, -self.time_steps :]

        return [decoder_outputs, energy_outputs] + list(states)

    def compute_output_shape(self, input_shape):
        batch_size, decoder_time_dim, _ = input_shape[0]
        _, _, encoder_dim = input_shape[1]
        return [
            (batch_size, decoder_time_dim, self.decoder_dim + self.context_dim),
            (batch_size, decoder_time_dim, self.time_steps),
            (batch_size, self.decoder_dim),
            (batch_size, self.decoder_dim),
            (batch_size, self.decoder_dim),
        ]

    def get_config(self):
        config = super(LuongAttentionDecoder, self).get_config()
        config["attn_type"] = self.attn_type
        config["do_fc"] = self.do_fc
        return config
