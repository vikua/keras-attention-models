import keras

from layers import BahdanauAttention, LuongAttentionDecoder, BahdanauAttentionDecoder


def create_attention_models(
    rnn_hidden_dim,
    encoder_timesteps,
    encoder_input_dim,
    decoder_timesteps,
    decoder_input_dim,
):
    encoder_inputs = keras.layers.Input(
        shape=(encoder_timesteps, encoder_input_dim), name="encoder_inputs"
    )
    decoder_inputs = keras.layers.Input(
        shape=(None, decoder_input_dim), name="decoder_inputs"
    )

    encoder = keras.layers.LSTM(
        rnn_hidden_dim, return_sequences=True, return_state=True, name="encoder"
    )
    encoder_out, encoder_state_h, encoder_state_c = encoder(encoder_inputs)
    encoder_states = [encoder_state_h, encoder_state_c]

    decoder = keras.layers.LSTM(
        rnn_hidden_dim, return_sequences=True, return_state=True, name="decoder"
    )
    decoder_out, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

    attention = BahdanauAttention(name="attention")
    attention_out, _ = attention([encoder_out, decoder_out])
    decoder_attention_concat = keras.layers.Concatenate()
    decoder_attention = decoder_attention_concat([decoder_out, attention_out])

    dense = keras.layers.Dense(
        decoder_input_dim, activation="softmax", name="softmax_layer"
    )
    decoder_pred = dense(decoder_attention)

    model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_pred)

    encoder_model = keras.models.Model(encoder_inputs, [encoder_out] + encoder_states)

    encoder_out_inputs = keras.layers.Input(
        shape=(encoder_timesteps, rnn_hidden_dim), name="encoder_out_inputs"
    )
    decoder_state_input_h = keras.layers.Input(
        shape=(rnn_hidden_dim,), name="decoder_h_inputs"
    )
    decoder_state_input_c = keras.layers.Input(
        shape=(rnn_hidden_dim,), name="decoder_c_inputs"
    )
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_out, decoder_state_h, decoder_state_c = decoder(
        decoder_inputs, initial_state=decoder_state_inputs
    )
    decoder_states = [decoder_state_h, decoder_state_c]
    attention_out, attention_state = attention([encoder_out_inputs, decoder_out])
    decoder_attention = decoder_attention_concat([decoder_out, attention_out])

    decoder_pred = dense(decoder_attention)
    decoder_model = keras.models.Model(
        [encoder_out_inputs, decoder_inputs] + decoder_state_inputs,
        [decoder_pred, attention_state] + decoder_states,
    )

    return model, encoder_model, decoder_model


def create_input_feeding_attention_models(
    rnn_hidden_dim,
    encoder_timesteps,
    encoder_input_dim,
    decoder_timesteps,
    decoder_input_dim,
    attention='luong_dot',
):
    keras.utils.generic_utils.get_custom_objects().update(
        {
            "LuongAttentionDecoder": LuongAttentionDecoder,
            "BahdanauAttentionDecoder": BahdanauAttentionDecoder,
        }
    )

    encoder_inputs = keras.layers.Input(
        shape=(encoder_timesteps, encoder_input_dim), name="encoder_inputs"
    )
    decoder_inputs = keras.layers.Input(
        shape=(None, decoder_input_dim), name="decoder_inputs"
    )
    context_inputs = keras.layers.Input(shape=(rnn_hidden_dim,), name="context_inputs")

    encoder = keras.layers.LSTM(
        rnn_hidden_dim, return_sequences=True, return_state=True, name="encoder"
    )
    encoder_out, encoder_state_h, encoder_state_c = encoder(encoder_inputs)
    encoder_states = [encoder_state_h, encoder_state_c]

    decoder = keras.layers.LSTM(
        rnn_hidden_dim, return_sequences=True, return_state=True, name="decoder"
    )

    if attention == 'luong_dot':
        attention_decoder = LuongAttentionDecoder(decoder, attn_type="dot")
    elif attention == 'luong_general':
        attention_decoder = LuongAttentionDecoder(decoder, attn_type="general")
    elif attention == 'luong_concat':
        attention_decoder = LuongAttentionDecoder(decoder, attn_type="concat")
    elif attention == 'bahdanau':
        attention_decoder = BahdanauAttentionDecoder(decoder)

    outputs = attention_decoder(
        [decoder_inputs, encoder_out, context_inputs] + encoder_states
    )
    decoder_out = outputs[0]
    attention_weights = outputs[1]
    last_context = outputs[2]
    decoder_states = outputs[3:]

    dense = keras.layers.Dense(
        decoder_input_dim, activation="softmax", name="softmax_layer"
    )
    decoder_pred = dense(decoder_out)

    model = keras.models.Model(
        [encoder_inputs, decoder_inputs, context_inputs], decoder_pred
    )

    encoder_model = keras.models.Model(encoder_inputs, [encoder_out] + encoder_states)

    encoder_out_inputs = keras.layers.Input(
        shape=(encoder_timesteps, rnn_hidden_dim), name="encoder_out_inputs"
    )
    last_context_input = keras.layers.Input(shape=(rnn_hidden_dim,))
    decoder_state_inpit_h = keras.layers.Input(shape=(rnn_hidden_dim,))
    decoder_state_input_c = keras.layers.Input(shape=(rnn_hidden_dim,))
    decoder_state_inputs = [
        last_context_input,
        decoder_state_inpit_h,
        decoder_state_input_c,
    ]

    outputs = attention_decoder(
        [decoder_inputs, encoder_out_inputs] + decoder_state_inputs
    )
    decoder_out = outputs[0]
    attention_weights = outputs[1]
    last_context = outputs[2]
    decoder_states = outputs[3:]

    decoder_pred = dense(decoder_out)
    decoder_model = keras.models.Model(
        [encoder_out_inputs, decoder_inputs] + decoder_state_inputs,
        [decoder_pred, last_context, attention_weights] + decoder_states,
    )

    return model, encoder_model, decoder_model
