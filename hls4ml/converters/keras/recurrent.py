from hls4ml.converters.keras_to_hls import get_weights_data, keras_handler, parse_default_keras_layer

rnn_layers = ['SimpleRNN', 'LSTM', 'GRU']


@keras_handler(*rnn_layers)
def parse_rnn_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] in rnn_layers or keras_layer['class_name'][1:] in rnn_layers

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['return_sequences'] = keras_layer['config']['return_sequences']
    layer['return_state'] = keras_layer['config']['return_state']

    if 'SimpleRNN' not in layer['class_name']:
        layer['recurrent_activation'] = keras_layer['config']['recurrent_activation']

    layer['time_major'] = keras_layer['config']['time_major'] if 'time_major' in keras_layer['config'] else False

    # TODO Should we handle time_major?
    if layer['time_major']:
        raise Exception('Time-major format is not supported by hls4ml')

    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]

    layer['n_out'] = keras_layer['config']['units']

    layer['weight_data'], layer['recurrent_weight_data'], layer['bias_data'] = get_weights_data(
        data_reader, layer['name'], ['kernel', 'recurrent_kernel', 'bias']
    )

    if 'GRU' in layer['class_name']:
        layer['apply_reset_gate'] = 'after' if keras_layer['config']['reset_after'] else 'before'

        # biases array is actually a 2-dim array of arrays (bias + recurrent bias)
        # both arrays have shape: n_units * 3 (z, r, h_cand)
        biases = layer['bias_data']
        layer['bias_data'] = biases[0]
        layer['recurrent_bias_data'] = biases[1]

    if layer['return_sequences']:
        output_shape = [input_shapes[0][0], layer['n_timesteps'], layer['n_out']]
    else:
        output_shape = [input_shapes[0][0], layer['n_out']]

    if layer['return_state']:
        raise Exception('"return_state" of {} layer is not yet supported.')

    layer['pass_initial_states'] = False

    return layer, output_shape
