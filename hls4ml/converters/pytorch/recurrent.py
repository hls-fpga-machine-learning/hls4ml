import numpy as np

from hls4ml.converters.pytorch_to_hls import pytorch_handler

rnn_layers = ['RNN', 'LSTM', 'GRU']


@pytorch_handler(*rnn_layers)
def parse_rnn_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation in rnn_layers

    layer = {}

    layer["name"] = layer_name

    layer['inputs'] = input_names
    if 'IOType' in config.keys():
        if len(input_names) > 1 and config['IOType'] == 'io_stream':
            raise Exception('Passing initial values for the hidden state is not support for io_stream input type.')

    layer['class_name'] = operation
    if operation == 'RNN':
        layer['class_name'] = 'SimpleRNN'

    layer['return_sequences'] = False  # parameter does not exist in pytorch
    layer['return_state'] = False  # parameter does not exist in pytorch

    if layer['class_name'] == 'SimpleRNN':
        layer['activation'] = class_object.nonlinearity  # Default is tanh, can also be ReLU in pytorch
    else:
        layer['activation'] = 'tanh'  # GRU and LSTM are hard-coded to use tanh in pytorch

    if layer['class_name'] == 'GRU' or layer['class_name'] == 'LSTM':
        layer['recurrent_activation'] = 'sigmoid'  # GRU and LSTM are hard-coded to use sigmoid in pytorch

    layer['time_major'] = not class_object.batch_first
    # TODO Should we handle time_major?
    if layer['time_major']:
        raise Exception('hls4ml only supports "batch-first == True"')

    layer['n_timesteps'] = input_shapes[0][1]
    layer['n_in'] = input_shapes[0][2]

    layer['n_out'] = class_object.hidden_size

    if class_object.num_layers > 1:
        raise Exception('hls4ml does not support num_layers > 1')

    if class_object.bidirectional:
        raise Exception('hls4ml does not support birectional RNNs')
    if class_object.dropout > 0:
        raise Exception('hls4ml does not support RNNs with dropout')

    layer['weight_data'] = class_object.weight_ih_l0.data.numpy()
    layer['recurrent_weight_data'] = class_object.weight_hh_l0.data.numpy()
    layer['bias_data'] = class_object.bias_ih_l0.data.numpy()
    layer['recurrent_bias_data'] = class_object.bias_hh_l0.data.numpy()

    if class_object.bias is False:
        layer['bias_data'] = np.zeros(layer['weight_data'].shape[0])
        layer['recurrent_bias_data'] = np.zeros(layer['recurrent_weight_data'].shape[0])

    if layer['class_name'] == 'GRU':
        layer['apply_reset_gate'] = 'after'  # Might be true for pytorch? It's not a free parameter

    output_shape = [input_shapes[0][0], layer['n_out']]

    layer['pytorch'] = True  # need to switch some behaviors to match pytorch implementations
    if len(input_names) == 1:
        layer['pass_initial_states'] = False
    else:
        layer['pass_initial_states'] = True

    return layer, output_shape
