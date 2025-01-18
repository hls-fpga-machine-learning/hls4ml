import numpy as np

from hls4ml.converters.pytorch_to_hls import pytorch_handler


@pytorch_handler('MultiheadAttention')
def parse_multiheadattention_layer(
    operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config
):
    assert 'MultiheadAttention' in operation
    assert len(input_shapes) == 3

    layer = {}

    layer['class_name'] = 'MultiHeadAttention'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    layer['num_heads'] = class_object.num_heads
    layer['head_dim_key'] = class_object.kdim // layer['num_heads']
    layer['head_dim_value'] = class_object.vdim // layer['num_heads']
    layer['query_shape'] = input_shapes[0]
    layer['key_shape'] = input_shapes[1]
    layer['value_shape'] = input_shapes[2]

    if not (len(layer['query_shape']) == len(layer['key_shape']) == len(layer['value_shape']) == 3):
        raise Exception('only 3D shapes for query, key, and value are currently supported by hls4ml')

    layer['feature_dim'] = class_object.embed_dim
    layer['seq_len'] = layer['query_shape'][-2]

    output_shape = layer['query_shape']

    layer['attention_axes'] = [1]
    layer['softmax_axis'] = [3]

    in_proj_weights = class_object.in_proj_weight.data.numpy()
    in_proj_bias = class_object.in_proj_bias.data.numpy()

    weight_data = np.split(in_proj_weights, [class_object.embed_dim, class_object.embed_dim + class_object.kdim], axis=0)
    bias_data = np.split(in_proj_bias, [class_object.embed_dim, class_object.embed_dim + class_object.kdim], axis=0)

    for weight_type, weight, bias in zip(['query', 'key', 'value'], weight_data, bias_data):
        layer[f'{weight_type}_weight_data'] = weight.T.reshape(
            layer['feature_dim'], layer['num_heads'], layer['head_dim_key']
        ).transpose(1, 0, 2)
        layer[f'{weight_type}_bias_data'] = bias.reshape(layer['num_heads'], layer['head_dim_key'])

    layer['attention_output_weight_data'] = class_object.out_proj.weight.data.numpy().T.reshape(
        layer['num_heads'], layer['head_dim_key'], layer['feature_dim']
    )
    layer['attention_output_bias_data'] = class_object.out_proj.bias.data.numpy()

    return layer, output_shape
