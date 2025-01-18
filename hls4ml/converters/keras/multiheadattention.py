from hls4ml.converters.keras_to_hls import get_weights_data, keras_handler, parse_default_keras_layer


@keras_handler('MultiHeadAttention')
def parse_mutiheadattention_layer(keras_layer, input_names, input_shapes, data_reader):
    # assume input_shapes is: [[None, seq, dim]]
    assert 'MultiHeadAttention' in keras_layer['class_name']
    assert input_shapes[0] == keras_layer['config']['query_shape']

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['num_heads'] = keras_layer['config']['num_heads']
    layer['head_dim_key'] = keras_layer['config']['key_dim']
    layer['head_dim_value'] = keras_layer['config']['value_dim']
    layer['query_shape'] = keras_layer['config']['query_shape']
    layer['key_shape'] = keras_layer['config']['key_shape']
    layer['value_shape'] = keras_layer['config']['value_shape']
    layer['feature_dim'] = layer['query_shape'][-1]
    layer['seq_len'] = layer['query_shape'][-2]

    if keras_layer['config']['output_shape']:
        raise Exception('hls4ml does not support a defined output shape, the output shape must equal to the query shape')
    else:
        output_shape = layer['query_shape']

    layer['attention_axes'] = (
        keras_layer['config']['attention_axes'] if (keras_layer['config']['attention_axes'][0] == 1) else False
    )
    if layer['attention_axes'] is False:
        raise Exception('assigning the attention_axes is not currently supported by hls4ml')

    if not (len(layer['query_shape']) == 3 and len(layer['key_shape']) == 3 and len(layer['value_shape']) == 3):
        raise Exception('only 3D shapes for query, key, and value are currently supported by hls4ml')

    attn_scores_rank = 4
    layer['softmax_axis'] = list(range(attn_scores_rank - len(layer['attention_axes']), attn_scores_rank))

    weights_sources = [
        ('attention_output', 'kernel'),
        ('attention_output', 'bias'),
        ('key', 'kernel'),
        ('key', 'bias'),
        ('query', 'kernel'),
        ('query', 'bias'),
        ('value', 'kernel'),
        ('value', 'bias'),
    ]

    for lname, wtype in weights_sources:
        data = get_weights_data(data_reader, layer['name'], f'{lname}/{wtype}')
        if wtype == 'kernel':
            vtype = 'weight'
            if lname in ['key', 'query', 'value']:
                data = data.transpose((1, 0, 2))
        else:
            vtype = 'bias'

        layer[f'{lname}_{vtype}_data'] = data

    return layer, output_shape
