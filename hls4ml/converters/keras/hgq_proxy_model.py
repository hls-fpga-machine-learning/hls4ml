from hls4ml.converters.keras_to_hls import KerasReader, keras_handler, parse_default_keras_layer


@keras_handler('FixedPointQuantizer', 'HGQ>FixedPointQuantizer')
def fixedpoint_quantizer_handler(keras_layer, input_names, input_shapes, data_reader: KerasReader):
    config = parse_default_keras_layer(keras_layer, input_names)

    name = config['name']
    fusible = keras_layer['config']['fusible']
    config['RND'] = keras_layer['config']['RND']
    config['SAT'] = keras_layer['config']['SAT']
    config['fusible'] = fusible
    k = data_reader.get_weights_data(name, 'keep_negative')
    b = data_reader.get_weights_data(name, 'bits')
    i = data_reader.get_weights_data(name, 'integers')

    if fusible:
        k, b, i = k.ravel()[0], b.ravel()[0], i.ravel()[0]

    config['mask_kbi'] = k, b, i
    config['overrides'] = keras_layer['config']['overrides']

    layer = config
    return layer, input_shapes[0]


@keras_handler('UnaryLUT', 'HGQ>UnaryLUT')
def unary_lut_keras_handler(keras_layer, input_names, input_shapes, data_reader: KerasReader):
    config = parse_default_keras_layer(keras_layer, input_names)

    table = data_reader.get_weights_data(config['name'], 'table')
    k, i, f = keras_layer['config']['kif_out']
    k, b, i = k, k + i + f, k + i
    config['table_t'] = f'{"" if k else "u"}fixed<{b},{i}>'
    config['table'] = table
    config['table_size'] = len(table)
    config['activation'] = 'unary_lut'

    layer = config
    return layer, input_shapes[0]
