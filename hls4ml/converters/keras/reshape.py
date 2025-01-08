import numpy as np

from hls4ml.converters.keras_to_hls import keras_handler, parse_default_keras_layer
from hls4ml.converters.utils import parse_data_format


@keras_handler('Flatten')
def parse_flatten_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer["class_name"] == 'Flatten'

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['class_name'] = 'Reshape'
    layer['target_shape'] = [np.prod(input_shapes[0][1:])]  # target shape has no batch dimension
    output_shape = input_shapes[0][:1] + layer['target_shape']

    return layer, output_shape


@keras_handler('Reshape')
def parse_reshape_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer["class_name"] == 'Reshape'

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['target_shape'] = keras_layer['config']['target_shape']
    output_shape = input_shapes[0][:1] + list(keras_layer['config']['target_shape'])

    return layer, output_shape


@keras_handler('UpSampling1D')
def parse_upsampling1d_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'UpSampling' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['in_height'] = 1
    (layer['in_width'], layer['n_chan']) = parse_data_format(input_shapes[0], layer['data_format'])

    layer['algorithm'] = 'nearest'

    layer['width_factor'] = keras_layer['config']['size']

    layer['out_height'] = 1
    layer['out_width'] = layer['in_width'] * layer['width_factor']

    if layer['data_format'] == 'channels_first':
        output_shape = [input_shapes[0][0], layer['n_chan'], layer['out_width']]
    else:
        output_shape = [input_shapes[0][0], layer['out_width'], layer['n_chan']]

    return layer, output_shape


@keras_handler('UpSampling2D')
def parse_upsampling2d_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'UpSampling2D' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    (layer['in_height'], layer['in_width'], layer['n_chan']) = parse_data_format(input_shapes[0], layer['data_format'])

    layer['algorithm'] = keras_layer['config']['interpolation']

    layer['height_factor'] = keras_layer['config']['size'][0]
    layer['width_factor'] = keras_layer['config']['size'][1]

    layer['out_height'] = layer['in_height'] * layer['height_factor']
    layer['out_width'] = layer['in_width'] * layer['width_factor']

    if layer['data_format'] == 'channels_first':
        output_shape = [input_shapes[0][0], layer['n_chan'], layer['out_height'], layer['out_width']]
    else:
        output_shape = [input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_chan']]

    return layer, output_shape


@keras_handler('Permute')
def parse_permute_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] == 'Permute'

    layer = parse_default_keras_layer(keras_layer, input_names)

    layer['class_name'] = 'Transpose'
    dims = keras_layer['config']['dims']
    layer['perm'] = [dim - 1 for dim in keras_layer['config']['dims']]

    output_shape = [input_shapes[0][0]] + [input_shapes[0][s] for s in dims]

    return layer, output_shape
