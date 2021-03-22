import numpy as np

from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler
from hls4ml.converters.utils import parse_data_format

@keras_handler('Reshape')
def parse_reshape_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer["class_name"] == 'Reshape')

    layer = parse_default_keras_layer(keras_layer, input_names)
    
    layer['target_shape'] = keras_layer['config']['target_shape']
    output_shape = input_shapes[0][:1] + keras_layer['config']['target_shape']
    
    return layer, output_shape


@keras_handler('UpSampling2D')
def parse_conv2d_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('UpSampling2D' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)
    
    (
        layer['in_height'],
        layer['in_width'],
        layer['n_chan']
    ) = parse_data_format(input_shapes[0], layer['data_format'])

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