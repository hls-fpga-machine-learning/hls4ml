import math
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler
from hls4ml.converters.utils import parse_data_format, compute_padding_1d, compute_padding_2d

pooling_layers = ['MaxPooling1D', 'MaxPooling2D', 'AveragePooling1D', 'AveragePooling2D']
@keras_handler(*pooling_layers)
def parse_pooling_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('Pooling' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)

    if int(layer['class_name'][-2]) == 1:
        (
            layer['n_in'],
            layer['n_filt']
        ) = parse_data_format(input_shapes[0], layer['data_format'])
        
        layer['pool_width']=keras_layer['config']['pool_size'][0]
        layer['stride_width']=keras_layer['config']['strides'][0]
        layer['padding']=keras_layer['config']['padding']

        (
            layer['n_out'],
            layer['pad_left'],
            layer['pad_right']
        ) = compute_padding_1d(
            layer['padding'],
            layer['n_in'],
            layer['stride_width'],
            layer['pool_width']
        )

        if layer['data_format'] == 'channels_last':
            output_shape=[input_shapes[0][0], layer['n_out'], layer['n_filt']]
        elif layer['data_format'] == 'channels_first':
            output_shape=[input_shapes[0][0], layer['n_filt'], layer['n_out']]
    elif int(layer['class_name'][-2]) == 2:
        (
            layer['in_height'],
            layer['in_width'],
            layer['n_filt']
        ) = parse_data_format(input_shapes[0], layer['data_format'])

        layer['stride_height']=keras_layer['config']['strides'][0]
        layer['stride_width']=keras_layer['config']['strides'][1]
        layer['pool_height']=keras_layer['config']['pool_size'][0]
        layer['pool_width']=keras_layer['config']['pool_size'][1]
        layer['padding']=keras_layer['config']['padding']

        (
            layer['out_height'],
            layer['out_width'],
            layer['pad_top'],
            layer['pad_bottom'],
            layer['pad_left'],
            layer['pad_right']
        ) = compute_padding_2d(
            layer['padding'],
            layer['in_height'],
            layer['in_width'],
            layer['stride_height'],
            layer['stride_width'],
            layer['pool_height'],
            layer['pool_width']
        )

        if layer['data_format'] == 'channels_last':
            output_shape=[input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]
        elif layer['data_format'] == 'channels_first':
            output_shape=[input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]
    
    return layer, output_shape

pooling_layers = ['GlobalMaxPooling1D', 'GlobalMaxPooling2D', 'GlobalAveragePooling1D', 'GlobalAveragePooling2D']
@keras_handler(*pooling_layers)
def parse_global_pooling_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert('Pooling' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)

    if int(layer['class_name'][-2]) == 1:
        (
            layer['n_in'],
            layer['n_filt']
        ) = parse_data_format(input_shapes[0], layer['data_format'])
        
        output_shape=[input_shapes[0][0], layer['n_filt']]
    elif int(layer['class_name'][-2]) == 2:
        (
            layer['in_height'],
            layer['in_width'],
            layer['n_filt']
        ) = parse_data_format(input_shapes[0], layer['data_format'])

        output_shape=[input_shapes[0][0], layer['n_filt']]
    
    return layer, output_shape
