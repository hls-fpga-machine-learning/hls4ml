from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.utils import compute_padding_1d_pytorch, compute_padding_2d_pytorch, parse_data_format

pooling_layers = [
    'MaxPool1d',
    'MaxPool2d',
    'AvgPool1d',
    'AvgPool2d',
    'QuantMaxPool1d',
    'QuantMaxPool2d',
]  # TODO add support for special quantized average pool layers


@pytorch_handler(*pooling_layers)
def parse_pooling_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'Pool' in operation or 'pool' in operation

    layer = {}

    if 'MaxPool1d' in operation:
        layer['class_name'] = 'MaxPooling1D'
    if 'MaxPool2d' in operation:
        layer['class_name'] = 'MaxPooling2D'
    if operation == 'AvgPool1d':
        layer['class_name'] = 'AveragePooling1D'
    if operation == 'AvgPool2d':
        layer['class_name'] = 'AveragePooling2D'

    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['data_format'] = 'channels_first'  # Pytorch default (can't change)
    if node.op == 'call_module' and 'Avg' in operation:
        if class_object.count_include_pad:
            layer['count_pad'] = True
        else:
            layer['count_pad'] = False
    else:
        layer['count_pad'] = True

    if int(layer['class_name'][-2]) == 1:
        (layer['n_in'], layer['n_filt']) = parse_data_format(input_shapes[0], layer['data_format'])
        if node.op == 'call_module':
            layer['pool_width'] = (
                class_object.kernel_size if not type(class_object.kernel_size) is tuple else class_object.kernel_size[0]
            )
            layer['stride_width'] = class_object.stride if not type(class_object.stride) is tuple else class_object.stride[0]

            if type(class_object.padding) is tuple:
                padding = class_object.padding[0]
            else:
                padding = class_object.padding

        else:
            layer['pool_width'] = int(node.args[1])
            layer['stride_width'] = node.kwargs['stride'] if node.kwargs['stride'] is not None else int(node.args[1])
            padding = node.kwargs['padding']

        if padding == 0:  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
            layer['padding'] = 'valid'
        else:  # Only 'valid' and 'same' padding are available in Keras
            layer['padding'] = 'same'

        (layer['n_out'], layer['pad_left'], layer['pad_right']) = compute_padding_1d_pytorch(
            padding, layer['n_in'], layer['stride_width'], layer['pool_width'], 1
        )

        if layer['data_format'] == 'channels_last':
            output_shape = [input_shapes[0][0], layer['n_out'], layer['n_filt']]
        elif layer['data_format'] == 'channels_first':
            output_shape = [input_shapes[0][0], layer['n_filt'], layer['n_out']]

    elif int(layer['class_name'][-2]) == 2:
        (layer['in_height'], layer['in_width'], layer['n_filt']) = parse_data_format(input_shapes[0], layer['data_format'])

        if node.op == 'call_module':
            if type(class_object.stride) is tuple:
                layer['stride_height'] = class_object.stride[0]
                layer['stride_width'] = class_object.stride[1]
            else:
                layer['stride_height'] = class_object.stride
                layer['stride_width'] = class_object.stride

            if type(class_object.kernel_size) is tuple:
                layer['pool_height'] = class_object.kernel_size[0]
                layer['pool_width'] = class_object.kernel_size[1]
            else:
                layer['pool_height'] = class_object.kernel_size
                layer['pool_width'] = class_object.kernel_size

            if type(class_object.padding) is tuple:
                padding = class_object.padding
            else:
                padding = [class_object.padding, class_object.padding]

        else:
            if type(node.kwargs['stride']) is tuple:
                layer['stride_height'] = node.kwargs['stride'][0]
                layer['stride_width'] = node.kwargs['stride'][1]
            else:
                if node.kwargs['stride'] is None:
                    # if stride is not set it is supposed to default to the kernel size
                    layer['stride_height'] = node.args[1]
                    layer['stride_width'] = node.args[1]
                else:
                    layer['stride_height'] = node.kwargs['stride']
                    layer['stride_width'] = node.kwargs['stride']
            if type(node.args[1]) is tuple:
                layer['pool_height'] = node.args[1][0]
                layer['pool_width'] = node.args[1][1]
            else:
                layer['pool_height'] = node.args[1]
                layer['pool_width'] = node.args[1]
            if type(node.kwargs['padding']) is tuple:
                padding = node.kwargs['padding']
            else:
                padding = [node.kwargs['padding'], node.kwargs['padding']]

        if all(x == 0 for x in padding):  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
            layer['padding'] = 'valid'
        else:  # Only 'valid' and 'same' padding are available in Keras
            layer['padding'] = 'same'

        (
            layer['out_height'],
            layer['out_width'],
            layer['pad_top'],
            layer['pad_bottom'],
            layer['pad_left'],
            layer['pad_right'],
        ) = compute_padding_2d_pytorch(
            padding,
            layer['in_height'],
            layer['in_width'],
            layer['stride_height'],
            layer['stride_width'],
            layer['pool_height'],
            layer['pool_width'],
            1,
            1,
        )

        if layer['data_format'] == 'channels_last':
            output_shape = [input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]
        elif layer['data_format'] == 'channels_first':
            output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]

    return layer, output_shape
