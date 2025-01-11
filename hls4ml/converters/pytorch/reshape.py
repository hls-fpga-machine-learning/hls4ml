import numpy as np

from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.utils import parse_data_format

reshape_layers = ['View']


@pytorch_handler(*reshape_layers)
def parse_reshape_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'View'

    layer = {}
    layer['class_name'] = 'Reshape'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    layer['target_shape'] = [int(i) for i in node.args[1:]]
    # View can have -1 as one as the dimensions,
    # leaving it to us to deduce it from the other dimensions and the overall size
    if -1 in layer['target_shape']:
        size = np.prod(input_shapes[0][1:])
        for i in range(0, len(layer['target_shape'])):
            if layer['target_shape'][i] == -1:
                cl = layer['target_shape'][:]
                cl.remove(-1)
                layer['target_shape'][i] = int(size / np.prod(cl))

    output_shape = input_shapes[0][:1] + layer['target_shape']

    return layer, output_shape


@pytorch_handler('squeeze')
def parse_squeeze_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'squeeze'

    layer = {}
    layer['class_name'] = 'Reshape'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    if len(node.args) > 1 or len(node.kwargs) > 0:  # 'dim' argument is specified
        output_shape = [i for i in input_shapes[0]]
        squeeze_dim = node.kwargs.get('dim', None)
        if squeeze_dim is None:
            squeeze_dim = node.args[1]
        if isinstance(squeeze_dim, tuple):
            for dim in squeeze_dim:
                del output_shape[dim]
        else:
            del output_shape[squeeze_dim]
    else:
        output_shape = [i for i in input_shapes[0] if i != 1]

    layer['target_shape'] = output_shape.copy()
    if layer['target_shape'][0] is None:
        del layer['target_shape'][0]

    return layer, output_shape


@pytorch_handler('unsqueeze')
def parse_unsqueeze_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'unsqueeze'

    layer = {}
    layer['class_name'] = 'Reshape'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    # Unlike in 'squeeze' in 'unsqueeze', dim argument must exist
    output_shape = [i for i in input_shapes[0]]
    if len(node.args) > 1:  # Specified as unsqueeze(x, n)
        squeeze_dim = node.args[1]
    else:  # Specified as unsqueeze(x, dim=n)
        squeeze_dim = node.kwargs['dim']
    # insert() will add an element before the index, unsqueeze expects the location
    index = output_shape.index(output_shape[squeeze_dim])  # + 1
    output_shape.insert(index, 1)

    layer['target_shape'] = output_shape.copy()
    if layer['target_shape'][0] is None:
        del layer['target_shape'][0]

    return layer, output_shape


@pytorch_handler('Flatten')
def parse_flatten_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert operation == 'Flatten'

    layer = {}
    layer['class_name'] = 'Reshape'
    layer['name'] = layer_name
    layer['inputs'] = input_names
    if node.op == 'call_module':
        start_dim = class_object.start_dim
        end_dim = class_object.end_dim
        if end_dim + 1 == 0 or end_dim + 1 > len(input_shapes[0]):
            end_dim = len(input_shapes[0])
        else:
            end_dim = end_dim + 1
    else:
        start_dim = node.args[1]
        if len(node.args) == 3:
            end_dim = node.args[2]
        else:
            end_dim = -1
        if end_dim + 1 == 0 or end_dim + 1 > len(input_shapes[0]):
            end_dim = len(input_shapes[0])
        else:
            end_dim = end_dim + 1

    layer['target_shape'] = (
        input_shapes[0][0:start_dim] + [np.prod(input_shapes[0][start_dim:end_dim])] + input_shapes[0][end_dim:]
    )
    output_shape = layer['target_shape']

    return layer, output_shape


@pytorch_handler('Upsample', 'UpsamplingNearest2d', 'UpsamplingBilinear2d')
def handle_upsample(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):

    assert operation in ['Upsample', 'UpsamplingNearest2d', 'UpsamplingBilinear2d']
    layer = {}
    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['class_name'] = 'Resize'
    layer['data_format'] = 'channels_first'

    input_shape = parse_data_format(input_shapes[0], 'channels_first')
    if len(input_shape) == 2:
        layer['in_height'] = 1
        layer['in_width'], layer['n_chan'] = input_shape

        layer['out_height'] = 1
        layer['out_width'] = int(layer['in_width'] * class_object.scale_factor)

        output_shape = [input_shapes[0][0], layer['n_chan'], layer['out_width']]
    elif len(input_shape) == 3:
        layer['in_height'], layer['in_width'], layer['n_chan'] = input_shape

        scale_factor = class_object.scale_factor
        if isinstance(scale_factor, tuple):
            scale_height = scale_factor[0]
            scale_width = scale_factor[1]
        else:
            scale_height = scale_factor
            scale_width = scale_factor

        layer['out_height'] = int(layer['in_height'] * scale_height)
        layer['out_width'] = int(layer['in_width'] * scale_width)

        output_shape = [layer['n_chan'], layer['out_height'], layer['out_width']]
    else:
        raise Exception(f'Parsing "Upsample" with {len(input_shape)}-dimensional tensors is not yet supported.')

    layer['algorithm'] = class_object.mode
    layer['align_corners'] = bool(class_object.align_corners)

    return layer, output_shape
