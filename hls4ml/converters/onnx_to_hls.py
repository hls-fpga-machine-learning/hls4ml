from hls4ml.model import ModelGraph


# ----------------------Helpers---------------------
def sanitize_layer_name(layer):
    new_name = layer['name']
    if new_name[0].isdigit():
        new_name = layer['class_name'].lower() + new_name

    layer['name'] = new_name


def replace_char_inconsitency(name):
    """
    Replace some inconsistent characters that cause issues when writing into HLS.
    """
    return name.replace('.', '_')


def get_onnx_attribute(operation, name, default=None):
    from onnx import helper

    attr = next((x for x in operation.attribute if x.name == name), None)
    if attr is None:
        value = default
    else:
        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode()
    return value


def get_global_input_shape(graph, inp):
    """Return the global input shape of the graph with name inp

    Arguments:
        graph:  the onnx graph
        inp (str):  the global input name

    Returns:
        list: The shape

    Raises:
        StopIteration:  If the global input name is not found
    """
    inp_shape = next(x.type.tensor_type.shape.dim for x in graph.input if x.name == inp)
    return list(x.dim_value for x in inp_shape)


def get_input_shape(graph, node):
    """Return the input shapes of the node in the model

    Arguments:
        graph:  the onnx graph
        node:  the onnx node for which the input is desired

    Returns:
        list of lists: The shapes of all the inputs

    Raises:
        StopIteration:  If the an input name is not found in the graph
    """
    rv = []
    for inp in node.input:
        try:
            value_info_idx = next((i for i, x in enumerate(graph.value_info) if x.name == inp))
            dim = list(d.dim_value for d in graph.value_info[value_info_idx].type.tensor_type.shape.dim)
        except StopIteration:
            # The input is not in the graph, likely it's the input
            dim = get_global_input_shape(graph, inp)
        if dim:
            rv.append(dim)
    return rv


def get_constant_value(graph, constant_name):
    tensor = next((x for x in graph.initializer if x.name == constant_name), None)
    from onnx import numpy_helper
    return numpy_helper.to_array(tensor)


def compute_pads_1d(operation, layer):
    auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
    if auto_pad != 'NOTSET':
        if layer['in_width'] % layer['stride_width'] == 0:
            pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
        else:
            pad_along_width = max(layer['filt_width'] - (layer['in_width'] % layer['stride_width']), 0)

        pads = [pad_along_width // 2, pad_along_width - (pad_along_width // 2)]

        if auto_pad == 'SAME_UPPER':
            pads = sorted(pads)
        elif auto_pad == 'SAME_LOWER':
            pads = sorted(pads, reverse=True)
        else:  # 'VALID' padding
            pads = [0, 0]
    else:
        pads = get_onnx_attribute(operation, 'pads', [0, 0])

    return pads


def compute_pads_2d(operation, layer):
    auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
    if auto_pad != 'NOTSET':
        # Height
        if layer['in_height'] % layer['stride_height'] == 0:
            pad_along_height = max(layer['filt_height'] - layer['stride_height'], 0)
        else:
            pad_along_height = max(layer['filt_height'] - (layer['in_height'] % layer['stride_height']), 0)
        pad_height = [pad_along_height // 2, pad_along_height - pad_along_height // 2]

        # Width
        if layer['in_width'] % layer['stride_width'] == 0:
            pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
        else:
            pad_along_width = max(layer['filt_width'] - (layer['in_width'] % layer['stride_width']), 0)
        pad_width = [pad_along_width // 2, pad_along_width - pad_along_width // 2]

        if auto_pad == 'SAME_UPPER':
            pads = [min(pad_height), min(pad_width), max(pad_height), max(pad_width)]
        elif auto_pad == 'SAME_LOWER':
            pads = [max(pad_height), max(pad_width), min(pad_height), min(pad_width)]
        else:  # 'VALID' padding
            pads = [0, 0, 0, 0]
    else:
        pads = get_onnx_attribute(operation, 'pads', [0, 0, 0, 0])

    return pads


# ----------------------Layer handling---------------------
layer_handlers = {}


def register_onnx_layer_handler(layer_name, handler_func):
    if layer_name in layer_handlers:
        raise Exception(f'Layer {layer_name} already registered')
    else:
        layer_handlers[layer_name] = handler_func


def get_supported_onnx_layers():
    return list(layer_handlers.keys())


def onnx_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function

    return decorator


def get_out_layer_name(graph):
    """
    Get the output layer's name for the model.
    graph.output only returns the output's node index
    """
    output_index_list = [x.name for x in graph.output]
    return [node.name for node in graph.node if node.output[0] in output_index_list]


def parse_onnx_model(onnx_model):
    """Parses the onnx model, both for configuration building and general processing.

    Args:
        onnx_model: an ONNX model object.

    Raises:
        Exception: Raised if an unsupported operation is found in the ONNX model.

    Returns:
        layer_list (list):  The onnx layers
        input_layers (list):  The input layers
        output_layers (list):  The output layers
    """
    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    # We don't infer the shapes because the qonnx package preprocessing does it.

    # Obtain list of input/ouput layers
    all_inputs = [x.name for x in onnx_model.graph.input]
    all_initializers = [x.name for x in onnx_model.graph.initializer]
    input_layers = [x for x in all_inputs if x not in all_initializers]
    constant_layers = all_initializers  # no need to copy it even though we change it
    output_layers = get_out_layer_name(onnx_model.graph)

    print("Output layers: ", output_layers)

    for i, inp in enumerate(input_layers):
        input_layer = {}
        input_layer['name'] = replace_char_inconsitency(inp)
        input_layer['class_name'] = 'InputLayer'
        inp_shape = get_global_input_shape(onnx_model.graph, inp)
        # We only support ONNX where the first dimension is the batch dimension.
        # Remove the batch dimension in all subsequnt use
        input_layer['input_shape'] = inp_shape[1:]

        print('Input shape:', input_layer['input_shape'])
        # Clean the layer name for specific models
        sanitize_layer_name(input_layer)
        input_layers[i] = input_layer['name']

        layer_list.append(input_layer)

    for i, constant in enumerate(constant_layers):
        constant_layer = {}
        constant_layer['name'] = replace_char_inconsitency(constant)
        constant_layer['class_name'] = 'Constant'
        constant_layer['value'] = get_constant_value(onnx_model.graph, constant)

        # Clean the layer name for specific models
        sanitize_layer_name(constant_layer)
        constant_layers[i] = constant_layer['name']

        layer_list.append(constant_layer)

    # Defined supported layers and check for unsupported layer type
    skip_layers = ['Dropout', 'Identity']

    # Map inputs of skipped layers
    inputs_map = {}

    supported_layers = get_supported_onnx_layers() + skip_layers

    print('Topology:')
    for node in onnx_model.graph.node:
        if node.op_type not in supported_layers:
            raise Exception(f'ERROR: Unsupported operation type: {node.op_type}')

        # Note that at this point, input shape still contains batch dimension
        # in cases where it appears. That is not filtered out till later.
        input_shapes = get_input_shape(onnx_model.graph, node)

        if node.op_type in skip_layers:
            # Currently supported skipped layers have only one input and output
            # Skipped layers can follow each other

            # Mapping inputs
            input_name = inputs_map.get(node.input[0], node.input[0])
            output_name = node.output[0]
            inputs_map[output_name] = input_name
            continue

        input_names = [inputs_map.get(x, x) for x in node.input]

        # Process the layer
        layer = layer_handlers[node.op_type](node, input_names, input_shapes, onnx_model.graph)

        sanitize_layer_name(layer)
        print(f"Layer name: {layer['name']}, layer type: {layer['class_name']}, current shape: {input_shapes}")
        layer_list.append(layer)

    return layer_list, input_layers, output_layers


def onnx_to_hls(config):
    """Convert onnx model to hls model from configuration.

    Args:
        config (dict): ONNX configuration from yaml file or passed through API.

    Raises:
        Exception: Raised if an unsupported operation is found in the ONNX model.

    Returns:
        ModelGraph: hls4ml model object
    """

    # Extract model architecture
    print('Interpreting Model ...')

    import onnx
    onnx_model = onnx.load(config['OnnxModel']) if isinstance(config['OnnxModel'], str) else config['OnnxModel']

    layer_list, input_layers, output_layers = parse_onnx_model(onnx_model)

    #################
    # Generate HLS
    #################

    print('Creating HLS model')
    hls_model = ModelGraph(config, layer_list, input_layers, output_layers)
    return hls_model
