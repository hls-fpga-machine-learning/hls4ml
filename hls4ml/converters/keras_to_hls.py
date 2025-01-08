import json

import h5py

from hls4ml.model import ModelGraph

MAXMULT = 4096


class KerasReader:
    def get_weights_data(self, layer_name, var_name):
        raise NotImplementedError


class KerasFileReader(KerasReader):
    def __init__(self, config):
        self.config = config
        self.h5file = h5py.File(config['KerasH5'], mode='r')

    def __del__(self):
        if self.h5file:
            self.h5file.close()

    def _find_data(self, layer_name, var_name):
        def h5_visitor_func(name):
            if var_name in name:
                return name

        if 'model_weights' in list(self.h5file.keys()):  # h5 file comes from model.save()
            layer_path = f'model_weights/{layer_name}'
        else:
            layer_path = layer_name

        data_path = self.h5file[layer_path].visit(h5_visitor_func)
        if data_path:
            return self.h5file[f'/{layer_path}/{data_path}']
        else:
            return None

    def get_weights_data(self, layer_name, var_name):
        data = self._find_data(layer_name, var_name)
        if data:
            return data[()]
        else:
            return None


class KerasNestedFileReader(KerasFileReader):
    def __init__(self, data_reader, nested_path):
        super().__init__(data_reader.config)
        self.nested_path = nested_path

    def _find_data(self, layer_name, var_name):
        def h5_visitor_func(name):
            if var_name in name:
                return name

        layer_path = f'model_weights/{self.nested_path}/{layer_name}'

        data_path = self.h5file[layer_path].visit(h5_visitor_func)
        if data_path:
            return self.h5file[f'/{layer_path}/{data_path}']
        else:
            return None


class KerasModelReader(KerasReader):
    def __init__(self, keras_model):
        self.model = keras_model

    def get_weights_data(self, layer_name, var_name):
        layer = self.model.get_layer(layer_name)
        for i, w in enumerate(layer.weights):
            if var_name in w.name:
                try:
                    return w.numpy()  # TF 2.x
                except Exception:
                    return layer.get_weights()[i]  # TF 1.x

        return None


def get_weights_data(data_reader, layer_name, var_name):
    if not isinstance(var_name, (list, tuple)):
        var_name = [var_name]

    data = [data_reader.get_weights_data(layer_name, var) for var in var_name]

    if len(data) == 1:
        return data[0]
    else:
        return (*data,)


layer_handlers = {}


def register_keras_layer_handler(layer_cname, handler_func):
    """Register a handler function for the given layer class name.

    The handler function should have the following signature:
        parse_func(keras_layer, input_names, input_shapes, data_reader, config):

    Args:
        layer_cname (str): The name of Keras layer (the 'class_name' property in the layer's config)
        handler_func (callable): The handler function

    Raises:
        Exception: If the layer class has already been registered.
    """
    if layer_cname in layer_handlers:
        raise Exception(f'Layer {layer_cname} already registered')
    else:
        layer_handlers[layer_cname] = handler_func


def get_supported_keras_layers():
    """Returns the list of Keras layers that the converter can parse.

    The returned list contains all Keras layers that can be parsed into the hls4ml internal representation. Support for
    computation of these layers may vary across hls4ml backends and conversion configuration.

    Returns:
        list: The names of supported Keras layers.
    """
    return list(layer_handlers.keys())


def keras_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function

    return decorator


def parse_default_keras_layer(keras_layer, input_names):
    layer = {}

    # Extract name for finding weights and biases
    layer['name'] = keras_layer['config']['name']
    layer['class_name'] = keras_layer['class_name']
    if input_names is not None:
        layer['inputs'] = input_names

    layer['data_format'] = keras_layer['config'].get('data_format', 'channels_last')

    if 'activation' in keras_layer['config']:
        layer['activation'] = keras_layer['config']['activation']
    if 'epsilon' in keras_layer['config']:
        layer['epsilon'] = keras_layer['config']['epsilon']
    if 'use_bias' in keras_layer['config']:
        layer['use_bias'] = keras_layer['config']['use_bias']

    return layer


def get_model_arch(config):
    if 'KerasModel' in config:
        # Model instance passed in config from API
        keras_model = config['KerasModel']
        if isinstance(keras_model, str):
            import keras

            keras_model = keras.models.load_model(keras_model)
        model_arch = json.loads(keras_model.to_json())
        reader = KerasModelReader(keras_model)
    elif 'KerasJson' in config:
        # Extract model architecture from json
        with open(config['KerasJson']) as json_file:
            model_arch = json.load(json_file)
        reader = KerasFileReader(config)
    elif 'KerasH5' in config:
        # Model arch and weights are in H5 file (from model.save() function)
        with h5py.File(config['KerasH5'], mode='r') as h5file:
            # Load the configuration from h5 using json's decode
            model_arch = h5file.attrs.get('model_config')
            if model_arch is None:
                raise ValueError('No model found in config file.')
            else:
                # model_arch is string by default since h5py 3.0.0, keeping this condition for compatibility.
                if isinstance(model_arch, bytes):
                    model_arch = model_arch.decode('utf-8')
                model_arch = json.loads(model_arch)
        reader = KerasFileReader(config)
    else:
        raise ValueError('No model found in config file.')

    return model_arch, reader


def parse_keras_model(model_arch, reader):
    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    # Define layers to skip for conversion to HLS
    skip_layers = ['Dropout']
    # Activation layers
    activation_layers = [
        'Activation',
        'LeakyReLU',
        'ThresholdedReLU',
        'ELU',
        'PReLU',
        'Softmax',
        'TernaryTanh',
        'HardActivation',
        'UnaryLUT',
        'HGQ>UnaryLUT',
    ]
    # Recurrent layers
    recurrent_layers = ['SimpleRNN', 'LSTM', 'GRU']
    # All supported layers
    supported_layers = get_supported_keras_layers() + skip_layers

    # Map inputs of skipped and split (activation) layers
    inputs_map = {}

    # Loop through layers
    layer_counter = 0

    input_layers = None
    output_layers = None

    layer_config = None
    if model_arch['class_name'] == 'Sequential':
        print('Interpreting Sequential')
        layer_config = model_arch['config']
        if 'layers' in layer_config:  # Newer Keras versions have 'layers' in 'config' key
            layer_config = layer_config['layers']
        # Sequential doesn't have InputLayer in TF < 2.3 (Keras 2.4.0)
        if layer_config[0]['class_name'] != 'InputLayer':
            input_layer = {}
            input_layer['name'] = 'input1'
            input_layer['class_name'] = 'InputLayer'
            input_layer['input_shape'] = layer_config[0]['config']['batch_input_shape'][1:]
            layer_list.append(input_layer)
            print('Input shape:', input_layer['input_shape'])
    elif model_arch['class_name'] in ['Model', 'Functional']:  # TF >= 2.3 calls it 'Functional' API
        print('Interpreting Model')
        layer_config = model_arch['config']['layers']
        input_layers = [inp[0] for inp in model_arch['config']['input_layers']]
        output_layers = [out[0] for out in model_arch['config']['output_layers']]

    # Get input shape and check for unsupported layer type
    for keras_layer in layer_config:
        if keras_layer['class_name'] not in supported_layers:
            raise Exception('ERROR: Unsupported layer type: {}'.format(keras_layer['class_name']))

    output_shapes = {}
    output_shape = None

    print('Topology:')
    for keras_layer in layer_config:
        if 'batch_input_shape' in keras_layer['config']:
            if 'inbound_nodes' in keras_layer and len(keras_layer['inbound_nodes']) > 0:
                input_shapes = [output_shapes[inbound_node[0]] for inbound_node in keras_layer['inbound_nodes'][0]]
            else:
                input_shapes = [keras_layer['config']['batch_input_shape']]
        else:
            if 'inbound_nodes' in keras_layer:
                input_shapes = [output_shapes[inbound_node[0]] for inbound_node in keras_layer['inbound_nodes'][0]]
            else:
                # Sequential model, so output_shape from the previous layer is still valid
                input_shapes = [output_shape]

        keras_class = keras_layer['class_name']

        if keras_class in skip_layers:
            if 'inbound_nodes' in keras_layer:
                name = keras_layer['config']['name']
                # Currently supported skipped layers have only one input
                parent_input = keras_layer['inbound_nodes'][0][0][0]
                # Skipped layers can follow each other (e.g., Dropout -> Flatten)
                inputs_map[name] = inputs_map.get(parent_input, parent_input)

            output_shapes[keras_layer['config']['name']] = input_shapes[0]

            continue

        if keras_class in supported_layers:
            layer_counter = layer_counter + 1

        # Extract inbound nodes
        if 'inbound_nodes' in keras_layer and len(keras_layer['inbound_nodes']) > 0:
            input_names = [inputs_map.get(inp[0], inp[0]) for inp in keras_layer['inbound_nodes'][0]]
        else:
            input_names = None

        layer, output_shape = layer_handlers[keras_class](keras_layer, input_names, input_shapes, reader)

        print(
            'Layer name: {}, layer type: {}, input shapes: {}, output shape: {}'.format(
                layer['name'], layer['class_name'], input_shapes, output_shape
            )
        )
        layer_list.append(layer)
        if 'activation' in layer and layer['class_name'] not in activation_layers + recurrent_layers:  # + qkeras_layers:
            act_layer = {}
            act_details = layer['activation']
            # Workaround for QKeras activations passed as an argument
            if isinstance(act_details, dict):
                act_layer['class_name'] = 'QActivation'
                act_layer['config'] = {
                    'name': layer['name'] + '_' + act_details['class_name'],
                    'activation': act_details,
                }
            else:
                act_layer['class_name'] = 'Activation'
                act_layer['config'] = {'name': layer['name'] + '_' + act_details, 'activation': act_details}
            act_layer, output_shape = layer_handlers[act_layer['class_name']](act_layer, None, [output_shape], reader)
            inputs_map[layer['name']] = act_layer['name']
            if output_layers is not None and layer['name'] in output_layers:
                output_layers = [act_layer['name'] if name == layer['name'] else name for name in output_layers]
            output_shapes[act_layer['name']] = output_shape
            layer_list.append(act_layer)

        assert output_shape is not None

        output_shapes[layer['name']] = output_shape

    return layer_list, input_layers, output_layers, output_shapes


def keras_to_hls(config):
    model_arch, reader = get_model_arch(config)
    layer_list, input_layers, output_layers, _ = parse_keras_model(model_arch, reader)
    print('Creating HLS model')
    hls_model = ModelGraph(config, layer_list, input_layers, output_layers)
    return hls_model
