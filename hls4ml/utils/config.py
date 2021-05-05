from __future__ import print_function
import numpy as np
import h5py
import json
import math
import tensorflow.keras as keras
from hls4ml.model.profiling import activations_keras, weights_keras, array_to_summary
from hls4ml.model.hls_layers import layer_map, FixedPrecisionType, IntegerPrecisionType
from hls4ml.templates import VivadoBackend
from collections import OrderedDict


def create_vivado_config(output_dir='my-hls-test', project_name='myproject',
    fpga_part='xcku115-flvb2104-2-i', clock_period=5, io_type='io_parallel'):
    
    config = {}
    
    config['OutputDir'] = output_dir
    config['ProjectName'] = project_name
    config['XilinxPart'] = fpga_part
    config['ClockPeriod'] = clock_period
    config['Backend'] = 'Vivado'
    config['IOType'] = io_type
    config['HLSConfig'] = {}

    return config

def _get_precision_from_quantizer(quantizer):
    import qkeras
    if isinstance(quantizer, str):
        quantizer_obj = qkeras.get_quantizer(quantizer)
        quantizer = {}
        # Some activations are classes with get_config method
        if hasattr(quantizer_obj, 'get_config'):
            quantizer['class_name'] = quantizer_obj.__class__.__name__
            quantizer['config'] = quantizer_obj.get_config()
        # Some activations are just functions
        else: 
            quantizer['class_name'] = quantizer_obj.__name__

    supported_quantizers = ['quantized_bits', 'quantized_relu', 'quantized_tanh', 'quantized_po2', 'quantized_relu_po2']
    if quantizer['class_name'] in supported_quantizers:
        bits = int(quantizer['config']['bits']) + 1
        # if integer isn't specified, it should be the same as bits
        integer = int(quantizer['config'].get('integer', bits-1)) + 1
        
    elif quantizer['class_name'] in ['binary', 'stochastic_binary', 'binary_tanh']:
        bits = 2
        integer = 2
    
    elif quantizer['class_name'] in ['ternary', 'stochastic_ternary', 'ternary_tanh']:
        bits = 2
        integer = 2
    else:
        raise Exception('ERROR: Unsupported quantizer: {}'.format(quantizer['class_name']))

    decimal = bits - integer

    if decimal > 0:
        return 'ap_fixed<{},{}>'.format(bits, integer)
    else:
        return 'ap_int<{}>'.format(bits)


def set_accum_from_keras_model(config, model):
    """Adjust accum_t of relevant layers in a given HLSModel configuration based on both a Keras model and data types
    set in the configuration.

    The function aims for setting accum_t in applicable layers in a way that no overflow is possible during machine
    learning inference and the minimum non-zero possible value of an accumulator during calculations is covered. The
    maximum bit width (excluding the sign bit) of a resultant value of accum_t is 64.

    Similarly to set_data_types_from_keras_model(), set_accum_from_keras_model() works in a heuristic way. Therefore,
    the optimal result is not guaranteed and post-tuning may be required in order to achieve the best outcome.

    Contrary to set_data_types_from_keras_model(), set_accum_from_keras_model() does not use profiling information.
    Instead, for each applicable layer, it uses data types set in the HLSModel config along with information about the
    layer shape.

    The function supports the following layers only:
    * Dense + its subclasses
    * Conv1D, Conv2D + their subclasses
    * AveragePooling1D, AveragePooling2D + their subclasses

    If your model contains other layers with accum_t, consider using set_data_types_from_keras_model(). QKeras
    equivalents of layers from the above list should count as their subclasses and therefore be supported by this
    function.

    Args:
        config (dict): HLSModel configuration dictionary to be updated. Its granularity must be 'name'.
        model: Keras model to be used for adjusting accum_t in relevant layers.

    Returns:
        None. The function makes changes directly to the supplied config.
    """
    if 'LayerName' not in config:
        raise RuntimeError("The granularity of the supplied config is not 'name'.")

    def find_optimal_a_b(max_val, min_val):
        max_bits = 64

        for a in range(1, max_bits + 1):
            for b in range(0, a + 1):
                max_possible = 2 ** b - 2 ** (b - a)
                min_possible = 2 ** (b - a)

                if min_possible <= min_val and max_possible >= max_val:
                    return a + 1, b + 1

        return None, None

    def get_max_min(hls_type):
        if isinstance(hls_type, FixedPrecisionType) or isinstance(hls_type, IntegerPrecisionType):
            if hls_type.signed:
                subtract = 1
            else:
                subtract = 0

            return 2 ** (hls_type.integer - subtract) - 2 ** (hls_type.integer - hls_type.width), \
                2 ** (hls_type.integer - hls_type.width)
        else:
            raise RuntimeError(f"Unexpected type: {hls_type}")

    for i, layer in enumerate(model.layers):
        name = layer.name

        if name not in config['LayerName']:
            print(f"accum_t profiling: {name} not present in config['LayerName'], ignoring.")
            continue

        if i == 0:
            previous_layer_config = config['LayerName'][name + '_input']
        else:
            index = i - 1
            while index >= 0 and model.layers[index].name not in config['LayerName']:
                index -= 1

            previous_layer_config = config['LayerName'][model.layers[index].name]

        if isinstance(previous_layer_config['Precision'], dict):
            type_i = VivadoBackend.convert_precision_string(None, previous_layer_config['Precision']['result'])
        else:
            type_i = VivadoBackend.convert_precision_string(None, previous_layer_config['Precision'])

        max_i, min_i = get_max_min(type_i)

        if isinstance(layer, keras.layers.AveragePooling1D):
            max_val = layer.pool_size * max_i
            min_val = min_i
        elif isinstance(layer, keras.layers.AveragePooling2D):
            if isinstance(layer.pool_size, int):
                max_val = layer.pool_size * layer.pool_size * max_i
            else:
                max_val = layer.pool_size[0] * layer.pool_size[1] * max_i

            min_val = min_i
        else:
            if isinstance(layer, keras.layers.Dense):
                n = layer.output_shape[1]
            elif isinstance(layer, keras.layers.Conv1D) or isinstance(layer, keras.layers.Conv2D):
                n = layer.input_shape[-1]
            else:
                print(f"accum_t profiling: {name} is not a supported layer or it doesn't have accum_t, ignoring.")
                continue

            type_w = VivadoBackend.convert_precision_string(None, config['LayerName'][name]['Precision']['weight'])
            max_w, min_w = get_max_min(type_w)

            if layer.use_bias:
                type_b = VivadoBackend.convert_precision_string(None, config['LayerName'][name]['Precision']['bias'])
                max_b, min_b = get_max_min(type_b)

                max_val = n * max_w * max_i + max_b
                min_val = min(min_b, min_w * min_i)
            else:
                max_val = n * max_w * max_i
                min_val = min_w * min_i

        a, b = find_optimal_a_b(max_val, min_val)

        if a is None or b is None:
            raise RuntimeError(f"Could not find an optimal accum_t type for {name}.")

        if isinstance(config['LayerName'][name]['Precision'], dict):
            config['LayerName'][name]['Precision']['accum'] = f'ap_fixed<{a},{b}>'
        else:
            config['LayerName'][name]['accum_t'] = f'ap_fixed<{a},{b}>'


def set_data_types_from_keras_model(config, model, max_bits, change_flagged_types=False, test_inputs=None,
                                    best_type_algorithm=None):
    """Adjust data types in a given HLSModel configuration based on a Keras model and test inputs (if supplied).

    The function aims for setting precision of the layers in the configuration to match the distribution of both
    weights in the model and outputs of the model resulting from the test inputs (if supplied). Types flagged
    as inferred from QKeras (i.e. present in QKerasInferred in a layer config) are not adjusted unless
    change_flagged_types is set to True. Moreover, accumulator types are set to the same type as output types
    (when test inputs are provided and where applicable).

    set_data_types_from_keras_model() works in a heuristic way and does not account for optimizations that can be
    subsequently made by hls4ml. Therefore, the optimal result is not guaranteed and it might be necessary to do
    post-tuning of the data types in order to achieve the best outcome. A user-defined algorithm can be passed to
    this function as best_type_algorithm to help obtain precision types closer to the optimal ones.

    Args:
        config (dict): HLSModel configuration dictionary to be updated. Its granularity must be 'name'.
        model: Keras model to be used for adjusting the data types.
        max_bits (int): The maximum bit width (excluding the sign bit) all data types in the config should have.
        change_flagged_types (bool, optional): Whether types flagged as inferred from QKeras should be adjusted. If
            not provided, these types will not be updated.
        test_inputs (array-like, optional): Inputs to be used for producing the distribution of model outputs.
            The type of test_inputs is the same as the type of X in hls4ml.model.profiling.numerical(). If not provided,
            precision of the layer outputs/activations will not be updated.
        best_type_algorithm (function (layer_type, max_val, min_val, median, q1, q3, max_bits) -> (A [int], B [int]),
                             optional):
            Algorithm to be used for determining the best data type ap_fixed<A, B> for a specific layer, given the
            following profiling information: corresponding hls4ml layer class name (layer_type), max value (max_val),
            min value (min_val), median (median), 1st quartile (q1), 3rd quartile (q3) and the max bit width without
            the sign bit (max_bits). Because the bit width doesn't include the sign bit, A may exceed max_bits by 1.

            If the algorithm does not find any suitable data type, it should return (None, None).

            If best_type_algorithm is not provided, the default algorithm will be used for all layers (using only
            max_val and min_val to find a data type both covering max_val and with the minimum absolute distance
            between min_val and the minimum representable number).

    Returns:
        None. The function makes changes directly to the supplied config.
    """
    if 'LayerName' not in config:
        raise RuntimeError("The granularity of the supplied config is not 'name'.")

    def find_optimal_a_b(layer_type, max_val, min_val, median, q1, q3, max_bits):
        a_final = None
        b_final = None

        distance_to_min = math.inf

        for a in range(1, max_bits + 1):
            for b in range(0, a + 1):
                max_possible = 2 ** b - 2 ** (b - a)
                min_possible = 2 ** (b - a)

                if max_possible >= max_val and abs(min_possible - min_val) < distance_to_min:
                    a_final = a
                    b_final = b

                    distance_to_min = abs(min_possible - min_val)

        if a_final is not None and b_final is not None:
            # An extra integer bit must be added for the number sign
            return a_final + 1, b_final + 1
        else:
            return None, None

    def get_precision(layer_dict, key):
        if key is None:
            return layer_dict['Precision']
        else:
            return layer_dict['Precision'][key]

    def set_precision(layer_dict, key, value):
        if key is None:
            layer_dict['Precision'] = value
        else:
            layer_dict['Precision'][key] = value

    def process_precision_in_dict(layer_dict, stats, precision_type=None, data_type=None):
        if precision_type is not None and precision_type not in layer_dict['Precision']:
            return None

        if not change_flagged_types and precision_type is not None and 'QKerasInferred' in layer_dict and \
                precision_type in layer_dict['QKerasInferred']:
            # This data type comes from QKeras and change_flagged_types is False, so don't change the type.
            return None

        if data_type is None:
            min_value = stats['whislo']
            max_value = stats['whishi']
            median = stats['med']
            q1 = stats['q1']
            q3 = stats['q3']

            if 'LayerType' in config['LayerName'][layer_name]:
                layer_type = config['LayerName'][layer_name]['LayerType']
            else:
                raise RuntimeError(f"config['LayerName']['{layer_name}'] doesn't have the LayerType key. "
                                    "Make sure that the config has been made by config_from_keras_model().")

            a, b = best_type_algorithm(layer_type, max_value, min_value, median, q1, q3, max_bits)

            if a is None or b is None:
                if precision_type is not None:
                    raise RuntimeError(f"Could not find an optimal data type for {layer_name} ({precision_type}).")
                else:
                    raise RuntimeError(f"Could not find an optimal data type for {layer_name}.")

            data_type = f'ap_fixed<{a},{b}>'

        set_precision(layer_dict, precision_type, data_type)
        return data_type

    if best_type_algorithm is None:
        best_type_algorithm = find_optimal_a_b

    weight_data = weights_keras(model, fmt='summary', plot='boxplot')

    suffix_map = {
        'w': 'weight',
        's': 'scale',
        'b': 'bias'
    }

    for weight_info in weight_data:
        layer_name = weight_info['layer']

        if layer_name not in config['LayerName']:
            print(f"Weight profiling: {layer_name} not present in config['LayerName'], ignoring.")
            continue

        suffix = weight_info['weight'].split('/')[1]

        if suffix not in suffix_map or suffix_map[suffix] not in config['LayerName'][layer_name]['Precision']:
            continue

        process_precision_in_dict(config['LayerName'][layer_name], weight_info, suffix_map[suffix])

    if test_inputs is not None:
        test_inputs_prof = test_inputs.flatten()
        test_inputs_prof = abs(test_inputs_prof[test_inputs_prof != 0])

        input_data = array_to_summary(test_inputs_prof)
        activation_data = activations_keras(model, test_inputs, fmt='summary', plot='boxplot')

        input_data['weight'] = activation_data[0]['weight'] + '_input'

        for info in [input_data] + activation_data:
            layer_name = info['weight']

            if layer_name not in config['LayerName']:
                print(f"Activation profiling: {layer_name} not present in config['LayerName'], ignoring.")
                continue

            data_type = None

            if isinstance(config['LayerName'][layer_name]['Precision'], dict):
                data_type = process_precision_in_dict(config['LayerName'][layer_name], info,
                                                      'result', data_type)
                data_type = process_precision_in_dict(config['LayerName'][layer_name], info,
                                                      'accum', data_type)
            else:
                data_type = process_precision_in_dict(config['LayerName'][layer_name], info,
                                                      data_type=data_type)

            if data_type is not None and layer_name + '_linear' in config['LayerName']:
                config['LayerName'][layer_name + '_linear']['Precision'] = data_type


def config_from_keras_model(model, granularity='model', default_precision='ap_fixed<16,6>', default_reuse_factor=1,
                            data_type_mode='default', max_bits=15, test_inputs=None):
    """Create an HLS conversion config given the Keras model.

    This function serves as the initial step in creating the custom conversion configuration.
    Users are advised to inspect the returned object to tweak the conversion configuration.
    The return object can be passed as `hls_config` parameter to `convert_from_keras_model`.

    Args:
        model: Keras model
        granularity (str, optional): Granularity of the created config. Defaults to 'model'.
            Can be set to 'model', 'type' and 'layer'.

            Granularity can be used to generate a more verbose config that can be fine-tuned.
            The default granularity ('model') will generate config keys that apply to the whole
            model, so changes to the keys will affect the entire model. 'type' granularity will
            generate config keys that affect all layers of a given type, while the 'name' granularity
            will generate config keys for every layer separately, allowing for highly specific
            configuration tweaks.
        default_precision (str, optional): Default precision to use. Defaults to 'ap_fixed<16,6>'.
        default_reuse_factor (int, optional): Default reuse factor. Defaults to 1.
        data_type_mode (str, optional): Data type inference mode for layers. It may be one of the following:
            * 'default': Set default precision for all layers except for QKeras layers with enough information for type
            inference to be done already in config_from_keras_model(). This option is default if no value for
            data_type_mode is provided.
            * 'auto': Infer data types for all layers (except QKeras layers where applicable) automatically by calling
            set_data_types_from_keras_model() before returning a generated HLS conversion config.
            * 'auto_accum': Same as 'auto', but infer accumulator data types for applicable layers as well by calling
            set_accum_from_keras_model() before returning a generated HLS conversion config and after calling
            set_data_types_from_keras_model(). Note that set_accum_from_keras_model() doesn't use profiling information
            unlike set_data_types_from_keras_model().
            * 'auto_accum_only': Same as 'default', but infer accumulator data types for applicable layers as well by
            calling set_accum_from_keras_model() before returning a generated HLS conversion config. This option is
            not the same as 'auto_accum': it doesn't call set_data_types_from_keras_model() at any point.

            It must be noted that using an automatic mode does not mean that users no longer have to tweak the resultant
            configuration manually regardless of their case.
        max_bits (int, optional): Maximum bit width (excluding the sign bit) to be fed into
            set_data_types_from_keras_model() if data_type_mode is set to either 'auto' or 'auto_accum'.
            See the docstring for set_data_types_from_keras_model() for more details. The default value for this
            argument is 15.
        test_inputs (array-like, optional): Test inputs to be fed into set_data_types_from_keras_model() if
            data_type_mode is set to either 'auto' or 'auto_accum'. See the docstring for
            set_data_types_from_keras_model() for more details. The default value for this argument is None.

    Raises:
        Exception: If Keras model has layers not supported by hls4ml or data_type_mode is invalid.

    Returns:
        [dict]: The created config.
    """
    if data_type_mode not in ['default', 'auto', 'auto_accum', 'auto_accum_only']:
        raise Exception('data_type_mode must be one of "default", "auto", "auto_accum" or "auto_accum_only".')

    if granularity.lower() not in ['model', 'type', 'name']:
        raise Exception('Invalid configuration granularity specified, expected "model", "type" or "name" got "{}"'.format(granularity))

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    if isinstance(model, dict):
        model_arch = model
    else:
        model_arch = json.loads(model.to_json())

    #Define supported laers
    core_layers = ['InputLayer', 'Dropout', 'Flatten', 'Reshape']
    dense_layers = ['Dense', 'BinaryDense', 'TernaryDense']
    conv_layers = ['Conv1D', 'Conv2D', 'BinaryConv2D']
    pooling_layers = ['MaxPooling1D', 'MaxPooling2D', 'GlobalMaxPooling1D', 'GlobalMaxPooling2D', 'AveragePooling1D', 'AveragePooling2D', 'GlobalAveragePooling1D', 'GlobalAveragePooling2D']
    norm_layers = ['BatchNormalization']
    activation_layers = ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'ReLU']
    merge_layers = ['Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate', 'Dot']
    qkeras_layers = ['QDense', 'QActivation', 'QConv1D', 'QConv2D', 'QBatchNormalization', 'QConv2DBatchnorm']
    #Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Flatten']
    #All supported layers
    supported_layers = core_layers + dense_layers + conv_layers + pooling_layers + norm_layers + activation_layers + merge_layers + qkeras_layers + skip_layers

    keras_layer_config = None
    if model_arch['class_name'] == 'Sequential':
        print('Interpreting Sequential')
        keras_layer_config = model_arch['config']
        if 'layers' in keras_layer_config: # Newer Keras versions have 'layers' in 'config' key
            keras_layer_config = keras_layer_config['layers']
        # Sequential doesn't have InputLayer in TF < 2.3 (Keras 2.4.0)
        if keras_layer_config[0]['class_name'] != 'InputLayer':
            input_layer = {}
            input_layer['name'] = 'input1'
            input_layer['class_name'] = 'Input'
            layer_list.append(input_layer)
    elif model_arch['class_name'] in ['Model', 'Functional']:
        print('Interpreting Model')
        keras_layer_config = model_arch['config']['layers']

    print('Topology:')
    for keras_layer in keras_layer_config:
        if keras_layer['class_name'] not in supported_layers:
            raise Exception('ERROR: Unsupported layer type: {}'.format(keras_layer['class_name']))
        if keras_layer['class_name'] in skip_layers:
            continue

        #Dictionary to fill in and append to layer_list
        layer = {}

        #Extract name for finding weights and biases
        layer['name'] = keras_layer['config']['name']
        layer['class_name'] = keras_layer['class_name']
        layer['config'] = keras_layer['config']

        if layer['class_name'] == 'InputLayer':
            layer['class_name'] = 'Input'

        if layer['class_name'] in qkeras_layers:
            def add_to_inferred(name):
                if 'qkeras_inferred' not in layer:
                    layer['qkeras_inferred'] = []

                layer['qkeras_inferred'].append(name)

            layer['precision'] = {}
            for qname, qclass in layer['config'].items():
                if 'quantizer' in qname.lower():
                    pname = qname.split('_quantizer')[0]
                    if pname == 'kernel': pname = 'weight'
                    if qclass is not None:
                        precision = _get_precision_from_quantizer(qclass)
                        layer['precision'][pname] = precision
                        add_to_inferred(pname)
                elif qname == 'activation' and layer['class_name'] == 'QActivation':
                    precision = _get_precision_from_quantizer(qclass)
                    layer['precision']['result'] = precision
                    add_to_inferred('result')

        print('Layer name: {}, layer type: {}'.format(layer['name'], layer['class_name']))
        layer_list.append( layer )
        if 'activation' in layer['config'] and layer['class_name'] not in activation_layers + qkeras_layers:
            act_layer = {}
            act_layer['name'] = layer['name'] + '_' + layer['config']['activation']
            act_layer['class_name'] = 'Activation'
            print('  -> Activation ({}), layer name: {}'.format(layer['config']['activation'], layer['name']))
            layer_list.append(act_layer)


    def make_layer_config(layer):
        layer_config = {}
        layer_type = layer_map[layer['class_name']].__name__

        hls_layers_accum = ['Dense', 'Conv1D', 'Conv2D']

        layer_config['LayerType'] = layer_map[layer['class_name']].__name__

        if 'qkeras_inferred' in layer:
            layer_config['QKerasInferred'] = layer['qkeras_inferred']

        if layer['class_name'] in dense_layers + conv_layers:
            layer_config['Precision'] = {}
            layer_config['Precision']['weight'] = default_precision
            layer_config['Precision']['bias'] = default_precision
            layer_config['Precision']['result'] = default_precision
            layer_config['Precision']['accum'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor

        elif layer['class_name'] in activation_layers:
            layer_config['Precision'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor
            layer_config['table_size'] = 1024
            is_softmax = layer['class_name'] == 'Softmax'
            if 'config' in layer.keys():
                if 'activation' in layer['config'].keys():
                    is_softmax = is_softmax or (layer['config']['activation'] == 'softmax')
            if is_softmax:
               layer_config['LayerType'] = 'Softmax'
               layer_config['exp_table_t'] = 'ap_fixed<18,8,AP_RND,AP_SAT>'
               layer_config['inv_table_t'] = 'ap_fixed<18,8,AP_RND,AP_SAT>'
            else:
                layer_config['table_t'] = 'ap_fixed<18,8>'
        
        elif layer['class_name'] in norm_layers:
            layer_config['Precision'] = {}
            layer_config['Precision']['scale'] = default_precision
            layer_config['Precision']['bias'] = default_precision
            layer_config['Precision']['result'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor
        
        elif layer['class_name'] in qkeras_layers:
            if 'precision' in layer:
                layer_config['Precision'] = {}
                layer_config['Precision']['result'] = default_precision

                for name, precision in layer['precision'].items():
                    layer_config['Precision'][name] = precision

                if layer_type in hls_layers_accum:
                    layer_config['Precision']['accum'] = default_precision
            else:
                print('WARNING: Found no precision information in QKeras layer {} ({})'.format(layer['name'], layer['class_name']))
                layer_config['Precision'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor

        elif layer['class_name'] == 'Input':
            layer_config['Precision'] = {}
            layer_config['Precision']['result'] = default_precision

        else:
            layer_config['Precision'] = default_precision

        return layer_config

    config = {}

    model_config = {}
    model_config['Precision'] = default_precision
    model_config['ReuseFactor'] = default_reuse_factor
    model_config['Strategy'] = 'Latency'
    #model_config['Compression'] = False
    #model_config['Trace'] = False

    config['Model'] = model_config
    
    if granularity.lower() == 'type':
        type_config = {}
        for layer in layer_list:
            if layer['class_name'] in type_config:
                continue
            layer_config = make_layer_config(layer)
            type_config[layer['class_name']] = layer_config
        
        config['LayerType'] = type_config

    elif granularity.lower() == 'name':
        name_config = {}
        for layer in layer_list:
            layer_config = make_layer_config(layer)
            name_config[layer['name']] = layer_config
        
        config['LayerName'] = name_config

    if data_type_mode in ['auto', 'auto_accum']:
        set_data_types_from_keras_model(config, model, max_bits=max_bits, test_inputs=test_inputs)

    if data_type_mode in ['auto_accum', 'auto_accum_only']:
        set_accum_from_keras_model(config, model)

    return config
