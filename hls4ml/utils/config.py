from __future__ import print_function
import numpy as np
import h5py
import json
import math
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

def config_from_keras_model(model, granularity='model', default_precision='ap_fixed<16,6>', default_reuse_factor=1):
    """Create an HLS conversion config given the Keras model.

    This function serves as the initial step in creating the custom conversion configuration.
    Users are advised to inspect the returned object to tweak the conversion configuration.
    The return object can be passed as `hls_config` parameter to `convert_from_keras_model`.

    Args:
        model: Keras model
        granularity (str, optional): Granularity of the created config. Defaults to 'model'.
            Can be set to 'model', 'type' and 'layer'.

            Granularity can be used to generate a more verbose config that can be fine-tuned.
            The default granulrity ('model') will generate config keys that apply to the whole
            model, so changes to the keys will affect the entire model. 'type' granularity will
            generate config keys that affect all layers of a given type, while the 'name' granularity
            will generate config keys for every layer separately, allowing for highly specific
            configuration tweaks.
        default_precision (str, optional): Default precision to use. Defaults to 'ap_fixed<16,6>'.
        default_reuse_factor (int, optional): Default reuse factor. Defaults to 1.

    Raises:
        Exception: If Keras model has layers not supported by hls4ml.

    Returns:
        [dict]: The created config.
    """
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
    qkeras_layers = ['QDense', 'QActivation', 'QConv1D', 'QConv2D', 'QBatchNormalization']
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
            layer['precision'] = {}
            for qname, qclass in layer['config'].items():
                if 'quantizer' in qname.lower():
                    pname = qname.split('_quantizer')[0]
                    if pname == 'kernel': pname = 'weight'
                    if qclass is not None:
                        precision = _get_precision_from_quantizer(qclass)
                        layer['precision'][pname] = precision
                elif qname == 'activation' and layer['class_name'] == 'QActivation':
                    precision = _get_precision_from_quantizer(qclass)
                    layer['precision']['result'] = precision

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
        if layer['class_name'] in dense_layers + conv_layers:
            layer_config['Precision'] = {}
            layer_config['Precision']['weight'] = default_precision
            layer_config['Precision']['bias'] = default_precision
            layer_config['Precision']['result'] = default_precision
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
               layer_config['exp_table_t'] = 'ap_fixed<18,8,AP_RND,AP_SAT>'
               layer_config['inv_table_t'] = 'ap_fixed<18,8,AP_RND,AP_SAT>'
            else:
                layer_config['table_t'] = 'ap_fixed<18,8>'
        
        elif layer['class_name'] in norm_layers:
            layer_config['Precision'] = {}
            layer_config['Precision']['scale'] = default_precision
            layer_config['Precision']['bias'] = default_precision
            layer_config['ReuseFactor'] = default_reuse_factor
        
        elif layer['class_name'] in qkeras_layers:
            if 'precision' in layer:
                layer_config['Precision'] = {}
                for name, precision in layer['precision'].items():
                    layer_config['Precision'][name] = precision
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

    return config


def config_from_pytorch_model(model, granularity='model', default_precision='ap_fixed<16,6>', default_reuse_factor=1):
    """Generate configuration dictionary from a Pytorch model.
    
    Parameters
    ----------
    model : Pytorch model object.
        Model to be converted to hls model object.
    granularity : string, optional
        How granular you want the configuration to be.
    default_precision : string, optional
        Defines the precsion of your inputs, outputs, weights and biases.
        It is denoted by ap_fixed<X,Y>, where Y is the number of bits representing 
        the signed number above the binary point (i.e. the integer part),
        and X is the total number of bits. Additionally, integers in fixed precision 
        data type (ap_int<N>, where N is a bit-size from 1 to 1024) can also be used. 
    default_reuse_factor : int, optional
        Reuse factor for hls model
        
    Returns
    -------
    config : dict
        configuration dictionary to be used in Pytorch converter.
        
    See Also
    --------
    hls4ml.config_from_keras_model, hls4ml.convert_from_onnx_model
    
    Examples
    --------
    >>> import hls4ml
    >>> config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    >>> hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)

    """
    
    config = {}

    model_config = {}
    model_config['Precision'] = default_precision
    model_config['ReuseFactor'] = default_reuse_factor
    model_config['Strategy'] = 'Latency'

    config['Model'] = model_config
    
    return config


def config_from_onnx_model(model, granularity='model', default_precision='ap_fixed<16,6>', default_reuse_factor=1):
    """Generate configuration dictionary from an ONNX model.
    
    Parameters
    ----------
    model : ONNX model object.
        Model to be converted to hls model object.
    granularity : string, optional
        How granular you want the configuration to be.
    default_precision : string, optional
        Defines the precsion of your inputs, outputs, weights and biases.
        It is denoted by ap_fixed<X,Y>, where Y is the number of bits representing 
        the signed number above the binary point (i.e. the integer part),
        and X is the total number of bits. Additionally, integers in fixed precision 
        data type (ap_int<N>, where N is a bit-size from 1 to 1024) can also be used. 
    default_reuse_factor : int, optional
        Reuse factor for hls model
        
    Returns
    -------
    config : dict
        configuration dictionary to be used in ONNX converter.
        
    See Also
    --------
    hls4ml.config_from_keras_model, hls4ml.convert_from_pytorch_model
    
    Examples
    --------
    >>> import hls4ml
    >>> config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    >>> hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)

    """
    
    config = {}

    model_config = {}
    model_config['Precision'] = default_precision
    model_config['ReuseFactor'] = default_reuse_factor
    model_config['Strategy'] = 'Latency'

    config['Model'] = model_config
    
    return config