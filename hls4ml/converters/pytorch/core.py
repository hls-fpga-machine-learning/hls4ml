import numpy as np

from hls4ml.converters.pytorch_to_hls import convert_uaq_to_apfixed, pytorch_handler
from hls4ml.model.quantizers import BrevitasQuantizer
from hls4ml.model.types import FixedPrecisionType


@pytorch_handler('Constant')
def parse_constant_layer(operation, layer_name, node):
    assert 'Constant' in operation

    layer = {}
    layer['inputs'] = []

    layer['class_name'] = 'Constant'
    layer['name'] = layer_name

    constant = np.array(node._args)
    layer['value'] = constant
    output_shape = constant.shape

    return layer, output_shape


@pytorch_handler('Linear', 'QuantLinear')
def parse_linear_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'Linear' in operation

    layer = {}

    layer['class_name'] = 'Dense'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    layer['weight_data'] = class_object.weight.data.numpy()
    if class_object.bias is not None:
        layer['bias_data'] = class_object.bias.data.numpy()
    else:
        layer['bias_data'] = None

    if "Quant" in operation:
        if class_object.weight_quant.is_quant_enabled:
            width = int(class_object.quant_weight().bit_width)
            ap_fixed_params = convert_uaq_to_apfixed(width, float(class_object.quant_weight().scale))
            layer['weight_data'] = class_object.quant_weight().detach().value.numpy()
            layer['weight_quantizer'] = BrevitasQuantizer(
                width, FixedPrecisionType(width=width, integer=int(ap_fixed_params[1]), signed=True)
            )
        else:
            layer['weight_data'] = class_object.weight.data.numpy()

        if class_object.bias_quant.is_quant_enabled:
            width = int(class_object.quant_bias().bit_width)
            ap_fixed_params = convert_uaq_to_apfixed(width, float(class_object.quant_bias().scale))
            layer['bias_data'] = class_object.quant_bias().detach().value.numpy()
            layer['bias_quantizer'] = BrevitasQuantizer(
                width, FixedPrecisionType(width=width, integer=int(ap_fixed_params[1]), signed=True)
            )
        else:
            layer['bias_data'] = class_object.bias.data.numpy()
    else:
        layer['weight_data'] = class_object.weight.data.numpy()
        if class_object.bias is not None:
            layer['bias_data'] = class_object.bias.data.numpy()
        else:
            layer['bias_data'] = None

    if class_object is not None:
        layer['n_in'] = class_object.in_features
        layer['n_out'] = class_object.out_features
    else:
        raise Exception('parsing of torch.nn.functional.linear not supported yet, please use torch.nn.Linear class')

    # Handling whether bias is used or not
    if class_object.bias is None:
        layer['use_bias'] = False
    else:
        layer['use_bias'] = True

    output_shape = input_shapes[0][:]
    output_shape[-1] = layer['n_out']

    return layer, output_shape


activation_layers = [
    'Softmax',
    'ReLU',
    'LeakyReLU',
    'Threshold',
    'ELU',
    'PReLU',
    'Sigmoid',
    'Tanh',
    'QuantReLU',
    'QuantSigmoid',
    'QuantTanh',
]


@pytorch_handler(*activation_layers)
def parse_activation_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    layer = {}

    layer['class_name'] = operation
    layer['activation'] = layer['class_name'].lower()
    layer['name'] = layer_name
    layer['inputs'] = input_names

    if "Quant" in operation:
        layer['class_name'] = operation.split('Quant')[-1]
        layer['activation'] = layer['class_name']
        if class_object.act_quant.is_quant_enabled:
            bit_width = int(class_object.act_quant.bit_width())
            ap_fixed_params = convert_uaq_to_apfixed(bit_width, float(class_object.act_quant.scale()))
            layer['activation_quantizer'] = BrevitasQuantizer(
                bit_width, FixedPrecisionType(width=bit_width, integer=ap_fixed_params[1], signed=False)
            )

    if node.op == 'call_module':
        if layer['class_name'] in ['ReLU', 'Sigmoid', 'Tanh']:
            layer['class_name'] = 'Activation'
        if layer['class_name'] == 'LeakyReLU':
            layer['activ_param'] = class_object.negative_slope
        if layer['class_name'] == 'ELU':
            layer['activ_param'] = class_object.alpha
        if layer['class_name'] == 'PReLU':
            layer['param_data'] = class_object.weight.data.numpy()
        if layer['class_name'] == 'Threshold':
            layer['activ_param'] = class_object.threshold
            layer['class_name'] = 'ThresholdedReLU'
            layer['activation'] = 'ThresholdedReLU'
            if layer['activ_param'] < 0:
                raise Exception('negative threshold values not supported')
        if hasattr(class_object, 'dim'):
            layer['axis'] = class_object.dim
            if layer['class_name'] == 'Softmax' and layer['axis'] is None:
                layer['axis'] = -1
            if 'IOType' in config:
                if layer['class_name'] == 'Softmax' and config['IOType'] == 'io_stream' and layer['axis'] != -1:
                    raise Exception('dim needs to be -1 for io_stream')
    else:
        if layer['class_name'] in ['ReLU', 'Sigmoid', 'Tanh']:
            layer['class_name'] = 'Activation'
        if layer['class_name'] == 'LeakyReLU':
            layer['activ_param'] = node.kwargs['negative_slope']
        if layer['class_name'] == 'ELU':
            layer['activ_param'] = node.kwargs['alpha']
        if layer['class_name'] == 'Threshold':
            layer['activ_param'] = node.args[1]
            if layer['activ_param'] < 0:
                raise Exception('negative threshold values not supported')
            layer['class_name'] = 'ThresholdedReLU'
            layer['activation'] = 'ThresholdedReLU'
        if 'dim' in node.kwargs:
            layer['axis'] = node.kwargs['dim']
            if layer['class_name'] == 'Softmax' and layer['axis'] is None:
                layer['axis'] = -1
            if 'IOType' in config:
                if layer['class_name'] == 'Softmax' and config['IOType'] == 'io_stream' and layer['axis'] != -1:
                    raise Exception('dim needs to be -1 for io_stream')

    output_shape = input_shapes[0]
    return layer, output_shape


batchnorm_layers = ['BatchNorm2d', 'BatchNorm1d', 'Batch_norm']


@pytorch_handler(*batchnorm_layers)
def parse_batchnorm_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'BatchNorm' in operation

    layer = {}

    layer['class_name'] = 'BatchNormalization'
    layer['data_format'] = 'channels_first'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    # batchnorm para
    if node.op == 'call_module':
        layer['epsilon'] = class_object.eps
        layer['use_gamma'] = layer['use_beta'] = class_object.affine

        if layer['use_gamma']:
            layer['gamma_data'] = class_object.weight.data.numpy()
        else:
            layer['gamma_data'] = 1

        if layer['use_beta']:
            layer['beta_data'] = class_object.bias.data.numpy()
        else:
            layer['beta_data'] = 0

        layer['mean_data'] = class_object.running_mean.data.numpy()
        layer['variance_data'] = class_object.running_var.data.numpy()

    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim

    layer['n_in'] = layer['n_out'] = in_size

    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) > 2:
        layer['n_filt'] = input_shapes[0][1]  # Always channel first for Pytorch

    return layer, [shape for shape in input_shapes[0]]
