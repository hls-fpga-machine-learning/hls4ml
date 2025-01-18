from hls4ml.converters.keras.convolution import parse_conv1d_layer, parse_conv2d_layer
from hls4ml.converters.keras.core import parse_batchnorm_layer, parse_dense_layer
from hls4ml.converters.keras.recurrent import parse_rnn_layer
from hls4ml.converters.keras_to_hls import keras_handler, parse_default_keras_layer
from hls4ml.model.quantizers import QKerasBinaryQuantizer, QKerasPO2Quantizer, QKerasQuantizer
from hls4ml.model.types import FixedPrecisionType


def get_quantizer_from_config(keras_layer, quantizer_var):
    quantizer_config = keras_layer['config'].get(f'{quantizer_var}_quantizer', None)
    if quantizer_config is None:
        return None  # No quantizer specified in the layer
    if keras_layer['class_name'] == 'QBatchNormalization':
        return QKerasQuantizer(quantizer_config)
    elif 'binary' in quantizer_config['class_name']:
        return QKerasBinaryQuantizer(quantizer_config, xnor=(quantizer_var == 'kernel'))
    elif quantizer_config['class_name'] == 'quantized_po2':
        return QKerasPO2Quantizer(quantizer_config)
    else:
        return QKerasQuantizer(quantizer_config)


@keras_handler('QDense')
def parse_qdense_layer(keras_layer, input_names, input_shapes, data_reader):
    layer, output_shape = parse_dense_layer(keras_layer, input_names, input_shapes, data_reader)

    layer['weight_quantizer'] = get_quantizer_from_config(keras_layer, 'kernel')
    layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')

    return layer, output_shape


@keras_handler('QConv1D', 'QConv2D')
def parse_qconv_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'QConv' in keras_layer['class_name']

    if '1D' in keras_layer['class_name']:
        layer, output_shape = parse_conv1d_layer(keras_layer, input_names, input_shapes, data_reader)
    elif '2D' in keras_layer['class_name']:
        layer, output_shape = parse_conv2d_layer(keras_layer, input_names, input_shapes, data_reader)

    layer['weight_quantizer'] = get_quantizer_from_config(keras_layer, 'kernel')
    layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')

    return layer, output_shape


@keras_handler('QDepthwiseConv2D')
def parse_qdepthwiseqconv_layer(keras_layer, input_names, input_shapes, data_reader):
    layer, output_shape = parse_conv2d_layer(keras_layer, input_names, input_shapes, data_reader)

    layer['depthwise_quantizer'] = get_quantizer_from_config(keras_layer, 'depthwise')
    layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')

    return layer, output_shape


@keras_handler('QSeparableConv1D', 'QSeparableConv2D')
def parse_qsepconv_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'QSeparableConv' in keras_layer['class_name']

    if '1D' in keras_layer['class_name']:
        layer, output_shape = parse_conv1d_layer(keras_layer, input_names, input_shapes, data_reader)
    elif '2D' in keras_layer['class_name']:
        layer, output_shape = parse_conv2d_layer(keras_layer, input_names, input_shapes, data_reader)

    layer['depthwise_quantizer'] = get_quantizer_from_config(keras_layer, 'depthwise')
    layer['pointwise_quantizer'] = get_quantizer_from_config(keras_layer, 'pointwise')
    layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')

    return layer, output_shape


@keras_handler('QSimpleRNN', 'QLSTM', 'QGRU')
def parse_qrnn_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] in ['QSimpleRNN', 'QLSTM', 'QGRU']

    layer, output_shape = parse_rnn_layer(keras_layer, input_names, input_shapes, data_reader)

    layer['weight_quantizer'] = get_quantizer_from_config(keras_layer, 'kernel')
    layer['recurrent_quantizer'] = get_quantizer_from_config(keras_layer, 'recurrent')
    layer['bias_quantizer'] = get_quantizer_from_config(keras_layer, 'bias')

    return layer, output_shape


@keras_handler('QActivation')
def parse_qactivation_layer(keras_layer, input_names, input_shapes, data_reader):
    from qkeras.quantizers import get_quantizer

    assert keras_layer['class_name'] == 'QActivation'
    supported_activations = [
        'quantized_relu',
        'quantized_tanh',
        'binary_tanh',
        'ternary_tanh',
        'quantized_sigmoid',
        'quantized_bits',
        'binary',
        'ternary',
    ]

    layer = parse_default_keras_layer(keras_layer, input_names)

    activation_config = keras_layer['config']['activation']
    quantizer_obj = get_quantizer(activation_config)
    activation_config = {}
    # some activations are classes
    if hasattr(quantizer_obj, 'get_config'):
        activation_config['class_name'] = quantizer_obj.__class__.__name__
        if activation_config['class_name'] == 'ternary' or activation_config['class_name'] == 'binary':
            activation_config['class_name'] += '_tanh'
        activation_config['config'] = quantizer_obj.get_config()
    # some activation quantizers are just functions with no config
    else:
        activation_config['config'] = {}
        if 'binary' in quantizer_obj.__name__:
            activation_config['class_name'] = 'binary_tanh'
            activation_config['config']['bits'] = 1
            activation_config['config']['integer'] = 1
        elif 'ternary' in quantizer_obj.__name__:
            activation_config['class_name'] = 'ternary_tanh'
            activation_config['config']['bits'] = 2
            activation_config['config']['integer'] = 2
        else:
            activation_config['class_name'] = 'unknown'

    if activation_config['class_name'] not in supported_activations:
        raise Exception('Unsupported QKeras activation: {}'.format(activation_config['class_name']))

    if activation_config['class_name'] == 'quantized_bits':
        activation_config['class_name'] = 'linear'

    if activation_config['class_name'] == 'ternary_tanh':
        layer['class_name'] = 'TernaryTanh'
        layer['threshold'] = activation_config.get('config', {}).get('threshold', 0.33)
        if layer['threshold'] is None:
            layer['threshold'] = 0.33  # the default ternary tanh threshold for QKeras
        layer['activation'] = 'ternary_tanh'
    elif (
        activation_config['class_name'] == 'quantized_sigmoid'
        and not activation_config['config'].get('use_real_sigmoid', False)
    ) or (
        activation_config['class_name'] == 'quantized_tanh' and not activation_config['config'].get('use_real_tanh', False)
    ):
        layer['class_name'] = 'HardActivation'
        layer['slope'] = 0.5  # the default values in QKeras
        layer['shift'] = 0.5
        # Quartus seems to have trouble if the width is 1.
        layer['slope_prec'] = FixedPrecisionType(width=2, integer=0, signed=False)
        layer['shift_prec'] = FixedPrecisionType(width=2, integer=0, signed=False)
        layer['activation'] = activation_config['class_name'].replace('quantized_', 'hard_')
    elif activation_config['class_name'] == 'quantized_relu' and activation_config['config']['negative_slope'] != 0:
        layer['class_name'] = 'LeakyReLU'
        layer['activation'] = activation_config['class_name'].replace('quantized_', 'leaky_')
        layer['activ_param'] = activation_config['config']['negative_slope']
    else:
        layer['class_name'] = 'Activation'
        layer['activation'] = activation_config['class_name'].replace('quantized_', '')

    layer['activation_quantizer'] = activation_config
    return layer, [shape for shape in input_shapes[0]]


@keras_handler('QBatchNormalization')
def parse_qbatchnorm_layer(keras_layer, input_names, input_shapes, data_reader):
    layer, output_shape = parse_batchnorm_layer(keras_layer, input_names, input_shapes, data_reader)

    layer['mean_quantizer'] = get_quantizer_from_config(keras_layer, 'mean')
    layer['variance_quantizer'] = get_quantizer_from_config(keras_layer, 'variance')
    layer['beta_quantizer'] = get_quantizer_from_config(keras_layer, 'beta')
    layer['gamma_quantizer'] = get_quantizer_from_config(keras_layer, 'gamma')

    return layer, output_shape


@keras_handler('QConv2DBatchnorm')
def parse_qconv2dbatchnorm_layer(keras_layer, input_names, input_shapes, data_reader):
    intermediate_shape = list()
    conv_layer, shape_qconv = parse_qconv_layer(keras_layer, input_names, input_shapes, data_reader)
    intermediate_shape.append(shape_qconv)
    temp_shape = intermediate_shape
    batch_layer, out_shape = parse_batchnorm_layer(keras_layer, input_names, temp_shape, data_reader)
    return {**conv_layer, **batch_layer}, out_shape
