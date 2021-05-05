from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.hls_layers import BatchNormalization
from hls4ml.model.hls_model import IntegerPrecisionType, FixedPrecisionType, ExponentPrecisionType, register_layer
from hls4ml.templates import templates
import tensorflow as tf
import numpy as np
from qkeras import get_quantizer

class QKerasPO2Quantizer(object):
    def __init__(self, config):
        self.bits = config['config']['bits']
        self.quantizer_fn = get_quantizer(config)
        self.hls_type = ExponentPrecisionType(width=self.bits, signed=True)

    def __call__(self, data):
        '''
        Weights are rounded to nearest power of 2
        '''
        x = tf.convert_to_tensor(data)
        y = self.quantizer_fn(x)
        if hasattr(y, 'numpy'):
            y = y.numpy()
        return y

class OutputRoundingSaturationMode(OptimizerPass):
    '''
    Set the Rounding and Saturation mode of the output (and accumulator, if applicable)
    of the layers specific in layer list.
    The layer list is empty by default.
    To specify which layer to apply this pass to, perform e.g.:
    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Dense', 'Activation', 'BatchNormalization']
    The Rounding and Saturation modes are 'None' by default (so use the compiler defaults)
    To set which mode to use:
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND_CONV'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    '''

    layers = [] 
    rounding_mode = None 
    saturation_mode = None 
    saturation_bits = None

    def match(self, node):
        layer_match = node.__class__.__name__ in self.layers or node.name in self.layers
        t = str(node.get_output_variable().type.precision)
        # check that the type doesn't already contain the rounding mode
        rs_match = False
        if self.rounding_mode is not None:
            rs_match = rs_match or not (self.rounding_mode in t)
        if self.saturation_mode is not None:
            rs_match = rs_match or not (self.saturation_mode in t)
        return layer_match and rs_match

    def transform(self, model, node):
        oldtype = node.get_output_variable().type.precision
        if isinstance(oldtype, IntegerPrecisionType):
            newtype = IntegerPrecisionType(oldtype.width, oldtype.signed)
        elif isinstance(oldtype, FixedPrecisionType):
            newtype = FixedPrecisionType(oldtype.width, oldtype.integer, oldtype.signed, self.rounding_mode, self.saturation_mode, self.saturation_bits)
        else: # in case the precision is a string
            newtype = self.precision_string_modify(oldtype)
        node.get_output_variable().type.precision = newtype
        if node.get_attr('accum_t') is not None:
            node.set_attr('accum_t', newtype)
        return False

    def precision_string_modify(self, pstr):
        # For when the type is a string not an Type
        mode = ''
        if self.rounding_mode is not None:
            mode += ',' + self.rounding_mode
        if self.saturation_mode is not None:
            mode += ',' + self.saturation_mode
        if self.saturation_bits is not None:
            mode += ',' + str(self.saturation_bits)
        mode += '>'
        pstr = pstr.replace('>', mode)
        return pstr

class ApplyAlpha(BatchNormalization):
    ''' A custom layer to scale the output of a QDense layer which used 'alpha != 1'
        Inference computation uses BatchNormalization methods'''

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)

    def add_weights(self, scale, quantizer=None):
        self.add_weights_variable(name='scale', var_name='s{index}', data=scale, quantizer=quantizer)

    def add_bias(self, bias, quantizer=None):
        self.add_weights_variable(name='bias', var_name='b{index}', data=bias, quantizer=quantizer)

# register the layer and its templates
register_layer('ApplyAlpha', ApplyAlpha)
# TODO ideally: for backend in backends
temps = templates.get_backend('Vivado')
temps.register_templates('ApplyAlpha', temps.get_function_template('BatchNormalization'), temps.get_config_template('BatchNormalization'), temps.get_include_list('BatchNormalization'))

class QKerasFactorizeAlpha(OptimizerPass):
    '''OptimizerPass for extracting alpha "scale" from QKeras quantized layer.
       The weights of the Q{Dense, Conv} layer are scaled to the common data type,
       and an 'ApplyAlpha' layer is inserted to reapply the scale.
    '''
    def match(self, node):
        q_layer = node.__class__.__name__ in ["Dense", "Conv1D", "Conv2D"]
        has_w_quant = node.get_attr('weight_quantizer') is not None 
        has_b_quant = node.get_attr('bias_quantizer') is not None
        has_w_alpha, has_b_alpha = False, False
        if has_w_quant:
            if hasattr(node.get_attr('weight_quantizer'), 'alpha'):
                w_alpha = node.get_attr('weight_quantizer').alpha
                has_w_alpha = w_alpha != 1 and w_alpha is not None
            else:
                has_w_alpha = False
        if has_b_quant:
            if hasattr(node.get_attr('bias_quantizer'), 'alpha'):
                b_alpha = node.get_attr('bias_quantizer').alpha
                has_b_alpha = b_alpha != 1 and b_alpha is not None
            else:
                has_b_alpha = False
        is_match = q_layer and ((has_w_quant and has_w_alpha) or (has_b_quant and has_b_alpha))
        return is_match

    def transform(self, model, node):
        # The quantizer has to be applied to set the scale attribute
        # This must be applied to the _unquantized_ weights to obtain the correct scale
        quantizer = node.weights['weight'].quantizer.quantizer_fn # get QKeras quantizer
        weights = node.weights['weight'].data_unquantized # get weights
        qweights = quantizer(tf.convert_to_tensor(weights))
        if isinstance(quantizer.scale, (int, float)):
            scale = np.ones(shape=node.get_output_variable().shape) * quantizer.scale
        else:
            scale = quantizer.scale.numpy()
        unscale = 1. / scale

        new_weights = unscale * qweights # use the quantized weights for safety


        qcfg = quantizer.get_config()
        alpha = qcfg['alpha']
        # Set the alpha to 1 to avoid hitting this pass again
        qcfg['alpha'] = 1
        node.weights['weight'].quantizer.quantizer_fn = quantizer.from_config(qcfg)

        # update the weights also applying the hls4ml quantizer
        # this is only needed for the binary layers which encode -1 as 0
        node.weights['weight'].data = node.weights['weight'].quantizer(new_weights.numpy())

        # Move the biases from the Dense layer to the ApplyAlpha layer
        bias = node.weights['bias'].data
        bias_quantizer = None
        if hasattr(node.weights['bias'], 'quantizer'):
            bias_quantizer = node.weights['bias'].quantizer
        node.weights['bias'].data = np.zeros(bias.shape)

        has_w_quant = node.get_attr('weight_quantizer') is not None 
        has_b_quant = node.get_attr('bias_quantizer') is not None
        if has_w_quant: 
            node.attributes['weight_quantizer'].alpha = 1
        if has_b_quant:
            node.attributes['bias_quantizer'].alpha = 1

        # insert a Batch Normalization layer to apply the alpha scale
        if alpha == 'auto_po2':
            scale_bits = np.abs(np.log2(scale)).max().astype('int') + 1
            scale_t = ExponentPrecisionType(width=scale_bits, signed=True)
            scale_q = QKerasPO2Quantizer({'class_name' : 'quantized_po2', 'config': {'bits': scale_bits}})
        else:
            scale_t = FixedPrecisionType() # TODO: automate this
            scale_q = None

        attrs = {
            'name' : node.get_attr('name') + '_alpha',
            'class_name' : 'Alpha',
            'inputs' : node.outputs,
            'n_in' : node.get_attr('n_out'),
            'n_filt' : node.get_attr('n_filt', -1),
            'reuse_factor' : node.get_attr('reuse_factor'),
            'bias_t' : node.weights['bias'].type, 
            'scale_t' : scale_t,
            'Trace' : node.get_attr('Trace', False) 
        }
        alpha_layer = model.make_node('ApplyAlpha', node.name + '_alpha', attrs, node.outputs)

        alpha_layer.add_weights(scale, quantizer=scale_q)
        alpha_layer.add_bias(bias, quantizer=bias_quantizer)
        model.insert_node(alpha_layer)
        return True

class FuseConsecutiveBatchNormalization(OptimizerPass):
    '''OptimizerPass to merge consecutive BatchNormalization layers.
       These may exist in a model after QKerasFactorizeAlpha layer.
       Scale and Bias of each layer are combined into scale and bias of a single layer.
    '''

    def match(self, node):
        return isinstance(node, BatchNormalization) and \
               isinstance(node.get_input_node(), BatchNormalization)

    def transform(self, model, node):
        bn0 = node.get_input_node()
        bn1 = node

        s0 = bn0.weights['scale'].data
        b0 = bn0.weights['bias'].data
        s1 = bn1.weights['scale'].data
        b1 = bn1.weights['bias'].data

        s2 = s0 * s1
        b2 = s1 * b0 + b1

        bn0.weights['scale'].data = s2
        bn0.weights['bias'].data = b2

        model.remove_node(node, rewire=True)
        return True

class ExtractTernaryThreshold(OptimizerPass):
    ''' The input value (threshold) at which the output of a a ternary activation
    changes is configurable. This pass extracts that threshold point, inserting
    a BatchNormalization layer to execute the scaling. That BatchNormalization
    layer is then expected to be fused into a BatchNormalizationQuantizedTanh
    layer configured with the correct threshold.
    '''

    def match(self, node):
        return node.__class__.__name__ == 'TernaryTanh' and node.get_attr('threshold', None) != 0.5

    def transform(self, model, node):
        shape = node.get_input_variable().shape
        scale = np.full(shape, 0.5 / node.get_attr('threshold', 0.5))
        bias = np.zeros_like(scale)
        node.set_attr('threshold', 0.5)

        attrs = {
            'name' : node.get_attr('name') + '_scale',
            'class_name' : 'Alpha',
            'inputs' : node.get_input_node().outputs,
            'outputs' : node.inputs,
            'n_filt' : node.get_attr('n_filt', -1),
            'reuse_factor' : node.get_attr('reuse_factor'),
            # These should just be placeholders
            'bias_t' : IntegerPrecisionType(1),
            'scale_t' : FixedPrecisionType(16,6),
            'Trace' : node.get_attr('Trace', False)
        }

        layer = model.make_node('ApplyAlpha', node.name + '_scale', attrs, node.inputs.copy())
        layer.add_weights(scale)
        layer.add_bias(bias)
        model.insert_node(layer)
        return True

