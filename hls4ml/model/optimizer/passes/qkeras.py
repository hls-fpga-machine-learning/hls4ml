import numpy as np

from hls4ml.model.layers import ApplyAlpha
from hls4ml.model.optimizer import ConfigurableOptimizerPass, OptimizerPass, register_pass
from hls4ml.model.quantizers import QKerasPO2Quantizer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, NamedType


class OutputRoundingSaturationMode(ConfigurableOptimizerPass):
    '''
    Set the Rounding and Saturation mode of the output (and accumulator, if applicable)
    of the layers specific in layer list.
    The layer list is empty by default.
    To specify which layer to apply this pass to, perform e.g.:
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Dense', 'Activation'])
    The Rounding and Saturation modes are 'None' by default (so use the compiler defaults)
    To set which mode to use:
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(rounding_mode='AP_RND_CONV')
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(saturation_mode='AP_SAT')
    '''

    def __init__(self):
        self.layers = []
        self.rounding_mode = None
        self.saturation_mode = None
        self.saturation_bits = None

    def match(self, node):
        layer_match = node.class_name in self.layers or node.name in self.layers
        t = str(node.get_output_variable().type.precision)
        # check that the type doesn't already contain the rounding mode
        rs_match = False
        if self.rounding_mode is not None:
            rs_match = rs_match or not (self.rounding_mode in t)
        if self.saturation_mode is not None:
            rs_match = rs_match or not (self.saturation_mode in t)
        return layer_match and rs_match

    def transform(self, model, node):
        old_precision = node.get_output_variable().type.precision
        if isinstance(old_precision, IntegerPrecisionType):
            new_precision = IntegerPrecisionType(old_precision.width, old_precision.signed)
        elif isinstance(old_precision, FixedPrecisionType):
            new_precision = FixedPrecisionType(
                old_precision.width,
                old_precision.integer,
                old_precision.signed,
                self.rounding_mode,
                self.saturation_mode,
                self.saturation_bits,
            )
        else:  # in case the precision is a string
            new_precision = self.precision_string_modify(old_precision)

        out_var = node.get_output_variable()
        out_t = NamedType(out_var.type.name, new_precision)
        out_var.type = out_t
        node.attributes['result_t'] = out_t

        if node.get_attr('accum_t') is not None:
            accum_t = NamedType(f'layer{node.index}_accum_t', new_precision)
            node.set_attr('accum_t', accum_t)
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


def register_qkeras():
    # Register the optimization passes
    register_pass('output_rounding_saturation_mode', OutputRoundingSaturationMode)
    register_pass('qkeras_factorize_alpha', QKerasFactorizeAlpha)
    register_pass('extract_ternary_threshold', ExtractTernaryThreshold)


class QKerasFactorizeAlpha(OptimizerPass):
    '''OptimizerPass for extracting alpha "scale" from QKeras quantized layer.
    The weights of the Q{Dense, Conv} layer are scaled to the common data type,
    and an 'ApplyAlpha' layer is inserted to reapply the scale.
    '''

    def match(self, node):
        q_layer = node.class_name in ['Dense', 'Conv1D', 'Conv2D', 'Conv2DBatchnorm']
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
        import tensorflow as tf

        quantizer = node.weights['weight'].quantizer.quantizer_fn  # get QKeras quantizer
        weights = node.weights['weight'].data_unquantized  # get weights
        qweights = quantizer(tf.convert_to_tensor(weights))
        if isinstance(quantizer.scale, (int, float)):
            scale = np.ones(shape=node.get_output_variable().shape[-1]) * quantizer.scale
        else:
            scale = quantizer.scale.numpy()
        unscale = 1.0 / scale

        new_weights = unscale * qweights  # use the quantized weights for safety

        qcfg = quantizer.get_config()
        alpha = qcfg['alpha']
        # Set the alpha to 1 to avoid hitting this pass again
        qcfg['alpha'] = 1
        node.weights['weight'].quantizer.quantizer_fn = quantizer.from_config(qcfg)

        # update the weights also applying the hls4ml quantizer
        # this is only needed for the binary layers which encode -1 as 0
        quantized_new_weights = node.weights['weight'].quantizer(new_weights.numpy())
        node.weights['weight'].data = quantized_new_weights

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
            scale_bits = np.maximum(np.abs(np.log2(scale)).max().astype('int') + 1, 2)
            scale_quantizer = QKerasPO2Quantizer({'class_name': 'quantized_po2', 'config': {'bits': scale_bits}})
        else:
            scale_quantizer = None

        if 'Dense' in node.class_name:
            n_in = node.get_attr('n_out')
        elif 'Conv' in node.class_name:
            n_in = node.get_attr('out_width') * node.get_attr('out_height', 1) * node.get_attr('n_filt')
        else:
            n_in = node.get_attr('n_out')

        # the name of the new ApplyAlpha node
        alpha_name = node.get_attr('name') + '_alpha'

        # make the precision auto
        alpha_precision = {'Precision': 'auto'}
        model.config.set_name_config(alpha_name, alpha_precision)
        model.config.parse_name_config(alpha_name, alpha_precision)

        attrs = {
            'name': alpha_name,
            'class_name': 'Alpha',
            'inputs': node.outputs,
            'n_in': n_in,
            'n_filt': node.get_attr('n_filt', -1),
            'reuse_factor': node.get_attr('reuse_factor'),
            'scale_data': scale,
            'scale_quantizer': scale_quantizer,
            'bias_data': bias,
            'bias_quantizer': bias_quantizer,
            'trace': node.get_attr('trace', False),
        }
        alpha_layer = model.make_node(ApplyAlpha, node.name + '_alpha', attrs, node.outputs)
        model.insert_node(alpha_layer)
        return True


class ExtractTernaryThreshold(OptimizerPass):
    '''The input value (threshold) at which the output of a a ternary activation
    changes is configurable. This pass extracts that threshold point, inserting
    a BatchNormalization layer to execute the scaling. That BatchNormalization
    layer is then expected to be fused into a BatchNormalizationQuantizedTanh
    layer configured with the correct threshold.
    '''

    def match(self, node):
        return node.class_name == 'TernaryTanh' and node.get_attr('threshold', None) != 0.5

    def transform(self, model, node):
        shape = node.get_input_variable().shape
        scale = np.full(shape, 0.5 / node.get_attr('threshold', 0.5))
        bias = np.zeros_like(scale)
        node.set_attr('threshold', 0.5)

        attrs = {
            'name': node.get_attr('name') + '_scale',
            'class_name': 'Alpha',
            'inputs': node.get_input_node().outputs,
            'outputs': node.inputs,
            'n_in': node.get_attr('n_in'),
            'n_filt': node.get_attr('n_filt', -1),
            'reuse_factor': node.get_attr('reuse_factor'),
            'scale_data': scale,
            'bias_data': bias,
            'trace': node.get_attr('trace', False),
        }

        layer = model.make_node(ApplyAlpha, node.name + '_scale', attrs, node.inputs.copy())
        model.insert_node(layer, before=node)
        return True
