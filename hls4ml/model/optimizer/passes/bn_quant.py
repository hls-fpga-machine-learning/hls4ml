import numpy as np
import re

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.hls_model import Layer, IntegerPrecisionType, FixedPrecisionType, XnorPrecisionType, register_layer
from hls4ml.model.hls_layers import BatchNormalization
from hls4ml.templates import templates

class BatchNormalizationQuantizedTanh(Layer):
    ''' Merged Batch Normalization and quantized (binary or ternary) Tanh layer.
        The mean, variance, beta, gamma parameters are folded into the threshold(s) at which the
        sign of the input flips after the quantized (binary or ternary) Tanh activation.
    '''

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        if self.get_attr('quantize') == 2:
            self.add_output_variable(shape, dims, precision=XnorPrecisionType())
        elif self.get_attr('quantize') == 3:
            self.add_output_variable(shape, dims, precision=IntegerPrecisionType(width=2))
        else:
            print("Not adding output variable")

    def function_cpp(self):
        params = self._default_function_params()
        if self.get_attr('quantize') == 2:
            params['quantize'] = 'binary'
            params['threshold'] = self.get_weights('threshold').name
        elif self.get_attr('quantize') == 3:
            params['quantize'] = 'ternary'
            params['threshold'] = self.get_weights('threshold_hi').name + ', ' + self.get_weights('threshold_lo').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()

        return self._config_template.format(**params)

    def set_thresholds(self, scale, bias, ternary_threshold=0.5):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        precision = self.model.config.backend.convert_precision_string(inp.type.precision)
        W, I, F = precision.width, precision.integer, precision.fractional
        threshold = - bias / scale
        if self.get_attr('quantize') == 2:
            self.add_output_variable(shape, dims, precision=XnorPrecisionType())
            threshold = np.floor(threshold * 2**F) / 2**F
            self.add_weights_variable(name='threshold', var_name='t{index}', data=threshold, type_name='threshold{index}_t', precision=inp.type.precision)
        elif self.get_attr('quantize') == 3:
            self.add_output_variable(shape, dims, precision=IntegerPrecisionType(width=2))
            threshold_hi = ternary_threshold / scale + threshold
            threshold_lo = -ternary_threshold / scale + threshold
            threshold_hi = np.floor(threshold_hi * 2**F) / 2**F
            threshold_lo = np.floor(threshold_lo * 2**F) / 2**F
            self.add_weights_variable(name='threshold_hi', var_name='th{index}', data=threshold_hi, type_name='threshold_hi_{index}_t', precision=inp.type.precision)
            self.add_weights_variable(name='threshold_lo', var_name='tl{index}', data=threshold_lo, type_name='threshold_lo_{index}_t', precision=inp.type.precision)

batchnorm_quantized_tanh_config_template = """struct config{index} : nnet::batchnorm_quantized_tanh_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};\n"""

batchnorm_quantized_tanh_function_template = 'nnet::normalize_{quantize}_tanh<{input_t}, {config}>({input}, {output}, {threshold});'

# Register the layer types to the layer map
register_layer('BatchNormalizationQuantizedTanh', BatchNormalizationQuantizedTanh)

from hls4ml.templates.vivado_template import batchnorm_include_list

# Register the templates for config and function
templates.get_backend('Vivado').register_templates('BatchNormalizationQuantizedTanh', batchnorm_quantized_tanh_function_template, batchnorm_quantized_tanh_config_template, batchnorm_include_list)

class MergeBatchNormAndQuantizedTanh(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Activation'
            and node.get_attr('activation') in ['binary', 'binary_tanh', 'ternary', 'ternary_tanh']
            and isinstance(node.get_input_node(), BatchNormalization))
        is_match = is_match or node.__class__.__name__ == 'TernaryTanh'
        return is_match

    def transform(self, model, node):
        bn_layer = node.get_input_node()
        # Make a new layer with the new attributes
        quantize = 0
        if 'binary' in node.get_attr('activation'):
            quantize = 2
        if 'ternary' in node.get_attr('activation'):
            quantize = 3
        attrs = {
            'name' : bn_layer.get_attr('name'),
            'original_name' : bn_layer.get_attr('name'),
            'class_name' : 'BatchNormalizationQuantizedTanh',
            'n_in' : bn_layer.get_attr('n_in'),
            'n_out' : bn_layer.get_attr('n_in'),
            'n_filt' : bn_layer.get_attr('n_filt'),
            'epsilon' : bn_layer.get_attr('epsilon'),
            'quantize' : quantize,
            'Trace' : bn_layer.get_attr('Trace')
        }
        bnbt_layer = model.make_node('BatchNormalizationQuantizedTanh', 'bnbt_' + bn_layer.name, attrs, bn_layer.inputs)
        bnbt_layer.set_thresholds(bn_layer.get_weights('scale').data, bn_layer.get_weights('bias').data, node.get_attr('threshold',0.5))
        # Remove the BatchNormalization layer
        model.remove_node(bn_layer, rewire=True)
        # Replace the old Activation layer with this one
        model.remove_node(node, rewire=True)
        #model.replace_node(node, bnbt_layer)
        model.insert_node(bnbt_layer)

        return True

class QuantizeDenseOutput(OptimizerPass):
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Dense' and node.get_attr('weight_quantizer') is not None
            and node.get_input_node().__class__.__name__ == 'BatchNormalizationQuantizedTanh')
        return is_match

    def transform(self, model, node):
        # Compute the required precision and update the variables
        # Number of bits for output is log2 of number of input nodes
        # Since this is the number of uint<1>'s which are summed
        nbits = int(np.ceil(np.log2(node.attributes['n_in'])) + 2)
        out_type = IntegerPrecisionType(width=nbits)
        node.set_attr('accum_t', out_type)
        out_var = node.get_output_variable()
        out_var.type.precision = out_type

        quantized_data = None
        quantized_precision = None
        quantizer = node.get_attr('weight_quantizer')
        if quantizer.__class__.__name__ == 'BinaryQuantizer':
            quantized_precision = XnorPrecisionType()
        elif quantizer.__class__.__name__ == 'TernaryQuantizer':
            quantized_precision = IntegerPrecisionType(width=2)
        else:
            print('WARNING: Unknown quantizer - {}. Bailing out'.format(quantizer.__class__.__name__))
            return False
        quantizer.bits = quantized_precision.width
        quantizer.hls_type = quantized_precision
        quantized_data = quantizer(node.weights['weight'].data)

        weights = node.weights['weight']
        weights.data = quantized_data
        weights.type.name = 'weight{index}_t'.format(index=node.index)
        weights.update_precision(quantized_precision)

        bias = node.weights['bias']
        bias.data = np.zeros(shape=(node.get_attr('n_out')))
        bias.type.name = 'bias{index}_t'.format(index=node.index)
        bias.nzeros = 0
        bias.update_precision(quantized_precision)

        # If followed by the BatchNormalizationBinaryTanh, update its input
        # Also requantise the weights
        bd_out_nodes = node.get_output_nodes()
        for out_node in bd_out_nodes:
            if out_node.__class__.__name__ == 'BatchNormalizationQuantizedTanh':
                var_names = []
                if quantizer.__class__.__name__ == 'BinaryQuantizer':
                    var_names.append('threshold')
                elif quantizer.__class__.__name__ == 'TernaryQuantizer':
                    var_names.append('threshold_hi')
                    var_names.append('threshold_lo')
                for var_name in var_names:
                    threshold_var = out_node.weights[var_name]
                    threshold_var.update_precision(out_type)
                    threshold_var.data = np.floor(threshold_var.data)

        return False

