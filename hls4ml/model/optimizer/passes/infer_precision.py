import math
from typing import Iterable

import numpy as np

from hls4ml.model.optimizer import ConfigurableOptimizerPass
from hls4ml.model.types import (
    FixedPrecisionType,
    IntegerPrecisionType,
    PrecisionType,
    RoundingMode,
    SaturationMode,
    UnspecifiedPrecisionType,
)

# TODO:  The code assumes everything is Fixed or Integer precision. Need to add checks


class InferPrecisionTypes(ConfigurableOptimizerPass):
    def __init__(self):
        # The option, infer_no_bias, allows you to tailor for the given weights, in particular, zero bias
        self.infer_no_bias = False

    def match(self, node):
        input_var = node.get_input_variable()
        if input_var is not None and isinstance(input_var.type, UnspecifiedPrecisionType):
            # only infer types if the input type is known
            return False
        for layer_type in node.types.values():
            if isinstance(layer_type.precision, UnspecifiedPrecisionType):
                return True
        return False

    def transform(self, model, node):
        types_to_infer = []
        for type_name, type_obj in node.types.items():
            if isinstance(type_obj.precision, UnspecifiedPrecisionType):
                types_to_infer.append(type_name)

        inferred_types = self._infer_precision(node, types_to_infer)
        for type_name in types_to_infer:
            if type_name not in inferred_types:
                self._infer_default_type(node, type_name)

        # if the return type was set, this may allow InferPrecisionTypes to be run
        # on layers it was not previously able to
        return 'result_t' in types_to_infer

    def _infer_precision(self, node, types_to_infer):
        node_class = node.class_name
        if node_class in ['Dense']:
            return self._infer_dense_precision(node, types_to_infer)

        if node_class in ['BatchNormalization', 'ApplyAlpha']:
            return self._infer_bn_precision(node, types_to_infer)

        if node_class in ['Conv1D', 'Conv2D', 'PointwiseConv1D', 'PointwiseConv2D', 'Conv2DBatchnorm']:
            return self._infer_conv_precision(node, types_to_infer)

        if node_class in ['DepthwiseConv1D', 'DepthwiseConv2D']:
            return self._infer_depthconv_precision(node, types_to_infer)

        if node_class in ['SeparableConv1D', 'SeparableConv2D']:
            return self._infer_sepconv_precision(node, types_to_infer)

        if node_class in ['Pooling1D', 'Pooling2D']:
            return self._infer_pooling_precision(node, types_to_infer)

        if node_class in ['Clone', 'Reshape', 'Resize', 'Transpose', 'ZeroPadding1D', 'ZeroPadding2D']:
            return self._infer_output_matching_precision(node, types_to_infer)

        if node_class in ['Merge']:
            return self._infer_merge_precision(node, types_to_infer)

        if node_class in ['Concatenate']:
            return self._infer_cat_precision(node, types_to_infer)

        if node_class in ['Dot']:
            return self._infer_dot_precision(node, types_to_infer)

        if node_class in ['Embedding']:
            return self._infer_embedding_precision(node, types_to_infer)

        if node_class in ['SimpleRNN', 'LSTM', 'GRU']:
            return self._infer_rnn_precision(node, types_to_infer)

        if node_class in ['ParametrizedActivation']:
            return self._infer_par_act_precision(node, types_to_infer)

        if node_class in ['MultiHeadAttention']:
            return self._infer_mha_precision(node, types_to_infer)

        # What about quantized activation layer? Setting it to 'auto' manually will break it here. We should prevent
        # this in config_from_* functions

        return []

    def _get_default_precision(self, node):
        model_config = node.model.config
        return model_config.backend.convert_precision_string(model_config.model_precision['default'])

    def _get_maximum_precision(self, node):
        model_config = node.model.config
        if 'maximum' in model_config.model_precision:
            return model_config.backend.convert_precision_string(model_config.model_precision['maximum'])
        else:
            return None

    def _all_supported_types(self, types: Iterable[PrecisionType]):
        """Are all the types supported for inference--currently Integer or Fixed"""
        for tp in types:
            if not isinstance(tp, (IntegerPrecisionType, FixedPrecisionType)):
                return False
        return True

    def _infer_default_type(self, node, type_name):
        model_config = node.model.config
        default_precision = model_config.backend.convert_precision_string(model_config.model_precision['default'])
        # No need to change the name of the NamedType since we use the default precision
        node.types[type_name].precision = default_precision

    def _infer_output_matching_precision(self, node, types_to_infer):
        assert 'result_t' in types_to_infer and len(types_to_infer) == 1

        in_var = node.get_input_variable()
        out_var = node.get_output_variable()
        in_out_type = in_var.type.precision
        out_var.type.precision = in_out_type

        return ['result_t']

    def _infer_common_precision(self, node, types_to_infer, n_ops):
        inferred_types = []

        input_precision = node.get_input_variable().type.precision

        if 'weight_t' in types_to_infer:
            weight_quantizer = node.get_attr('weight_quantizer', None)
            if weight_quantizer is not None:
                node.types['weight_t'].name = node.name + '_weight_t'
                node.types['weight_t'].precision = weight_quantizer.hls_type
            else:
                self._infer_default_type(node, 'weight_t')
            node.weights['weight'].update_precision(node.types['weight_t'].precision)
            inferred_types.append('weight_t')

        if 'bias_t' in types_to_infer:
            bias_quantizer = node.get_attr('bias_quantizer', None)
            if bias_quantizer is not None:
                node.types['bias_t'].name = node.name + '_bias_t'
                node.types['bias_t'].precision = bias_quantizer.hls_type
            else:
                self._infer_default_type(node, 'bias_t')
            node.weights['bias'].update_precision(node.types['bias_t'].precision)
            inferred_types.append('bias_t')

        if self._all_supported_types((input_precision, node.types['weight_t'].precision, node.types['bias_t'].precision)):
            input_width = input_precision.width
            input_integers = input_precision.integer
            input_signed = input_precision.signed

            weight_width = node.types['weight_t'].precision.width
            weight_integers = node.types['weight_t'].precision.integer
            weight_signed = node.types['weight_t'].precision.signed

            bias_width = node.types['bias_t'].precision.width
            bias_integers = node.types['bias_t'].precision.integer
            bias_signed = node.types['bias_t'].precision.signed
            no_bias = node.weights['bias'].nonzeros == 0 and self.infer_no_bias  # no bias

            # using math.ceil instead of np.ceil because it returns an int
            bitwidth = weight_width + input_width + math.ceil(np.log2(n_ops))
            integers = weight_integers + input_integers + math.ceil(np.log2(n_ops))
            signed = weight_signed or input_signed

            frac = bitwidth - integers

            if not no_bias:
                integers = max(integers + (bias_signed and not signed), bias_integers + (signed and not bias_signed)) + 1
                bitwidth = integers + max(frac, bias_width - bias_integers)
                signed = signed or bias_signed

            # if max_precision is specified, limit the size to be less than max precisoin
            max_precision = self._get_maximum_precision(node)
            if max_precision is not None:
                bitwidth = min(bitwidth, max_precision.width)
                integers = min(integers, max_precision.integer)

            # Note:  this is guaranteed to not overflow or need rounding, so it's sufficient to use the simpler form.
            new_type = FixedPrecisionType(bitwidth, integers, signed)
        else:
            new_type = self._get_default_precision(node)

        if 'accum_t' in types_to_infer:
            node.types['accum_t'].name = node.name + '_accum_t'
            node.types['accum_t'].precision = new_type

            inferred_types.append('accum_t')

        if 'result_t' in types_to_infer:
            node.types['result_t'].name = node.name + '_result_t'
            node.types['result_t'].precision = new_type

            inferred_types.append('result_t')

        return inferred_types

    def _infer_dense_precision(self, node, types_to_infer):
        n_ops = node.get_attr('n_in')
        return self._infer_common_precision(node, types_to_infer, n_ops)

    def _infer_conv_precision(self, node, types_to_infer):
        n_ops = node.get_attr('n_chan') * node.get_attr('filt_height', 1) * node.get_attr('filt_width')
        return self._infer_common_precision(node, types_to_infer, n_ops)

    def _infer_depthconv_precision(self, node, types_to_infer):
        n_ops = node.get_attr('filt_height', 1) * node.get_attr('filt_width')
        return self._infer_common_precision(node, types_to_infer, n_ops)

    # This function should generally not be called because we split sepconv to depthwise and regular (pointwise).
    # It has not been updated.
    def _infer_sepconv_precision(self, node, types_to_infer):
        inferred_types = []

        input_precision = node.get_input_variable().type.precision
        input_width = input_precision.width
        input_integers = input_precision.integer

        if 'depthwise_t' in types_to_infer:
            # TODO Current HLS implementations use data_T (input type) as the result hence this doesn't affect the output
            # precision ATM, but this will probably change in the future
            depthwise_quantizer = node.get_attr('depthwise_quantizer', None)
            if depthwise_quantizer is not None:
                node.types['depthwise_t'].name = node.name + '_depthwise_t'
                node.types['depthwise_t'].precision = depthwise_quantizer.hls_type
            else:
                self._infer_default_type(node, 'depthwise_t')
            node.weights['depthwise'].update_precision(node.types['depthwise_t'].precision)

            inferred_types.append('depthwise_t')

        if 'pointwise_t' in types_to_infer:
            pointwise_quantizer = node.get_attr('pointwise_quantizer', None)
            if pointwise_quantizer is not None:
                pointwise_width = pointwise_quantizer.bits
                pointwise_integers = pointwise_quantizer.hls_type.integer
                node.types['pointwise_t'].name = node.name + '_pointwise_t'
                node.types['pointwise_t'].precision = pointwise_quantizer.hls_type
            else:
                self._infer_default_type(node, 'pointwise_t')
                pointwise_width = node.types['pointwise_t'].precision.width
                pointwise_integers = node.types['pointwise_t'].precision.integer
            node.weights['pointwise'].update_precision(node.types['pointwise_t'].precision)

            inferred_types.append('pointwise_t')
        else:
            pointwise_width = node.types['pointwise_t'].precision.width
            pointwise_integers = node.types['pointwise_t'].precision.integer

        if 'bias_t' in types_to_infer:
            bias_quantizer = node.get_attr('bias_quantizer', None)
            if bias_quantizer is not None:
                bias_width = bias_quantizer.bits
                bias_integers = bias_quantizer.hls_type.integer
                node.types['bias_t'].name = node.name + '_bias_t'
                node.types['bias_t'].precision = bias_quantizer.hls_type
            else:
                self._infer_default_type(node, 'bias_t')
                bias_width = node.types['bias_t'].precision.width
                bias_integers = node.types['bias_t'].precision.integer
            node.weights['bias'].update_precision(node.types['bias_t'].precision)

            inferred_types.append('bias_t')
        else:
            bias_width = node.types['bias_t'].precision.width
            bias_integers = node.types['bias_t'].precision.integer

        n_ops = node.get_attr('n_chan')
        new_type = FixedPrecisionType(
            width=int(max(np.ceil(input_width + pointwise_width + np.log2(n_ops)), bias_width) + 1),
            integer=int(max(np.ceil(input_integers + pointwise_integers + np.log2(n_ops)), bias_integers) + 1),
        )

        if 'accum_t' in types_to_infer:
            node.types['accum_t'].name = node.name + '_accum_t'
            node.types['accum_t'].precision = new_type

            inferred_types.append('accum_t')

        if 'result_t' in types_to_infer:
            node.types['result_t'].name = node.name + '_result_t'
            node.types['result_t'].precision = new_type

            inferred_types.append('result_t')

        return inferred_types

    def _infer_bn_precision(self, node, types_to_infer):
        """
        The batchnormalziation precision here is the more implementation-focused version. It propagates
        precision from scale and bias, not mean, variance, etc.
        """

        inferred_types = []

        if 'scale_t' in types_to_infer:
            self._infer_default_type(node, 'scale_t')
            node.weights['scale'].update_precision(node.types['scale_t'].precision)
            inferred_types.append('scale_t')

        if 'bias_t' in types_to_infer:
            self._infer_default_type(node, 'bias_t')
            node.weights['bias'].update_precision(node.types['bias_t'].precision)
            inferred_types.append('bias_t')

        if 'result_t' in types_to_infer:
            input_precision = node.get_input_variable().type.precision
            scale_precision = node.types['scale_t'].precision
            bias_precision = node.types['bias_t'].precision

            if self._all_supported_types((input_precision, scale_precision, bias_precision)):

                after_scale_signed = scale_precision.signed or input_precision.signed
                after_scale_width = input_precision.width + scale_precision.width
                after_scale_integer = input_precision.integer + scale_precision.integer

                out_precision_signed = after_scale_signed or bias_precision.signed
                out_precision_integer = (
                    max(
                        after_scale_integer + (bias_precision.signed and not after_scale_signed),
                        bias_precision.integer + (after_scale_signed and not bias_precision.signed),
                    )
                    + 1
                )
                out_precision_width = out_precision_integer + max(
                    after_scale_width - after_scale_integer, bias_precision.fractional
                )

                # if max_precision is specified, limit the size to be less than max precisoin
                max_precision = self._get_maximum_precision(node)
                if max_precision is not None:
                    out_precision_width = min(out_precision_width, max_precision.width)
                    out_precision_integer = min(out_precision_integer, max_precision.integer)

                # Note:  this is guaranteed to not overflow or need rounding, so it's sufficient to use the simpler form.
                out_precision = FixedPrecisionType(out_precision_width, out_precision_integer, out_precision_signed)

            else:
                out_precision = self._get_default_precision(node)

            node.types['result_t'].name = node.name + '_result_t'
            node.types['result_t'].precision = out_precision

            inferred_types.append('result_t')

        return inferred_types

    def _infer_pooling_precision(self, node, types_to_infer):
        inferred_types = []

        if 'accum_t' in types_to_infer:
            input_precision = node.get_input_variable().type.precision
            pool_op = node.attributes['pool_op'].lower()

            if pool_op == 'max':
                # This has the benefit of working for xnor types. I don't think "copy" is needed
                accum_type = input_precision

            elif pool_op == 'average':
                if self._all_supported_types((input_precision,)):
                    width = input_precision.width
                    integer = input_precision.integer
                    signed = input_precision.signed

                    pool_size = node.get_attr('pool_height', 1) * node.get_attr('pool_width')
                    extra_bits = int(np.ceil(np.log2(pool_size)))

                    # for now ignore max precision in this case
                    accum_type = FixedPrecisionType(
                        width=width + extra_bits * 2, integer=integer + extra_bits, signed=signed
                    )
                else:
                    accum_type = self._get_default_precision(node)

            else:
                raise ValueError(f'Unknown pooling operation: {pool_op}')

            node.types['accum_t'].name = node.name + '_accum_t'
            node.types['accum_t'].precision = accum_type

            inferred_types.append('accum_t')

        if 'result_t' in types_to_infer:
            self._infer_output_matching_precision(node, ['result_t'])
            inferred_types.append('result_t')

        return inferred_types

    def _infer_merge_precision(self, node, types_to_infer):
        assert 'result_t' in types_to_infer and len(types_to_infer) == 1

        input_1 = node.get_input_variable(node.inputs[0]).type.precision
        input_2 = node.get_input_variable(node.inputs[1]).type.precision

        op = node.get_attr('op').lower()
        if op in ('add', 'subtract', 'average'):
            if self._all_supported_types((input_1, input_2)):
                new_signed = input_1.signed or input_2.signed or op == 'subtract'
                new_int = (
                    max(
                        input_1.integer + (input_2.signed and not input_1.signed),
                        input_2.integer + (input_1.signed and not input_2.signed),
                    )
                    + 1
                )
                new_width = new_int + max(input_1.fractional, input_2.fractional)
                max_precision = self._get_maximum_precision(node)
                if max_precision is not None:
                    new_width = min(new_width, max_precision.width)
                    new_int = min(new_int, max_precision.integer)
                out_precision = FixedPrecisionType(new_width, new_int, new_signed)
            else:
                out_precision = self._get_default_precision(node)
        elif op == 'multiply':
            if self._all_supported_types((input_1, input_2)):
                new_signed = input_1.signed or input_2.signed
                new_int = input_1.integer + input_2.integer
                new_width = input_1.width + input_2.width
                # if max_precision is specified, limit the size to be less than max precisoin
                max_precision = self._get_maximum_precision(node)
                if max_precision is not None:
                    new_width = min(new_width, max_precision.width)
                    new_int = min(new_int, max_precision.integer)
                out_precision = FixedPrecisionType(new_width, new_int, new_signed)
            else:
                out_precision = self._get_default_precision(node)
        elif op in ('maximum', 'minimum'):
            if input_1 == input_2:
                # can handle binary and potentially others
                out_precision = input_1  # I assume copy is not necessary
            elif self._all_supported_types((input_1, input_2)):
                new_signed = input_1.signed or input_2.signed

                input_1_integer = input_1.integer
                input_2_integer = input_2.integer

                # add one to integer if unsigned while new is signed
                if new_signed and not input_1.signed:
                    input_1_integer += 1
                if new_signed and not input_2.signed:
                    input_2_integer += 1

                new_width = max(input_1.fractional, input_2.fractional) + max(input_1_integer, input_2_integer)
                new_int = max(input_1_integer, input_2_integer)
                out_precision = FixedPrecisionType(new_width, new_int, new_signed)
            else:
                out_precision = self._get_default_precision(node)
        else:
            print(f'Warning: not propagating weights for type {op}')
            out_precision = self._get_default_precision(node)

        node.types['result_t'].name = node.name + '_result_t'
        node.types['result_t'].precision = out_precision

        return ['result_t']

    def _infer_cat_precision(self, node, types_to_infer):
        assert 'result_t' in types_to_infer and len(types_to_infer) == 1

        input_1 = node.get_input_variable(node.inputs[0]).type.precision
        input_2 = node.get_input_variable(node.inputs[1]).type.precision

        if input_1 == input_2:
            # can handle binary and potentially others
            out_precision = input_1  # I assume copy is not necessary
        elif self._all_supported_types((input_1, input_2)):
            new_signed = input_1.signed or input_2.signed

            input_1_integer = input_1.integer
            input_2_integer = input_2.integer

            # add one to integer if unsigned while new is signed
            if new_signed and not input_1.signed:
                input_1_integer += 1
            if new_signed and not input_2.signed:
                input_2_integer += 1

            new_width = max(input_1.fractional, input_2.fractional) + max(input_1_integer, input_2_integer)
            new_int = max(input_1_integer, input_2_integer)

            # if max_precision is specified, limit the size to be less than max precisoin
            max_precision = self._get_maximum_precision(node)
            if max_precision is not None:
                new_width = min(new_width, max_precision.width)
                new_int = min(new_int, max_precision.integer)

            # some logic copied from former SetPrecisionConcat optimizer
            newrmode = input_1.rounding_mode if input_1.rounding_mode != RoundingMode.TRN else input_2.rounding_mode
            newsmode = input_1.saturation_mode if input_1.saturation_mode != SaturationMode.WRAP else input_2.saturation_mode
            newsbits = input_1.saturation_bits if input_1.saturation_bits != 0 else input_2.saturation_bits

            out_precision = FixedPrecisionType(new_width, new_int, new_signed, newrmode, newsmode, newsbits)
        else:
            out_precision = self._get_default_precision(node)

        node.types['result_t'].name = node.name + '_result_t'
        node.types['result_t'].precision = out_precision

        return ['result_t']

    def _infer_dot_precision(self, node, types_to_infer):
        assert 'result_t' in types_to_infer and len(types_to_infer) == 1

        input_1 = node.get_input_variable(node.inputs[0]).type.precision
        input_2 = node.get_input_variable(node.inputs[1]).type.precision

        if self._all_supported_types((input_1, input_2)):
            n_in = node.get_input_variable(node.inputs[0]).shape[0]

            new_signed = input_1.signed or input_2.signed
            new_width = input_1.width + input_2.width + math.ceil(np.log2(n_in))
            new_int = input_1.integer + input_2.integer + math.ceil(np.log2(n_in))

            # if max_precision is specified, limit the size to be less than max precisoin
            max_precision = self._get_maximum_precision(node)
            if max_precision is not None:
                new_width = min(new_width, max_precision.width)
                new_int = min(new_int, max_precision.integer)

            out_precision = FixedPrecisionType(new_width, new_int, new_signed)
        else:
            out_precision = self._get_default_precision(node)
        node.types['result_t'].name = node.name + '_result_t'
        node.types['result_t'].precision = out_precision

        return ['result_t']

    def _infer_embedding_precision(self, node, types_to_infer):
        inferred_types = []

        if 'embeddings_t' in types_to_infer:
            self._infer_default_type(node, 'embeddings_t')
            node.weights['embeddings'].update_precision(node.types['embeddings_t'].precision)
            inferred_types.append('embeddings_t')

        if 'result_t' in types_to_infer:
            out_precision = self._get_default_precision(node)
            node.types['result_t'].name = node.name + '_result_t'
            node.types['result_t'].precision = out_precision
            inferred_types.append('result_t')

        return inferred_types

    # TODO:  This is just a placeholder
    def _infer_rnn_precision(self, node, types_to_infer):
        inferred_types = []

        # for now just do the weights and leave the rest for the default catch
        for weightvar in ('weight', 'bias', 'recurrent_weight', 'recurrent_bias'):
            if f'{weightvar}_t' in types_to_infer:
                self._infer_default_type(node, f'{weightvar}_t')
                node.weights[weightvar].update_precision(node.types[f'{weightvar}_t'].precision)
                inferred_types.append(f'{weightvar}_t')

        return inferred_types

    def _infer_par_act_precision(self, node, types_to_infer):
        inferred_types = []

        # For threshold relu, set the parameter precision to be the input precision by default;
        # for other parametrized activations, just allow the default precision to be used.
        # Can override these values in the configuration by explicitly setting them.
        if 'param_t' in inferred_types and self.get_attr('activation').lower() == 'thresholdedrelu':
            in_type = node.get_input_variable().type.precision
            node.attributes['param_t'].type = in_type
            inferred_types.append('param_t')

        return inferred_types

    def _infer_mha_precision(self, node, types_to_infer):
        inferred_types = []

        for weightvar in (
            'attention_output_weight',
            'attention_output_bias',
            'key_weight',
            'key_bias',
            'query_weight',
            'query_bias',
            'value_weight',
            'value_bias',
        ):
            if f'{weightvar}_t' in types_to_infer:
                self._infer_default_type(node, f'{weightvar}_t')
                node.weights[weightvar].update_precision(node.types[f'{weightvar}_t'].precision)
                inferred_types.append(f'{weightvar}_t')

        if 'result_t' in types_to_infer:
            input_precision = node.get_input_variable().type.precision
            weight_precision = node.types['attention_output_weight_t'].precision
            bias_precision = node.types['attention_output_bias_t'].precision

            if self._all_supported_types((input_precision, weight_precision, bias_precision)):

                after_weight_width = input_precision.width + weight_precision.width
                after_weight_integer = input_precision.integer + weight_precision.integer
                after_weight_signed = input_precision.signed or weight_precision.signed

                out_signed = after_weight_signed or bias_precision.signed
                out_integer = (
                    max(
                        after_weight_integer + (bias_precision.signed and not after_weight_signed),
                        bias_precision.integer + (after_weight_signed and not bias_precision.signed),
                    )
                    + 1
                )
                out_width = out_integer + max(after_weight_width - after_weight_integer, bias_precision.fractional)

                # Apply max precision constraints if specified in model config
                max_precision = self._get_maximum_precision(node)
                if max_precision is not None:
                    out_width = min(out_width, max_precision.width)
                    out_integer = min(out_integer, max_precision.integer)

                out_precision = FixedPrecisionType(out_width, out_integer, out_signed)
            else:
                out_precision = self._get_default_precision(node)

            node.types['result_t'].name = f'{node.name}_result_t'
            node.types['result_t'].precision = out_precision
            inferred_types.append('result_t')

        return inferred_types
