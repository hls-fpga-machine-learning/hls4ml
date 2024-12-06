import math
import re
import subprocess
from bisect import bisect_left
from collections.abc import Iterable

import numpy as np

from hls4ml.backends.backend import Backend
from hls4ml.model.attributes import ConfigurableAttribute, TypeAttribute
from hls4ml.model.layers import (
    GRU,
    LSTM,
    Activation,
    BatchNormalization,
    BatchNormOnnx,
    Conv,
    Conv1D,
    Conv2D,
    Dense,
    Dot,
    Embedding,
    GarNet,
    GarNetStack,
    GlobalPooling1D,
    GlobalPooling2D,
    MatMul,
    Merge,
    Pooling1D,
    Pooling2D,
    Quant,
    SeparableConv1D,
    SeparableConv2D,
    SimpleRNN,
)
from hls4ml.model.optimizer import model_optimizer
from hls4ml.model.types import (
    ExponentPrecisionType,
    FixedPrecisionType,
    IntegerPrecisionType,
    PrecisionType,
    UnspecifiedPrecisionType,
    XnorPrecisionType,
)
from hls4ml.utils import attribute_descriptions as descriptions
from hls4ml.writer import get_writer


class FPGABackend(Backend):
    def __init__(self, name):
        super().__init__(name)

        self.writer = get_writer(self.name)

        self.attribute_map = {}

        accum_layers = [
            Dense,
            Conv1D,
            Conv2D,
            Pooling1D,
            Pooling2D,
            GlobalPooling1D,
            GlobalPooling2D,
            SimpleRNN,
            LSTM,
            GRU,
            Dot,
            Conv,
            MatMul,
        ]

        for layer in accum_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(TypeAttribute('accum', description=descriptions.accum_type))
            self.attribute_map[layer] = attrs

        rf_layers = accum_layers + [
            BatchNormalization,
            Activation,
            Embedding,
            GarNet,
            GarNetStack,
            Quant,
            BatchNormOnnx,
            Merge,
        ]

        for layer in rf_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('reuse_factor', default=1, description=descriptions.reuse_factor))
            self.attribute_map[layer] = attrs

        # seperable is kind of special because it is effectively two layers that will be split
        for layer in (SeparableConv1D, SeparableConv2D):
            attrs = self.attribute_map.get(layer, [])
            attrs.append(TypeAttribute('depthwise_accum'))
            attrs.append(TypeAttribute('pointwise_accum'))
            attrs.append(TypeAttribute('depthwise_result'))
            attrs.append(ConfigurableAttribute('depthwise_reuse_factor', default=1))
            attrs.append(ConfigurableAttribute('pointwise_reuse_factor', default=1))
            self.attribute_map[layer] = attrs

        act_attrs = self.attribute_map.get(Activation, [])
        act_attrs.append(ConfigurableAttribute('table_size', default=1024, description=descriptions.table_size))
        act_attrs.append(TypeAttribute('table', default=FixedPrecisionType(18, 8), description=descriptions.table_type))
        self.attribute_map[Activation] = act_attrs

    def create_layer_class(self, layer_class):
        new_attrubutes = []
        for cls, attributes in self.attribute_map.items():
            if issubclass(layer_class, cls):
                new_attrubutes.extend(attributes)

        return type(
            self.name + layer_class.__name__, (layer_class,), {'_expected_attributes': new_attrubutes, '_wrapped': True}
        )

    def compile(self, model):
        """Compile the generated project that can be linked into Python runtime.

        Args:
            model (ModelGraph): Model to compile.

        Raises:
            Exception: If the project failed to compile

        Returns:
            string: Returns the name of the compiled library.
        """

        lib_name = None
        ret_val = subprocess.run(
            ['./build_lib.sh'],
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=model.config.get_output_dir(),
        )
        if ret_val.returncode != 0:
            print(ret_val.stdout)
            raise Exception(f'Failed to compile project "{model.config.get_project_name()}"')
        lib_name = '{}/firmware/{}-{}.so'.format(
            model.config.get_output_dir(), model.config.get_project_name(), model.config.get_config_value('Stamp')
        )

        return lib_name

    def write(self, model):
        """Write the generated project to disk.

        This function converts the model to C++ and writes the generated files in the output
        directory specified in the `config`.

        Args:
            model (ModelGraph): Model to write.
        """

        model.apply_flow(self.get_writer_flow())

    def get_writer_flow(self):
        raise NotImplementedError

    def get_layer_mult_size(self, layer):
        if 'Dense' in layer.class_name:
            n_in = layer.get_attr('n_in')
            n_out = layer.get_attr('n_out')
            return n_in, n_out

        if 'Conv1D' in layer.class_name:
            n_in = layer.get_attr('n_chan') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')
            return n_in, n_out

        if 'Conv2D' in layer.class_name:
            n_in = layer.get_attr('n_chan') * layer.get_attr('filt_height') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')
            return n_in, n_out

        if 'LSTM' in layer.class_name:
            n_in = layer.get_attr('n_in')
            n_out = layer.get_attr('n_out') * 4
            n_in_recr = layer.get_attr('n_out')
            n_out_recr = n_out
            return n_in, n_out, n_in_recr, n_out_recr

        if 'GRU' in layer.class_name:
            n_in = layer.get_attr('n_in')
            n_out = layer.get_attr('n_out') * 3
            n_in_recr = layer.get_attr('n_out')
            n_out_recr = n_out
            return n_in, n_out, n_in_recr, n_out_recr

        raise Exception(f'Cannot get mult size for layer {layer.name} ({layer.class_name})')

    def get_valid_reuse_factors(self, n_in, n_out):
        max_rf = n_in * n_out
        valid_reuse_factors = []
        for rf in range(1, max_rf + 1):
            _assert = self._validate_reuse_factor(n_in, n_out, rf)
            if _assert:
                valid_reuse_factors.append(rf)
        return valid_reuse_factors

    def _validate_reuse_factor(self, n_in, n_out, rf):
        multfactor = min(n_in, rf)
        multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
        #
        # THIS ASSERTION IS FOR THE FUNCTIONAL CORRECTNESS OF THE DENSE LAYER
        #
        _assert = ((multiplier_limit % n_out) == 0) or (rf >= n_in)
        _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
        #
        # THIS ASSERTION IS FOR QoR AND EXECUTION TIME
        #
        _assert = _assert and (((n_in * n_out) % rf) == 0)

        return _assert

    def get_closest_reuse_factor(self, valid_rf, chosen_rf):
        """
        Returns closest value to chosen_rf. valid_rf is sorted (obtained from get_valid_reuse_factors())
        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(valid_rf, chosen_rf)
        if pos == 0:
            return valid_rf[0]
        if pos == len(valid_rf):
            return valid_rf[-1]
        before = valid_rf[pos - 1]
        after = valid_rf[pos]
        if after - chosen_rf < chosen_rf - before:
            return after
        else:
            return before

    def set_closest_reuse_factor(self, layer, n_in, n_out, attribute='reuse_factor', include_max_rf=True):
        assert attribute is not None, 'Reuse factor attribute cannot be None'

        valid_rf = self.get_valid_reuse_factors(n_in, n_out)
        if not include_max_rf:
            valid_rf.pop()
        chosen_rf = layer.get_attr(attribute)
        if chosen_rf not in valid_rf:
            closest_rf = self.get_closest_reuse_factor(valid_rf, chosen_rf)
            valid_rf_str = ','.join(map(str, valid_rf))
            print(
                f'WARNING: Invalid ReuseFactor={chosen_rf} in layer "{layer.name}".'
                f'Using ReuseFactor={closest_rf} instead. Valid ReuseFactor(s): {valid_rf_str}.'
            )
            layer.set_attr(attribute, closest_rf)

    def set_target_reuse_factor(self, layer):
        # TODO update target reuse factor for the RNN layers
        targ_cycles = layer.get_attr('target_cycles')

        shuffle_cycles = 6  # Number of clock cycles to move data around
        if targ_cycles is not None:
            if 'Dense' in layer.class_name:
                kernel_multiplies = layer.get_attr('n_out')
            elif 'Conv1D' in layer.class_name:
                kernel_multiplies = layer.get_attr('out_width')
            elif 'Conv2D' in layer.class_name:
                kernel_multiplies = layer.get_attr('out_height') * layer.get_attr('out_width')
            else:
                print(f'Unable to set target reuse factor for layer: {layer.name} ({layer.class_name})')
                return

            if targ_cycles < shuffle_cycles * kernel_multiplies:  # 6 clock min (6 * out_height * out_width)
                print(
                    'Latency can not be achieved with current target {}. Mininum {}.'.format(
                        targ_cycles, shuffle_cycles * kernel_multiplies + 1
                    )
                )
                return
            else:
                rf = targ_cycles - shuffle_cycles * kernel_multiplies  # subtract data shuffling overhead

            layer.set_attr('reuse_factor', float(rf) / kernel_multiplies)

    def get_valid_conv_partition_splits(self, out_height, out_width):
        """Generate valid partition splits of a Conv1D/2D layer.

        Essentially a list of divisors of the number of pixels of the output image.

        Args:
            out_height (int): The height of the output image
            out_width (int): The width of the output image

        Returns:
            list: List of valid partition splits
        """
        n_pixels = out_height * out_width
        valid_n_partitions = []
        for i in range(1, int(n_pixels / 2) + 1):
            if n_pixels % i == 0:
                valid_n_partitions.append(i)
        valid_n_partitions.append(n_pixels)

        return valid_n_partitions

    @classmethod
    def convert_precision_string(cls, precision):
        if isinstance(precision, PrecisionType):
            return precision

        if precision.lower() == 'auto':
            return cls._convert_auto_type(precision)

        if precision.startswith('ac_'):
            return cls._convert_ac_type(precision)
        else:
            return cls._convert_ap_type(precision)

    @classmethod
    def _convert_ap_type(cls, precision):
        '''
        Convert a precision string (e.g. "ap_fixed<16,6>" to the internal FixedPrecisionTypes etc)
        '''
        bits = re.search('.+<(.+?)>', precision).group(1).split(',')
        sat_mode = None
        round_mode = None
        sat_bits = None
        if 'fixed' in precision:
            width = int(bits[0])
            integer = int(bits[1])
            fields = 2
            signed = not ('u' in precision)
        elif 'int' in precision:
            width = int(bits[0])
            integer = width
            fields = 1
            signed = not ('u' in precision)
        if len(bits) > fields:
            round_mode = bits[fields]
        if len(bits) > fields + 1:
            sat_mode = bits[fields + 1]
        if len(bits) > fields + 2:
            sat_bits = int(bits[fields + 2])
        if 'fixed' in precision:
            return FixedPrecisionType(width, integer, signed, round_mode, sat_mode, sat_bits)
        elif 'int' in precision:
            return IntegerPrecisionType(width, signed)

    @classmethod
    def _convert_ac_type(cls, precision):
        '''
        Convert a precision string (e.g. "ac_fixed<16,6>" to the internal FixedPrecisionTypes etc)
        '''
        bits = re.search('.+<(.+?)>', precision).group(1).split(',')
        signed = True  # default is signed
        sat_mode = None
        round_mode = None
        if 'fixed' in precision:
            width = int(bits[0])
            integer = int(bits[1])
            fields = 2
            if len(bits) > 2:
                # only if the third argument is false or 0, set signed to False
                # (default is True)
                if bits[2].strip().lower() in ['false', '0']:
                    signed = False
                fields = 3
        elif 'int' in precision:
            width = int(bits[0])
            integer = width
            fields = 1
            if len(bits) > 1:
                # only if the second argument is false or 0, set signed to False
                # (default is True)
                if bits[1].strip().lower() in ['false', '0']:
                    signed = False
                fields = 2
        if len(bits) > fields:
            round_mode = bits[fields]
        if len(bits) > fields + 1:
            sat_mode = bits[fields + 1]
        if 'fixed' in precision:
            return FixedPrecisionType(width, integer, signed, round_mode, sat_mode)
        elif 'int' in precision:
            return IntegerPrecisionType(width, signed)

    @classmethod
    def _convert_auto_type(cls, precision):
        '''
        Convert a "auto" precision string into the UnspecifiedPrecisionType
        '''
        return UnspecifiedPrecisionType()

    def product_type(self, data_T, weight_T):
        '''
        Helper function to determine which product implementation to use during inference
        '''
        assert not isinstance(
            data_T, ExponentPrecisionType
        ), "Only ExponentPrecisionType (aka 'power of 2') weights are currently supported, not data."
        product = 'mult'
        if isinstance(weight_T, ExponentPrecisionType):
            product = 'weight_exponential'
        else:
            # if binary
            if isinstance(weight_T, XnorPrecisionType) and isinstance(data_T, XnorPrecisionType):
                product = 'both_binary'
            elif isinstance(weight_T, XnorPrecisionType):  # data is not xnor-binary
                product = 'weight_binary'
            elif isinstance(data_T, XnorPrecisionType):  # data is xnor, weight is not
                product = 'data_binary'
            elif isinstance(weight_T, IntegerPrecisionType) and weight_T.width == 2 and weight_T.signed:
                product = 'weight_ternary'
            else:
                product = 'mult'
        return product

    def compute_conv1d_instructions(self, in_W, in_C, kernel_size=3, stride=1, pad=0):
        # Current limitations
        assert pad == 0

        if kernel_size >= stride:
            min_W = (math.ceil(kernel_size / stride) - 1) * stride + kernel_size
        else:
            min_W = (math.ceil(stride / kernel_size) - 1) * stride + kernel_size

        # if the standard min_W is smaller than the in_W, then use unscaled
        if min_W > in_W:
            min_W = in_W

        min_oW = int((min_W - kernel_size) // stride + 1)

        out_W = int((in_W - kernel_size) // stride + 1)
        scaled_W = (out_W - 1) * stride + kernel_size

        if scaled_W < in_W:
            min_W += 1

        windows_bin = [[0 for _ in range(kernel_size)] for _ in range(min_W)]

        for i_ow in range(min_oW):
            for i_fw in range(kernel_size):
                index_data = i_ow * stride + i_fw - pad
                windows_bin[index_data][i_fw] = 1

        windows_int = []

        for i in range(min_W):
            windows_int.append(int(''.join(str(p) for p in reversed(windows_bin[i])), 2))

        return (min_W, windows_int)

    def compute_conv2d_instructions(self, in_H, in_W, in_C, kernel_size=3, stride=1, pad=0):
        if isinstance(kernel_size, Iterable):
            kernel_height = kernel_size[0]
            kernel_width = kernel_size[1]
        else:
            kernel_height = kernel_size
            kernel_width = kernel_size

        if isinstance(stride, Iterable):
            stride_height = stride[0]
            stride_width = stride[1]
        else:
            stride_height = stride
            stride_width = stride

        # Current limitations
        assert kernel_height == kernel_width
        assert stride_height == stride_width
        assert pad == 0

        if kernel_height >= stride_height:
            min_H = (math.ceil(kernel_height / stride_height) - 1) * stride_height + kernel_height
        else:
            min_H = (math.ceil(stride_height / kernel_height) - 1) * stride_height + kernel_height

        if min_H > in_H:
            min_H = in_H

        if kernel_width >= stride_width:
            min_W = (math.ceil(kernel_width / stride_width) - 1) * stride_width + kernel_width
        else:
            min_W = (math.ceil(stride_width / kernel_width) - 1) * stride_width + kernel_width

        if min_W > in_W:
            min_W = in_W

        min_oH = int((min_H - kernel_height) // stride_height + 1)
        min_oW = int((min_W - kernel_width) // stride_width + 1)

        out_H = int((in_H - kernel_height) // stride_height + 1)
        out_W = int((in_W - kernel_width) // stride_width + 1)
        scaled_H = (out_H - 1) * stride_height + kernel_height
        scaled_W = (out_W - 1) * stride_width + kernel_width

        if scaled_H < in_H:
            min_H += 1
        if scaled_W < in_W:
            min_W += 1

        # Let's hardcode a few common cases:
        if (
            min_H == 1
            and min_W == 1
            and kernel_height == 1
            and kernel_width == 1
            and stride == 1
            and scaled_H == in_H
            and scaled_W == in_W
        ):
            return (1, 1, map(str, [1]))
        if (
            min_H == 5
            and min_W == 5
            and kernel_height == 3
            and kernel_width == 3
            and stride == 1
            and scaled_H == in_H
            and scaled_W == in_W
        ):
            return (
                5,
                5,
                map(
                    str,
                    [
                        1,
                        3,
                        7,
                        6,
                        4,
                        9,
                        27,
                        63,
                        54,
                        36,
                        73,
                        219,
                        511,
                        438,
                        292,
                        72,
                        216,
                        504,
                        432,
                        288,
                        64,
                        192,
                        448,
                        384,
                        256,
                    ],
                ),
            )
        if (
            min_H == 9
            and min_W == 9
            and kernel_height == 5
            and kernel_width == 5
            and stride == 1
            and scaled_H == in_H
            and scaled_W == in_W
        ):
            return (
                9,
                9,
                map(
                    str,
                    [
                        1,
                        3,
                        7,
                        15,
                        31,
                        30,
                        28,
                        24,
                        16,
                        33,
                        99,
                        231,
                        495,
                        1023,
                        990,
                        924,
                        792,
                        528,
                        1057,
                        3171,
                        7399,
                        15855,
                        32767,
                        31710,
                        29596,
                        25368,
                        16912,
                        33825,
                        101475,
                        236775,
                        507375,
                        1048575,
                        1014750,
                        947100,
                        811800,
                        541200,
                        1082401,
                        3247203,
                        7576807,
                        16236015,
                        33554431,
                        32472030,
                        30307228,
                        25977624,
                        17318416,
                        1082400,
                        3247200,
                        7576800,
                        16236000,
                        33554400,
                        32472000,
                        30307200,
                        25977600,
                        17318400,
                        1082368,
                        3247104,
                        7576576,
                        16235520,
                        33553408,
                        32471040,
                        30306304,
                        25976832,
                        17317888,
                        1081344,
                        3244032,
                        7569408,
                        16220160,
                        33521664,
                        32440320,
                        30277632,
                        25952256,
                        17301504,
                        1048576,
                        3145728,
                        7340032,
                        15728640,
                        32505856,
                        31457280,
                        29360128,
                        25165824,
                        16777216,
                    ],
                ),
            )

        windows_bin = [[0 for _ in range(kernel_height * kernel_width)] for _ in range(min_H * min_W)]

        for i_oh in range(min_oH):
            for i_ow in range(min_oW):
                for i_fh in range(kernel_height):
                    for i_fw in range(kernel_width):
                        index_data = (i_oh * stride_height + i_fh - pad) * min_W + (i_ow * stride_width + i_fw - pad)
                        windows_bin[index_data][i_fh * kernel_width + i_fw] = 1

        windows_int = []

        for i in range(min_H):
            for j in range(min_W):
                windows_int.append(int(''.join(str(p) for p in reversed(windows_bin[i * min_W + j])), 2))

        return (min_H, min_W, windows_int)

    def _compute_conv1d_im2col(self, input_shape, kernel=3, stride=1, pad=(0, 0), dilation=1):
        W, C = input_shape
        pad_l, pad_r = pad

        out_w = (W + pad_l + pad_r - (dilation * (kernel - 1) + 1)) // stride + 1

        input_img = np.arange(1, W * C + 1)
        im_matrix = np.zeros((kernel * C * out_w,))

        index = 0
        for i_ow in range(out_w):
            for i_kw in range(kernel):
                for i_c in range(C):
                    input_col = -pad_l + i_kw * dilation + i_ow * stride
                    if input_col >= 0 and input_col < W:
                        im_matrix[index] = input_img[input_col * C + i_c]
                    else:
                        im_matrix[index] = 0
                    index += 1

        im_matrix = im_matrix.reshape(out_w, -1)
        return im_matrix

    def generate_conv1d_line_buffer_fn(self, layer_idx, n_partitions, in_W, in_C, kernel=3, stride=1, pad=0, dilation=1):
        """Generate a C++ function that mimics the im2col algorithm. This function works for 1D convolution.

        The HLS compiler produces suboptimal designs for a im2col algorithm implementation, so a trick we use is
        to generate a resulting a result of im2col transformation explicitly, instead of relying on loops. Since
        the result depends on the parameters of the convolution layer (the input size, the kernel size, stride etc),
        we need to do this for every convolution layer.

        Args:
            layer_idx (int): Index of layer ('index' attribute).
            n_partitions (int): Number of partitions to divide the input into.
                The pixels in each partition will be processed in parallel.
            in_W (int): Width of input.
            in_C (int): Number of channels.
            kernel (int, optional): Size of the kernel. Defaults to 3.
            stride (int, optional): Stride length. Defaults to 1.
            pad (int or Iterable, optional): Padding to apply. Defaults to 0.
                Specified as either a number or a list [left_pad, right_pad].
            dilation (int, optional): Dilation rate. Defaults to 1.

        Returns:
            str: Generated C++ function
        """
        if isinstance(pad, Iterable):
            pad_left = pad[0]
            pad_right = pad[1]
        else:
            pad_left = pad
            pad_right = pad

        im2col_matrix = self._compute_conv1d_im2col((in_W, in_C), kernel, stride, (pad_left, pad_right), dilation)

        generated_code = (
            "template<class data_T, typename CONFIG_T>\n"
            "class fill_buffer_{index} : public FillConv1DBuffer<data_T, CONFIG_T> {{\n"
            "    public:\n"
            "    static void fill_buffer(\n"
            "        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],\n"
            "        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],\n"
            "        const unsigned partition\n"
            "    ) {{\n"
        ).format(index=layer_idx)
        indent = '    '

        for partition_idx, partition in enumerate(np.split(im2col_matrix, n_partitions)):
            generated_code += indent * 2 + f'if (partition == {partition_idx:>3}) {{\n'
            for pixel_idx, arr in enumerate(partition):
                buffer_stmts = []
                for j, v in enumerate(arr):
                    if v == 0:
                        val = '0'
                    else:
                        val = f'data[{int(v - 1)}]'
                    buffer_stmts.append(f'buffer[{pixel_idx}][{j}] = {val:>10};')
                generated_code += indent * 3 + ' '.join(buffer_stmts) + '\n'
            generated_code += '\n' + indent * 2 + '}\n'

        generated_code += indent + '}\n'
        generated_code += '};\n'

        return generated_code

    def _compute_conv2d_im2col(self, input_shape, kernel=(3, 3), stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1)):
        H, W, C = input_shape
        kernel_h, kernel_w = kernel
        stride_h, stride_w = stride
        pad_t, pad_b, pad_l, pad_r = pad
        dilation_h, dilation_w = dilation

        out_h = (H + pad_t + pad_b - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        out_w = (W + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

        input_img = np.arange(1, H * W * C + 1)
        im_matrix = np.zeros((kernel_h * kernel_w * C * out_h * out_w,))

        index = 0
        for i_oh in range(out_h):
            for i_ow in range(out_w):
                for i_kh in range(kernel_h):
                    input_row = -pad_t + i_kh * dilation_h + i_oh * stride_h
                    for i_kw in range(kernel_w):
                        for i_c in range(C):
                            if input_row < 0 or input_row >= H:
                                im_matrix[index] = 0
                            else:
                                input_col = -pad_l + i_kw * dilation_w + i_ow * stride_w
                                if input_col >= 0 and input_col < W:
                                    im_matrix[index] = input_img[input_row * W * C + input_col * C + i_c]
                                else:
                                    im_matrix[index] = 0
                            index += 1

        im_matrix = im_matrix.reshape(out_h * out_w, -1)
        return im_matrix

    def generate_conv2d_line_buffer_fn(
        self, layer_idx, n_partitions, in_H, in_W, in_C, kernel=(3, 3), stride=(1, 1), pad=(0, 0, 0, 0), dilation=(1, 1)
    ):
        """Generate a C++ function that mimics the im2col algorithm. This function works for 2D convolution.

        The HLS compiler produces suboptimal designs for a im2col algorithm implementation, so a trick we use is
        to generate a resulting a result of im2col transformation explicitly, instead of relying on loops. Since
        the result depends on the parameters of the convolution layer (the input size, the kernel size, stride etc),
        we need to do this for every convolution layer.

        Args:
            layer_idx (int): Index of layer ('index' attribute).
            n_partitions (int): Number of partitions to divide the input into.
                The pixels in each partition will be processed in parallel.
            in_H (int): Height of input.
            in_W (int): Width of input.
            in_C (int): Number of channels.
            kernel (int or Iterable, optional): Size of the kernel. Defaults to (3,3).
            stride (int or Iterable, optional): Stride length. Defaults to (1,1).
            pad (int or Iterable, optional): Padding to apply. Defaults to 0.
                Specified as either a number or a list [top_pad, bottom_pad, left_pad, right_pad].
            dilation (int or Iterable, optional): Dilation rate. Defaults to (1,1).

        Returns:
            str: Generated C++ function
        """

        if isinstance(kernel, Iterable):
            kernel_height = kernel[0]
            kernel_width = kernel[1]
        else:
            kernel_height = kernel
            kernel_width = kernel

        if isinstance(stride, Iterable):
            stride_height = stride[0]
            stride_width = stride[1]
        else:
            stride_height = stride
            stride_width = stride

        if isinstance(pad, Iterable):
            pad_top = pad[0]
            pad_bottom = pad[1]
            pad_left = pad[2]
            pad_right = pad[3]
        else:
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        if isinstance(dilation, Iterable):
            dilation_height = dilation[0]
            dilation_width = dilation[1]
        else:
            dilation_height = dilation
            dilation_width = dilation

        im2col_matrix = self._compute_conv2d_im2col(
            (in_H, in_W, in_C),
            (kernel_height, kernel_width),
            (stride_height, stride_width),
            (pad_top, pad_bottom, pad_left, pad_right),
            (dilation_height, dilation_width),
        )

        generated_code = (
            "template<class data_T, typename CONFIG_T>\n"
            "class fill_buffer_{index} : public FillConv2DBuffer<data_T, CONFIG_T> {{\n"
            "    public:\n"
            "    static void fill_buffer(\n"
            "        data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],\n"
            "        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],\n"
            "        const unsigned partition\n"
            "    ) {{\n"
        ).format(index=layer_idx)
        indent = '    '

        for partition_idx, partition in enumerate(np.split(im2col_matrix, n_partitions)):
            generated_code += indent * 2 + f'if (partition == {partition_idx:>3}) {{\n'
            for pixel_idx, arr in enumerate(partition):
                buffer_stmts = []
                for j, v in enumerate(arr):
                    if v == 0:
                        val = '0'
                    else:
                        val = f'data[{int(v - 1)}]'
                    buffer_stmts.append(f'buffer[{pixel_idx}][{j}] = {val:>10};')
                generated_code += indent * 3 + ' '.join(buffer_stmts) + '\n'
            generated_code += '\n' + indent * 2 + '}\n'

        generated_code += indent + '}\n'
        generated_code += '};\n'

        return generated_code

    @model_optimizer()
    def write_hls(self, model):
        self.writer.write_hls(model)
        return True
