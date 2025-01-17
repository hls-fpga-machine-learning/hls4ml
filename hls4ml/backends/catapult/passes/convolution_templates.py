from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import (
    Conv1D,
    Conv2D,
    Conv2DBatchnorm,
    DepthwiseConv1D,
    DepthwiseConv2D,
    SeparableConv1D,
    SeparableConv2D,
)

# Shared multiplication template

conv_mult_config_template = """struct config{index}_mult : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned n_zeros = {nzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

# Conv1D templates

conv1d_config_template = """struct config{index} : nnet::conv1d_config {{
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_width = {filt_width};
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_width = {stride_width};
    static const unsigned dilation = {dilation};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const nnet::conv_implementation implementation = nnet::conv_implementation::{implementation};
    static const unsigned min_width = {min_width};
    static const ac_int<filt_width,false> pixels[min_width];
    static const unsigned n_partitions = {n_partitions};
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::{fill_fn}<data_T, CONFIG_T>;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::{scale_index_type}<K, S, W>;
}};
// really this allocation of pixels array ought to be in a .cpp file
#ifndef INCLUDED_MC_TESTBENCH_H
const ac_int<config{index}::filt_width,false> config{index}::pixels[] = {{{instructions}}};
#endif\n"""

conv1d_function_template = 'nnet::conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
depthconv1d_function_template = (
    'nnet::depthwise_conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)

conv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_conv1d_stream.h']


class Conv1DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Conv1D, DepthwiseConv1D))
        self.template = conv1d_config_template
        self.mult_template = conv_mult_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('weight').nzeros

        params['config_t'] = f'config{node.index}_mult'
        if node.get_attr('in_width') == node.get_attr('min_width'):
            params['scale_index_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_type'] = 'scale_index_regular'

        if node.model.config.get_config_value('IOType') == 'io_parallel':
            params['fill_fn'] = f'fill_buffer_{node.index}'
        else:
            params['fill_fn'] = 'FillConv1DBuffer'

        params.setdefault('min_width', node.get_attr('in_width'))
        params.setdefault('instructions', '0')

        conv_config = self.template.format(**params)

        mult_params = self._default_config_params(node)
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['nzeros'] = node.get_weights('weight').nzeros
        mult_params['product_type'] = get_backend('catapult').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )
        mult_config = self.mult_template.format(**mult_params)

        return mult_config + '\n' + conv_config


class Conv1DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Conv1D, include_header=conv1d_include_list)
        self.template = conv1d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)


class DepthwiseConv1DFunctionTemplate(Conv1DFunctionTemplate):
    def __init__(self):
        super(Conv1DFunctionTemplate, self).__init__(DepthwiseConv1D, include_header=sepconv1d_include_list)
        self.template = depthconv1d_function_template


# Conv2D Templates

conv2d_config_template = """struct config{index} : nnet::conv2d_config {{
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_height = {filt_height};
    static const unsigned filt_width = {filt_width};
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const nnet::conv_implementation implementation = nnet::conv_implementation::{implementation};
    static const unsigned min_height = {min_height};
    static const unsigned min_width = {min_width};
    static const ac_int<filt_height * filt_width,false> pixels[min_height * min_width];
    static const unsigned n_partitions = {n_partitions};
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::{fill_fn}<data_T, CONFIG_T>;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::{scale_index_height_type}<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::{scale_index_width_type}<K, S, W>;
}};
// really this allocation of pixels array ought to be in a .cpp file
#ifndef INCLUDED_MC_TESTBENCH_H
const ac_int<config{index}::filt_height * config{index}::filt_width,false> config{index}::pixels[] = {{{instructions}}};
#endif\n"""

conv2d_function_template = 'nnet::conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
depthconv2d_function_template = (
    'nnet::depthwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)

conv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_conv2d_stream.h']


class Conv2DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Conv2D, Conv2DBatchnorm, DepthwiseConv2D))
        self.template = conv2d_config_template
        self.mult_template = conv_mult_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('weight').nzeros

        params['config_t'] = f'config{node.index}_mult'

        if node.get_attr('in_height') == node.get_attr('min_height'):
            params['scale_index_height_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_height_type'] = 'scale_index_regular'

        if node.get_attr('in_width') == node.get_attr('min_width'):
            params['scale_index_width_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_width_type'] = 'scale_index_regular'

        if node.model.config.get_config_value('IOType') == 'io_parallel':
            params['fill_fn'] = f'fill_buffer_{node.index}'
        else:
            params['fill_fn'] = 'FillConv2DBuffer'

        params.setdefault('min_height', node.get_attr('in_height'))
        params.setdefault('min_width', node.get_attr('in_width'))
        params.setdefault('instructions', '0')

        conv_config = self.template.format(**params)

        mult_params = self._default_config_params(node)
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_height') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['nzeros'] = node.get_weights('weight').nzeros
        mult_params['product_type'] = get_backend('catapult').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )
        mult_config = self.mult_template.format(**mult_params)

        return mult_config + '\n' + conv_config


class Conv2DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Conv2D, Conv2DBatchnorm), include_header=conv2d_include_list)
        self.template = conv2d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name

        return self.template.format(**params)


class DepthwiseConv2DFunctionTemplate(Conv2DFunctionTemplate):
    def __init__(self):
        super(Conv2DFunctionTemplate, self).__init__(DepthwiseConv2D, include_header=sepconv2d_include_list)
        self.template = depthconv2d_function_template


# SeparableConv1D/2D Templates

sepconv_config_template = """struct config{index} {{
    typedef {depthwise_config} depthwise_config;
    typedef {pointwise_config} pointwise_config;
}};\n"""

sepconv1d_function_template = (
    'nnet::separable_conv_1d_{data_format}<{input_t}, {dw_output_t}, {output_t}, {config}>('
    '{input}, {output}, {d}, {p}, {z}, {b});'
)
sepconv2d_function_template = (
    'nnet::separable_conv_2d_{data_format}<{input_t}, {dw_output_t}, {output_t}, {config}>('
    '{input}, {output}, {d}, {p}, {z}, {b});'
)

sepconv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_sepconv1d_stream.h']
sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_sepconv2d.h', 'nnet_utils/nnet_sepconv2d_stream.h']


class SeparableConv1DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SeparableConv1D)
        self.template = sepconv_config_template
        self.depthwise_template = conv1d_config_template
        self.pointwise_template = conv1d_config_template
        self.depthwise_mult_template = conv_mult_config_template
        self.pointwise_mult_template = conv_mult_config_template

    def format(self, node):
        # Separable master config
        params = {}
        params['index'] = node.index
        params['depthwise_config'] = f'config{node.index}_depthwise'
        params['pointwise_config'] = f'config{node.index}_pointwise'
        sep_config = self.template.format(**params)

        # Depthwise config
        params = self._default_config_params(node)
        # Override bias and bias_t since these are zeros in depthwise step of SepConv1D
        params['bias'] = params['zero_bias']
        params['bias_t'] = params['zero_bias_t']
        params['n_filt'] = params['n_chan']  # In depthwise step n_chan == n_filt
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('depthwise').nzeros
        params['index'] = str(node.index) + '_depthwise'
        params['weight_t'] = node.get_weights('depthwise').type
        params['fill_fn'] = 'FillConv1DBuffer'

        if node.get_attr('unscaled'):
            params['scale_index_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_type'] = 'scale_index_regular'

        params['config_t'] = f'config{node.index}_depthwise_mult'
        depthwise_config = self.depthwise_template.format(**params)

        # Depthwise mult config
        mult_params = self._default_config_params(node)
        mult_params['index'] = str(node.index) + '_depthwise'
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_chan')
        mult_params['nzeros'] = node.get_weights('depthwise').nzeros
        mult_params['weight_t'] = node.get_weights('depthwise').type
        mult_params['product_type'] = get_backend('catapult').product_type(
            node.get_input_variable().type.precision, node.get_weights('depthwise').type.precision
        )
        depthwise_mult_config = self.depthwise_mult_template.format(**mult_params)

        # Pointwise config
        params = self._default_config_params(node)
        if node.get_attr('data_format') == 'channels_last':
            params['in_width'] = node.get_output_variable().shape[0]
        else:
            params['in_width'] = node.get_output_variable().shape[1]

        params['filt_width'] = 1
        params['stride_width'] = 1
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('pointwise').nzeros
        params['index'] = str(node.index) + '_pointwise'
        params['weight_t'] = node.get_weights('pointwise').type
        params['min_width'] = params['in_width']
        params['instructions'] = '0'
        params['fill_fn'] = 'FillConv1DBuffer'

        if node.get_attr('unscaled'):
            params['scale_index_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_type'] = 'scale_index_regular'

        params['config_t'] = f'config{node.index}_pointwise_mult'
        pointwise_config = self.pointwise_template.format(**params)

        # Pointwise mult config
        mult_params = self._default_config_params(node)
        mult_params['index'] = str(node.index) + '_pointwise'
        mult_params['n_in'] = node.get_attr('n_chan')
        mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['nzeros'] = node.get_weights('pointwise').nzeros
        mult_params['weight_t'] = node.get_weights('pointwise').type
        mult_params['product_type'] = get_backend('catapult').product_type(
            node.get_input_variable().type.precision, node.get_weights('pointwise').type.precision
        )
        pointwise_mult_config = self.pointwise_mult_template.format(**mult_params)

        return (
            depthwise_mult_config
            + '\n'
            + depthwise_config
            + '\n'
            + pointwise_mult_config
            + '\n'
            + pointwise_config
            + '\n'
            + sep_config
        )


class SeparableConv1DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SeparableConv1D, include_header=sepconv1d_include_list)
        self.template = sepconv1d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['dw_output_t'] = node.get_attr('dw_output_t').name
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'
        params['d'] = node.get_weights('depthwise').name
        params['p'] = node.get_weights('pointwise').name
        params['b'] = node.get_weights('bias').name
        params['z'] = node.get_weights('zero_bias').name

        return self.template.format(**params)


class SeparableConv2DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SeparableConv2D)
        self.template = sepconv_config_template
        self.depthwise_template = conv2d_config_template
        self.pointwise_template = conv2d_config_template
        self.depthwise_mult_template = conv_mult_config_template
        self.pointwise_mult_template = conv_mult_config_template

    def format(self, node):
        # Separable master config
        params = {}
        params['index'] = node.index
        params['depthwise_config'] = f'config{node.index}_depthwise'
        params['pointwise_config'] = f'config{node.index}_pointwise'
        sep_config = self.template.format(**params)

        # Depthwise config
        params = self._default_config_params(node)
        # Override bias and bias_t since these are zeros in depthwise step of SepConv2D
        params['bias'] = params['zero_bias']
        params['bias_t'] = params['zero_bias_t']
        params['n_filt'] = params['n_chan']  # In depthwise step n_chan == n_filt
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('depthwise').nzeros
        params['index'] = str(node.index) + '_depthwise'
        params['weight_t'] = node.get_weights('depthwise').type
        params['fill_fn'] = 'FillConv2DBuffer'

        if node.get_attr('unscaled_h'):
            params['scale_index_height_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_height_type'] = 'scale_index_regular'

        if node.get_attr('unscaled_w'):
            params['scale_index_width_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_width_type'] = 'scale_index_regular'

        params['config_t'] = f'config{node.index}_depthwise_mult'
        depthwise_config = self.depthwise_template.format(**params)

        # Depthwise mult config
        mult_params = self._default_config_params(node)
        mult_params['index'] = str(node.index) + '_depthwise'
        mult_params['n_in'] = node.get_attr('n_chan') * node.get_attr('filt_height') * node.get_attr('filt_width')
        mult_params['n_out'] = node.get_attr('n_chan')
        mult_params['nzeros'] = node.get_weights('depthwise').nzeros
        mult_params['weight_t'] = node.get_weights('depthwise').type
        mult_params['product_type'] = get_backend('catapult').product_type(
            node.get_input_variable().type.precision, node.get_weights('depthwise').type.precision
        )
        depthwise_mult_config = self.depthwise_mult_template.format(**mult_params)

        # Pointwise config
        params = self._default_config_params(node)
        if node.get_attr('data_format') == 'channels_last':
            params['in_height'] = node.get_output_variable().shape[0]
            params['in_width'] = node.get_output_variable().shape[1]
        else:
            params['in_height'] = node.get_output_variable().shape[1]
            params['in_width'] = node.get_output_variable().shape[2]

        params['filt_height'] = params['filt_width'] = 1
        params['stride_height'] = params['stride_width'] = 1
        params['dilation'] = node.get_attr('dilation', 1)
        params['nzeros'] = node.get_weights('pointwise').nzeros
        params['index'] = str(node.index) + '_pointwise'
        params['weight_t'] = node.get_weights('pointwise').type
        params['min_height'] = params['in_height']
        params['min_width'] = params['in_width']
        params['instructions'] = '0'
        params['fill_fn'] = 'FillConv2DBuffer'

        if node.get_attr('unscaled_h'):
            params['scale_index_height_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_height_type'] = 'scale_index_regular'

        if node.get_attr('unscaled_w'):
            params['scale_index_width_type'] = 'scale_index_unscaled'
        else:
            params['scale_index_width_type'] = 'scale_index_regular'
        params['config_t'] = f'config{node.index}_pointwise_mult'
        pointwise_config = self.pointwise_template.format(**params)

        # Pointwise mult config
        mult_params = self._default_config_params(node)
        mult_params['index'] = str(node.index) + '_pointwise'
        mult_params['n_in'] = node.get_attr('n_chan')
        mult_params['n_out'] = node.get_attr('n_filt')
        mult_params['nzeros'] = node.get_weights('pointwise').nzeros
        mult_params['weight_t'] = node.get_weights('pointwise').type
        mult_params['product_type'] = get_backend('catapult').product_type(
            node.get_input_variable().type.precision, node.get_weights('pointwise').type.precision
        )
        pointwise_mult_config = self.pointwise_mult_template.format(**mult_params)

        return (
            depthwise_mult_config
            + '\n'
            + depthwise_config
            + '\n'
            + pointwise_mult_config
            + '\n'
            + pointwise_config
            + '\n'
            + sep_config
        )


class SeparableConv2DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SeparableConv2D, include_header=sepconv2d_include_list)
        self.template = sepconv2d_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['dw_output_t'] = node.get_attr('dw_output_t').name
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'
        params['d'] = node.get_weights('depthwise').name
        params['p'] = node.get_weights('pointwise').name
        params['b'] = node.get_weights('bias').name
        params['z'] = node.get_weights('zero_bias').name

        return self.template.format(**params)
