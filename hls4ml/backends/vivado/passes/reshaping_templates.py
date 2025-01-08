from math import prod

import numpy as np

from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Resize, Transpose, ZeroPadding1D, ZeroPadding2D

# ZeroPadding templates

zeropad1d_config_template = """struct config{index} : nnet::padding1d_config {{
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned out_width = {out_width};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
}};\n"""

zeropad2d_config_template = """struct config{index} : nnet::padding2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
}};\n"""

zeropad1d_function_template = 'nnet::zeropad1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
zeropad2d_function_template = 'nnet::zeropad2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'

padding_include_list = ['nnet_utils/nnet_padding.h', 'nnet_utils/nnet_padding_stream.h']


class ZeroPaddingConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((ZeroPadding1D, ZeroPadding2D))
        self.templates = {
            'ZeroPadding1D': zeropad1d_config_template,
            'ZeroPadding2D': zeropad2d_config_template,
        }

    def format(self, node):
        params = self._default_config_params(node)
        return self.templates[node.class_name].format(**params)


class ZeroPaddingFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((ZeroPadding1D, ZeroPadding2D), include_header=padding_include_list)
        self.templates = {
            'ZeroPadding1D': zeropad1d_function_template,
            'ZeroPadding2D': zeropad2d_function_template,
        }

    def format(self, node):
        params = self._default_function_params(node)
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'

        return self.templates[node.class_name].format(**params)


# Resize templates

resize_config_template = """struct config{index} : nnet::resize_config {{
    static const unsigned height = {in_height};
    static const unsigned width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned new_height = {out_height};
    static const unsigned new_width = {out_width};
}};\n"""

resize_function_template = 'nnet::resize_{algorithm}<{input_t}, {config}>({input}, {output});'

resize_include_list = ['nnet_utils/nnet_image.h', 'nnet_utils/nnet_image_stream.h']


class ResizeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Resize)
        self.template = resize_config_template

    def format(self, node):
        params = self._default_config_params(node)

        return self.template.format(**params)


class ResizeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Resize, include_header=resize_include_list)
        self.template = resize_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['algorithm'] = node.get_attr('algorithm')

        return self.template.format(**params)


# Transpose templates


transpose_include_list = ['nnet_utils/nnet_transpose.h', 'nnet_utils/nnet_transpose_stream.h']

transpose_config_template = """struct {config_name} {{
    static const unsigned dims = {dims};
    static const unsigned N = {N};
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
}};

unsigned {config_name}_from_shape[{dims}] = {{{from_shape}}};
unsigned {config_name}_to_shape[{dims}] = {{{to_shape}}};
unsigned {config_name}_perm[{dims}] = {{{perm}}};
unsigned {config_name}_perm_strides[{dims}] = {{{perm_strides}}};

const unsigned* const {config_name}::from_shape = {config_name}_from_shape;
const unsigned* const {config_name}::to_shape = {config_name}_to_shape;
const unsigned* const {config_name}::perm = {config_name}_perm;
const unsigned* const {config_name}::perm_strides = {config_name}_perm_strides;
"""

transpose_function_template = 'nnet::transpose<{input_t}, {output_t}, {config_name}>({input}, {output});'


def transpose_config_gen(name: str, shape: tuple[int, ...], perm: tuple[int, ...]):
    new_shape = tuple(shape[i] for i in perm)
    strides = np.cumprod((shape[1:] + (1,))[::-1])[::-1]
    perm_strides = tuple(int(strides[i]) for i in perm)
    return transpose_config_template.format(
        dims=len(shape),
        N=prod(shape),
        from_shape=', '.join(str(x) for x in shape),
        perm=', '.join(str(x) for x in perm),
        perm_strides=', '.join(str(x) for x in perm_strides),
        to_shape=', '.join(str(x) for x in new_shape),
        config_name=name,
    )


class TransposeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Transpose)
        self.template = transpose_config_template

    def format(self, node):
        shape = tuple(node.get_input_variable().shape)
        perm = tuple(node.get_attr('perm'))
        name = f'config{node.index}'
        return transpose_config_gen(name, shape, perm)


class TransposeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        self.template = transpose_function_template
        super().__init__(Transpose, include_header=transpose_include_list)

    def format(self, node):
        params = self._default_function_params(node)
        params['config_name'] = f'config{node.index}'
        return self.template.format(**params)
