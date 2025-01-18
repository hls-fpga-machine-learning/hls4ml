import numpy as np

from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Reshape, Resize, Transpose, ZeroPadding1D, ZeroPadding2D

# ZeroPadding templates

zeropad1d_config_template = """struct config{index} : nnet::padding1d_config {{
    static const unsigned in_width = {in_width};
    static const unsigned out_width = {out_width};
    static const unsigned n_chan = {n_chan};

    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
}};\n"""

zeropad2d_config_template = """struct config{index} : nnet::padding2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned n_chan = {n_chan};

    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
}};\n"""

zeropad1d_function_template = 'nnet::zeropad1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
zeropad2d_function_template = 'nnet::zeropad2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'

zeropad1d_task_sequence_template = (
    'task_sequence<nnet::zeropad1d_{data_format}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)
zeropad2d_task_sequence_template = (
    'task_sequence<nnet::zeropad2d_{data_format}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)

reshaping_stream_function_template = '{name}.async();'

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
        if node.get_attr('data_format') == 'channels_first':
            raise Exception('oneAPI only supports channels_last data format')
        params['data_format'] = 'cl'

        return self.templates[node.class_name].format(**params)


class ZeroPaddingTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__((ZeroPadding1D, ZeroPadding2D))
        self.templates = {
            'ZeroPadding1D': zeropad1d_task_sequence_template,
            'ZeroPadding2D': zeropad2d_task_sequence_template,
        }

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('data_format') == 'channels_first':
            raise RuntimeError('channels_first not supported on oneAPI')
        params['data_format'] = 'cl'

        return self.templates[node.class_name].format(**params)


class ReshapingStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__((ZeroPadding1D, ZeroPadding2D, Resize, Reshape, Transpose))
        self.template = reshaping_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)


# Resize templates

resize_config_template = """struct config{index} : nnet::resize_config {{
    static const unsigned height = {in_height};
    static const unsigned width = {in_width};

    static const unsigned new_height = {out_height};
    static const unsigned new_width = {out_width};

    static const unsigned n_chan = {n_chan};
}};\n"""

resize_function_template = 'nnet::resize_{algorithm}<{input_t}, {output_t}, {config}>({input}, {output});'
resize_task_sequence_template = (
    'task_sequence<nnet::resize_{algorithm}_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
)
resize_include_list = ['nnet_utils/nnet_resize.h', 'nnet_utils/nnet_resize_stream.h']


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
        if node.get_attr('algorithm') != 'nearest':
            raise Exception('Currently only supporting resize_nearest')
        params['algorithm'] = node.get_attr('algorithm')

        return self.template.format(**params)


class ResizeTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Resize)
        self.template = resize_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.get_attr('algorithm') != 'nearest':
            raise Exception('Currently only supporting resize_nearest')
        params['algorithm'] = node.get_attr('algorithm')

        return self.template.format(**params)


# Transpose templates

transpose_config_template = """struct {config_name} : nnet::transpose_config {{
    static constexpr unsigned dims = {dims};
    static constexpr unsigned N = {N};
    static constexpr std::array<unsigned, dims> from_shape = {{{from_shape}}};
    static constexpr std::array<unsigned, dims> to_shape = {{{to_shape}}};
    static constexpr std::array<unsigned, dims> perm = {{{perm}}};
    static constexpr std::array<unsigned, dims> perm_strides = {{{perm_strides}}};
}};\n"""

transpose_function_template = 'nnet::transpose<{input_t}, {output_t}, {config}>({input}, {output});'
transpose_task_sequence_template = 'task_sequence<nnet::transpose_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
transpose_include_list = ['nnet_utils/nnet_transpose.h', 'nnet_utils/nnet_transpose_stream.h']


def permute_config_gen(name: str, shape: tuple[int, ...], perm: tuple[int, ...]):
    """
    Generate a configuration string for a permute operation. Operates by mapping the output index to input input index by:
     - unravel the output index
     - map each dimension to the corresponding stride in the input tensor, sum
    The operation can be expressed as:

    new_shape = tuple(shape[i] for i in perm)
    strides = np.cumprod((shapes[1:] + (1,))[::-1])[::-1]
    perm_strides = [strides[i] for i in perm]
    out[index] = inp[np.dot(np.unravel_index(index, new_shape), perm_strides)]

    Args:
        name (str): The name of the configuration.
        shape (tuple[int, ...]): The shape of the input tensor.
        perm (tuple[int, ...]): The permutation of the dimensions.

    Returns:
        str: The formatted configuration string for the permute operation.
    """
    new_shape = tuple(shape[i] for i in perm)
    strides = np.cumprod((shape[1:] + (1,))[::-1])[::-1]
    perm_strides = tuple(int(strides[i]) for i in perm)
    return transpose_config_template.format(
        dims=len(shape),
        N=np.prod(shape),
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
        return permute_config_gen(name, shape, perm)


class TransposeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Transpose, include_header=transpose_include_list)
        self.template = transpose_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['dim'] = node.get_attr('dim')

        return self.template.format(**params)


class TransposeTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Transpose)
        self.template = transpose_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        params['dim'] = node.get_attr('dim')

        return self.template.format(**params)


# Reshape template (only used in streaming)
reshape_task_sequence_template = 'task_sequence<nnet::repack_stream<{input_pipe}, {output_pipe}, {size}>> {name};'
reshape_include_list = ['nnet_utils/nnet_stream.h']


class ReshapeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Reshape)

    def format(self, node):
        return ''


class ReshapeFunctionTemplate(FunctionCallTemplate):
    """Only used to add the include list"""

    def __init__(self):
        super().__init__(Reshape, include_header=reshape_include_list)

    def format(self, node):
        return ''


class ReshapeTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(Reshape)
        self.template = reshape_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)
        params['size'] = np.prod(node.get_output_variable().shape)
        return self.template.format(**params)
