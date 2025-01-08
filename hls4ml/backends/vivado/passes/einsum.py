from math import ceil

from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Einsum

from .reshaping_templates import transpose_config_gen

# Shared Dense template
# Einsum template

einsum_config_template = '''
struct config{index} {{
    typedef config{index}_tpose_inp0 tpose_inp0_conf;
    typedef config{index}_tpose_inp1 tpose_inp1_conf;
    typedef config{index}_tpose_out tpose_out_conf;

    typedef {accum_t.name} accum_t;

    // Layer Sizes
    static const unsigned n_free0 = {n_free0};
    static const unsigned n_free1 = {n_free1};
    static const unsigned n_contract = {n_contract};
    static const unsigned n_inplace = {n_inplace};

    // Resource reuse info
    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned multiplier_limit = {multiplier_limit};
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};
'''

einsum_function_template = 'nnet::einsum<{input0_t}, {input1_t}, {output_t}, {config}>({input0}, {input1}, {output});'

einsum_include_list = ['nnet_utils/nnet_einsum.h']


class EinsumConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Einsum)
        self.template = einsum_config_template

    def format(self, node: Einsum):
        default_params = self._default_config_params(node)

        strategy = node.model.config.get_strategy(node)
        io_type = node.model.config.get_config_value('IOType')

        assert io_type == 'io_parallel', 'EinsumDense layer only supports io_parallel for now'
        assert strategy.lower() == 'latency', 'EinsumDense layer only supports Latency strategy for now'

        # EinsumDense config
        params = default_params.copy()
        params['strategy'] = strategy
        params['n_free0'] = node.attributes.attributes['n_free0']
        params['n_free1'] = node.attributes.attributes['n_free1']
        params['n_contract'] = node.attributes.attributes['n_contract']
        params['n_inplace'] = node.attributes.attributes['n_inplace']
        inp0_t = node.get_input_variable(node.inputs[0]).type.precision
        inp1_t = node.get_input_variable(node.inputs[1]).type.precision
        params['product_type'] = get_backend('vivado').product_type(inp0_t, inp1_t)

        total_mults = params['n_free0'] * params['n_free1'] * params['n_contract'] * params['n_inplace']
        params['multiplier_limit'] = ceil(total_mults / params['reuse_factor'])

        einsum_conf = self.template.format(**params)

        # inp/out transpose config
        inp0_shape = node.attributes.attributes['inp0_shape']
        inp1_shape = node.attributes.attributes['inp1_shape']
        out_interpert_shape = node.attributes.attributes['out_interpert_shape']
        inp0_tpose_idxs = node.attributes.attributes['inp0_tpose_idxs']
        inp1_tpose_idxs = node.attributes.attributes['inp1_tpose_idxs']
        out_tpose_idxs = node.attributes.attributes['out_tpose_idxs']
        tpose_inp0_conf_name = f'config{node.index}_tpose_inp0'
        tpose_inp1_conf_name = f'config{node.index}_tpose_inp1'
        tpose_out_conf_name = f'config{node.index}_tpose_out'

        inp0_tpose_conf = transpose_config_gen(tpose_inp0_conf_name, inp0_shape, inp0_tpose_idxs)
        inp1_tpose_conf = transpose_config_gen(tpose_inp1_conf_name, inp1_shape, inp1_tpose_idxs)
        out_tpose_conf = transpose_config_gen(tpose_out_conf_name, out_interpert_shape, out_tpose_idxs)

        return '\n\n'.join((inp0_tpose_conf, inp1_tpose_conf, out_tpose_conf, einsum_conf))


class EinsumFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Einsum, include_header=einsum_include_list)
        self.template = einsum_function_template

    def format(self, node: Einsum):
        params = {}
        params['config'] = f'config{node.index}'
        params['input0_t'] = node.get_input_variable(node.inputs[0]).type.name
        params['input1_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['output_t'] = node.get_output_variable().type.name
        params['input0'] = node.get_input_variable(node.inputs[0]).name
        params['input1'] = node.get_input_variable(node.inputs[1]).name
        params['output'] = node.get_output_variable().name
        return self.template.format(**params)
