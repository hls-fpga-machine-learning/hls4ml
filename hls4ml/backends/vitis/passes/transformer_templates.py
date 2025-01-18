from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import MultiHeadAttention

# dense layer template
mult_config_template = """struct config{index}_{mNum} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef {accum_t.name} accum_t;
    typedef {attention_output_bias_t.name} bias_t;
    typedef {attention_output_weight_t.name} weight_t;
    typedef ap_{index_t} index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::{dense_function}<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

# activation template
softmax_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::{implementation};
    typedef {table_t.name} exp_table_t;
    typedef {table_t.name} inv_table_t;
}};\n"""

mha_config_template = """struct config{index} : nnet::multiheadattention_config {{
    typedef {accum_t.name} accum_t;
    typedef {attention_output_bias_t.name} bias_t;
    typedef {attention_output_weight_t.name} weight_t;
    typedef {config_mult_t1} config_mult1;
    typedef {config_mult_t2} config_mult2;
    typedef {config_activ_t1} softmax_config1;

    static const unsigned num_heads = {num_heads};
    static const unsigned head_dim_key = {head_dim_key};
    static const unsigned head_dim_value = {head_dim_value};
    static const unsigned feature_dim = {feature_dim};
    static const unsigned seq_len = {seq_len};

    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
}};\n"""

mha_function_template = """nnet::multiheadattention<{input_t}, {output_t}, {config}>({input_q}, {input_kv},
                            {output}, {w_o}, {b_o}, {w_k}, {b_k}, {w_q}, {b_q}, {w_v}, {b_v});"""

mha_include_list = ['nnet_utils/nnet_multiheadattention.h']


class MhaConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(MultiHeadAttention)
        self.template = mha_config_template
        self.mult1_template = mult_config_template
        self.mult2_template = mult_config_template
        self.activ1_template = softmax_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['num_heads'] = node.get_attr('num_heads')
        params['head_dim_key'] = node.get_attr('head_dim_key')
        params['head_dim_value'] = node.get_attr('head_dim_value')
        params['feature_dim'] = node.get_attr('feature_dim')
        params['seq_len'] = node.get_attr('seq_len')
        params['config_mult_t1'] = f'config{node.index}_1'
        params['config_mult_t2'] = f'config{node.index}_2'
        params['config_activ_t1'] = '{}_config{}'.format("softmax", node.index)
        params['strategy'] = node.get_attr('strategy')
        mha_config = self.template.format(**params)

        mult_params1 = self._default_config_params(node)
        mult_params1['strategy'] = 'latency'
        mult_params1['mNum'] = '1'
        mult_params1['n_in'] = node.get_attr('feature_dim')
        mult_params1['n_out'] = node.get_attr('head_dim_key')
        mult_params1['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('query_weight').type.precision
        )
        mult_params1['reuse'] = params['reuse']
        mult_params1['index'] = str(node.index)
        mult_params1['nzeros'] = 0
        mult_params1['nonzeros'] = params['feature_dim'] * params['num_heads'] * params['head_dim_key']
        mult_params1['dense_function'] = 'DenseLatency'
        mult_config1 = self.mult1_template.format(**mult_params1)

        mult_params2 = self._default_config_params(node)
        mult_params2['strategy'] = 'latency'
        mult_params2['mNum'] = '2'
        mult_params2['n_in'] = node.get_attr('head_dim_value') * node.get_attr('num_heads')
        mult_params2['n_out'] = node.get_attr('feature_dim')
        mult_params2['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('attention_output_weight').type.precision
        )
        mult_params2['reuse'] = params['reuse']
        mult_params2['index'] = str(node.index)
        mult_params2['nzeros'] = 0
        mult_params2['nonzeros'] = params['feature_dim'] * params['num_heads'] * params['head_dim_key']
        mult_params2['dense_function'] = 'DenseLatency'
        mult_config2 = self.mult2_template.format(**mult_params2)

        act_params = self._default_config_params(node)
        act_params['n_in'] = node.get_attr('seq_len')
        act_params['type'] = 'softmax'
        act_params['implementation'] = 'legacy'  # in MHA: latency,stable not work， legacy works
        act_config = self.activ1_template.format(**act_params)

        return mult_config1 + '\n' + mult_config2 + '\n' + act_config + '\n' + mha_config


class MhaFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(MultiHeadAttention, include_header=mha_include_list)
        self.template = mha_function_template

    def format(self, node):
        params = {}
        params.update(node.attributes)
        params['config'] = f'config{node.index}'
        params['input_t'] = node.get_input_variable().type.name
        params['output_t'] = node.get_output_variable().type.name

        params['input_q'] = node.model.get_layer_output_variable(node.inputs[0]).name
        params['input_kv'] = node.model.get_layer_output_variable(node.inputs[1]).name
        params['output'] = node.get_output_variable().name
        params['w_o'] = node.get_weights('attention_output_weight').name
        params['b_o'] = node.get_weights('attention_output_bias').name
        params['w_k'] = node.get_weights('key_weight').name
        params['b_k'] = node.get_weights('key_bias').name
        params['w_q'] = node.get_weights('query_weight').name
        params['b_q'] = node.get_weights('query_bias').name
        params['w_v'] = node.get_weights('value_weight').name
        params['b_v'] = node.get_weights('value_bias').name

        return self.template.format(**params)
