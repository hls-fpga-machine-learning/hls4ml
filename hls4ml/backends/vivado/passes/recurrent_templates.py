from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import GRU, LSTM

# recurrent multiplication template

recr_mult_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = {dense_function}<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

# activation templates

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
}};\n"""

recr_activ_config_template = """struct {type}_config{index}_recr : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
}};\n"""

# LSTM + GRU templates

recr_config_template = """struct config{index} : nnet::{recr_type}_config {{
    typedef {accum_t.name} accum_t;
    typedef {weight_t.name} weight_t;  // Matrix
    typedef {bias_t.name} bias_t;  // Vector
    typedef {config_mult_t1} mult_config1;
    typedef {config_mult_t2} mult_config2;
    typedef {recr_act_t} ACT_CONFIG_{RECR_TYPE};
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::{recurrent_activation}<x_T, y_T, config_T>;
    typedef {act_t} ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;
    static const unsigned n_in  = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_state = {n_state};
    static const unsigned n_sequence = {n_sequence};
    static const unsigned n_sequence_out = {n_sequence_out};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    static const bool use_static = {static};
    static const bool pytorch_order = {pytorch};
}};\n"""

recr_function_template = 'nnet::{recr_type}_stack<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {wr}, {b}, {br});'

recr_include_list = ['nnet_utils/nnet_recurrent.h']


class RecurrentConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((LSTM, GRU))
        self.template = recr_config_template
        self.act_template = activ_config_template
        self.recr_act_template = recr_activ_config_template
        self.mult1_template = recr_mult_config_template
        self.mult2_template = recr_mult_config_template

    def format(self, node):
        params = self._default_config_params(node)

        params['n_in'] = node.get_input_variable().dim_names[1]
        params['n_sequence'] = node.get_input_variable().dim_names[0]
        if node.get_attr('return_sequences'):
            params['n_sequence_out'] = node.get_output_variable().dim_names[0]
            params['n_state'] = node.get_output_variable().dim_names[1]
            params['n_out'] = node.get_output_variable().dim_names[1]
        else:
            params['n_sequence_out'] = 1
            params['n_state'] = node.get_output_variable().dim_names[0]
            params['n_out'] = node.get_output_variable().dim_names[0]
        params['config_mult_t1'] = f'config{node.index}_1'
        params['config_mult_t2'] = f'config{node.index}_2'
        params['recr_act_t'] = '{}_config{}_recr'.format(node.get_attr('recurrent_activation'), node.index)
        params['act_t'] = '{}_config{}'.format(node.get_attr('activation'), node.index)
        params['strategy'] = node.get_attr('strategy')
        params['static'] = 'true' if node.attributes['static'] else 'false'
        params['pytorch'] = 'true' if node.get_attr('pytorch', False) else 'false'
        params['recr_type'] = node.class_name.lower()
        params['RECR_TYPE'] = node.class_name

        if node.class_name == 'LSTM':
            n_recr_mult = 4
        else:  # GRU
            n_recr_mult = 3

        recr_config = self.template.format(**params)

        act_params = self._default_config_params(node)
        recr_act_params = self._default_config_params(node)

        act_params['type'] = node.get_attr('activation')
        recr_act_params['type'] = node.get_attr('recurrent_activation')
        if node.get_attr('return_sequences'):
            act_params['n_in'] = node.get_output_variable().shape[1]
            recr_act_params['n_in'] = node.get_output_variable().shape[1] * (n_recr_mult - 1)
        else:
            act_params['n_in'] = node.get_output_variable().shape[0]
            recr_act_params['n_in'] = node.get_output_variable().shape[0] * (n_recr_mult - 1)

        act_config = self.act_template.format(**act_params)
        recr_act_config = self.recr_act_template.format(**recr_act_params)

        mult_params1 = self._default_config_params(node)
        mult_params2 = self._default_config_params(node)

        mult_params1['n_in'] = node.get_input_variable().shape[1]
        if node.get_attr('return_sequences'):
            mult_params1['n_out'] = node.get_output_variable().shape[1] * n_recr_mult
        else:
            mult_params1['n_out'] = node.get_output_variable().shape[0] * n_recr_mult
        mult_params1['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )
        mult_params1['reuse'] = params['reuse']
        mult_params1['index'] = str(node.index) + '_1'
        mult_params1['nzeros'] = node.get_weights('weight').nzeros
        mult_params1['nonzeros'] = node.get_weights('weight').nonzeros

        namespace = params['namespace']

        if node.get_attr('strategy').lower() == 'latency':
            mult_params1['dense_function'] = 'nnet::DenseLatency'
        elif node.get_attr('strategy').lower() == 'resource':
            if int(mult_params1['reuse_factor']) <= int(mult_params1['n_in']):
                mult_params1['dense_function'] = 'nnet::DenseResource_rf_leq_nin'
            else:
                mult_params1['dense_function'] = 'nnet::DenseResource_rf_gt_nin_rem0'
            # The 3rd case is never used
        elif node.get_attr('strategy').lower() == 'resource_unrolled':
            mult_params1['dense_function'] = f'{namespace}::dense_resource_unrolled_{node.index}_1'

        if node.get_attr('return_sequences'):
            mult_params2['n_in'] = node.get_output_variable().shape[1]
            mult_params2['n_out'] = node.get_output_variable().shape[1] * n_recr_mult
        else:
            mult_params2['n_in'] = node.get_output_variable().shape[0]
            mult_params2['n_out'] = node.get_output_variable().shape[0] * n_recr_mult
        mult_params2['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('recurrent_weight').type.precision
        )
        mult_params2['reuse'] = node.attributes['recurrent_reuse_factor']
        mult_params2['index'] = str(node.index) + '_2'
        mult_params2['nzeros'] = node.get_weights('recurrent_weight').nzeros
        mult_params2['nonzeros'] = node.get_weights('recurrent_weight').nonzeros

        if node.get_attr('strategy').lower() == 'latency':
            mult_params2['dense_function'] = 'nnet::DenseLatency'
        elif node.get_attr('strategy').lower() == 'resource':
            if int(mult_params2['reuse_factor']) <= int(mult_params2['n_in']):
                mult_params2['dense_function'] = 'nnet::DenseResource_rf_leq_nin'
            else:
                mult_params2['dense_function'] = 'nnet::DenseResource_rf_gt_nin_rem0'
            # The 3rd case is never used
        elif node.get_attr('strategy').lower() == 'resource_unrolled':
            mult_params2['dense_function'] = f'{namespace}::dense_resource_unrolled_{node.index}_2'

        mult_config1 = self.mult1_template.format(**mult_params1)
        mult_config2 = self.mult2_template.format(**mult_params2)

        return mult_config1 + '\n' + mult_config2 + '\n' + recr_act_config + '\n' + act_config + '\n' + recr_config


class RecurrentFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((LSTM, GRU), include_header=recr_include_list)
        self.template = recr_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name
        params['wr'] = node.get_weights('recurrent_weight').name
        params['br'] = node.get_weights('recurrent_bias').name
        params['activation'] = node.get_attr('activation')
        params['recurrent_activation'] = node.get_attr('recurrent_activation')
        params['recr_type'] = node.class_name.lower()

        return self.template.format(**params)
