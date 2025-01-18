from hls4ml.backends.backend import get_backend
from hls4ml.backends.oneapi.oneapi_template import StreamFunctionCallTemplate, TaskSequenceTemplate
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import GRU, LSTM, SimpleRNN

# Note:  currently only GRU is supported for stream; lstm and simpleRNN are parallel-only

recurrent_include_list = ['nnet_utils/nnet_recurrent.h', 'nnet_utils/nnet_recurrent_stream.h']

################################################
# Shared Matrix Multiplication Template (Dense)
################################################
recr_mult_x_config_template = '''struct config{index}_mult : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};

    static const unsigned rf_pad = {rfpad};
    static const unsigned bf_pad = {bfpad};
    static const unsigned reuse_factor = {reuse};
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n'''

recr_mult_h_config_template = '''struct config{index}_mult : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};

    static const unsigned rf_pad = {rfpad};
    static const unsigned bf_pad = {bfpad};
    static const unsigned reuse_factor = {reuse};
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;
    typedef {accum_t.name} accum_t;
    typedef {recurrent_bias_t.name} bias_t;
    typedef {recurrent_weight_t.name} weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n'''

################################################
# Shared Activation Template
################################################
activ_config_template = '''struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t.name} table_t;
}};\n'''

################################################
# GRU Template
################################################
gru_config_template = '''struct config{index} : nnet::gru_config {{
    static const unsigned n_in  = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_units = {n_units};
    static const unsigned n_timesteps = {n_timesteps};
    static const unsigned n_outputs = {n_outputs};
    static const bool return_sequences = {return_sequences};

    typedef {accum_t.name} accum_t;
    typedef {weight_t.name} weight_t;
    typedef {bias_t.name} bias_t;
    typedef {recurrent_weight_t.name} recurrent_weight_t;
    typedef {recurrent_bias_t.name} recurrent_bias_t;

    typedef {config_mult_x} mult_config_x;
    typedef {config_mult_h} mult_config_h;

    typedef {act_t} ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;

    typedef {act_recurrent_t} ACT_CONFIG_RECURRENT_T;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::{recurrent_activation}<x_T, y_T, config_T>;

    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
}};\n'''

gru_function_template = 'nnet::gru<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {wr}, {b}, {br});'
gru_function_initial_state_template = (
    'nnet::gru_init_state<{input_t}, {h_t}, {output_t}, {config}>({input}, {init_state}, {output}, {w}, {wr}, {b}, {br});'
)
gru_task_sequence_template = 'task_sequence<nnet::gru_stream<{input_pipe}, {output_pipe}, {config}>> {name};'
gru_stream_function_template = '{name}.async({w}, {wr}, {b}, {br});'


class GRUConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(GRU)
        self.gru_template = gru_config_template
        self.act_template = activ_config_template
        self.recr_act_template = activ_config_template
        self.mult_x_template = recr_mult_x_config_template
        self.mult_h_template = recr_mult_h_config_template

    def format(self, node):
        # Input has shape (n_timesteps, inp_dimensionality)
        # Output / hidden units has shape (1 if !return_sequences else n_timesteps , n_units)
        params = self._default_config_params(node)
        params['n_units'] = node.get_attr('n_out')
        params['n_outputs'] = node.get_attr('n_timesteps') if node.get_attr('return_sequences', False) else '1'
        params['return_sequences'] = 'true' if node.get_attr('return_sequences', False) else 'false'
        params['config_mult_x'] = f'config{node.index}_x_mult'
        params['config_mult_h'] = f'config{node.index}_h_mult'
        params['act_t'] = '{}_config{}'.format(node.get_attr('activation'), str(node.index) + '_act')
        params['act_recurrent_t'] = '{}_config{}'.format(node.get_attr('recurrent_activation'), str(node.index) + '_rec_act')
        gru_config = self.gru_template.format(**params)

        # Activation is on candidate hidden state, dimensionality (1, n_units)
        act_params = self._default_config_params(node)
        act_params['type'] = node.get_attr('activation')
        act_params['n_in'] = node.get_attr('n_out')
        act_params['index'] = str(node.index) + '_act'
        act_config = self.act_template.format(**act_params)

        # Recurrent activation is on reset and update gates (therefore x2), dimensionality (1, n_units)
        recr_act_params = self._default_config_params(node)
        recr_act_params['type'] = node.get_attr('recurrent_activation')
        recr_act_params['n_in'] = str(node.get_attr('n_out')) + ' * 2'
        recr_act_params['index'] = str(node.index) + '_rec_act'
        recr_act_config = self.recr_act_template.format(**recr_act_params)

        # Multiplication config for matrix multiplications of type Wx (reset, update and candidate states)
        mult_params_x = self._default_config_params(node)
        mult_params_x['n_in'] = node.get_attr('n_in')
        mult_params_x['n_out'] = str(node.get_attr('n_out')) + ' * 3'
        mult_params_x['product_type'] = get_backend('oneAPI').product_type(
            node.get_input_variable().type.precision, node.get_weights('weight').type.precision
        )
        mult_params_x['index'] = str(node.index) + '_x'
        mult_config_x = self.mult_x_template.format(**mult_params_x)

        # Multiplication config for matrix multiplications of type Wh (reset, update and candidate states)
        mult_params_h = self._default_config_params(node)
        mult_params_h['n_in'] = node.get_attr('n_out')
        mult_params_h['n_out'] = str(node.get_attr('n_out')) + ' * 3'
        mult_params_h['reuse_factor'] = params['recurrent_reuse_factor']
        mult_params_h['product_type'] = get_backend('oneAPI').product_type(
            node.get_input_variable().type.precision, node.get_weights('recurrent_weight').type.precision
        )
        mult_params_h['index'] = str(node.index) + '_h'
        mult_config_h = self.mult_h_template.format(**mult_params_h)

        return mult_config_x + '\n' + mult_config_h + '\n' + recr_act_config + '\n' + act_config + '\n' + gru_config


class GRUFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(GRU, include_header=recurrent_include_list)

    def format(self, node):
        params = self._default_function_params(node)
        if params['pass_initial_states'] == 'true':
            params['h_t'] = node.get_input_variable(node.inputs[1]).type.name
            params['init_state'] = node.get_input_variable(node.inputs[1]).name
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name
        params['wr'] = node.get_weights('recurrent_weight').name
        params['br'] = node.get_weights('recurrent_bias').name

        if params['pass_initial_states'] == 'true':
            template = gru_function_initial_state_template
        else:
            template = gru_function_template

        return template.format(**params)


class GRUTaskSequenceTemplate(TaskSequenceTemplate):
    def __init__(self):
        super().__init__(GRU)
        self.template = gru_task_sequence_template

    def format(self, node):
        params = self._default_function_params(node)

        return self.template.format(**params)


class GRUStreamFunctionTemplate(StreamFunctionCallTemplate):
    def __init__(self):
        super().__init__(GRU)
        self.template = gru_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name
        params['wr'] = node.get_weights('recurrent_weight').name
        params['br'] = node.get_weights('recurrent_bias').name

        return self.template.format(**params)


################################################
# LSTM Template
################################################
lstm_config_template = """struct config{index} : nnet::lstm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_timesteps = {n_timesteps};
    static const unsigned return_sequences = {return_sequences};

    typedef {accum_t.name} accum_t;
    typedef {weight_i_t.name} weight_i_t;
    typedef {bias_i_t.name} bias_i_t;
    typedef {weight_f_t.name} weight_f_t;
    typedef {bias_f_t.name} bias_f_t;
    typedef {weight_c_t.name} weight_c_t;
    typedef {bias_c_t.name} bias_c_t;
    typedef {weight_o_t.name} weight_o_t;
    typedef {bias_o_t.name} bias_o_t;
    typedef {recurrent_weight_i_t.name} recurrent_weight_i_t;
    typedef {recurrent_weight_f_t.name} recurrent_weight_f_t;
    typedef {recurrent_weight_c_t.name} recurrent_weight_c_t;
    typedef {recurrent_weight_o_t.name} recurrent_weight_o_t;
    typedef {act_t} ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;

    typedef {act_recurrent_t} ACT_CONFIG_RECURRENT_T;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::{recurrent_activation}<x_T, y_T, config_T>;

    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
}};\n"""

lstm_function_template = 'nnet::lstm<{input_t}, {output_t}, {config}>({input}, {output}, {weights});'
lstm_function_initial_state_template = (
    'nnet::lstm_init_state<{input_t}, {h_t}, {hc_t}, {output_t}, {config}>'
    '({input}, {init_state}, {init_cell}, {output}, {weights});'
)


class LSTMConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(LSTM)
        self.template = lstm_config_template
        self.act_template = activ_config_template
        self.recr_act_template = activ_config_template

    def format(self, node):
        lstm_params = self._default_config_params(node)
        lstm_params['n_in'] = node.get_attr('n_in')
        lstm_params['n_out'] = node.get_attr('n_out')
        lstm_params['n_outputs'] = node.get_attr('n_timesteps') if node.get_attr('return_sequences', False) else '1'

        lstm_params['return_sequences'] = str(node.get_attr('return_sequences')).lower()
        lstm_params['act_t'] = '{}_config{}'.format(node.get_attr('activation'), str(node.index) + '_act')
        lstm_params['act_recurrent_t'] = '{}_config{}'.format(
            node.get_attr('recurrent_activation'), str(node.index) + '_rec_act'
        )
        lstm_config = self.template.format(**lstm_params)

        act_params = self._default_config_params(node)
        act_params['type'] = node.get_attr('activation')
        act_params['n_in'] = node.get_attr('n_out')
        act_params['index'] = str(node.index) + '_act'
        act_config = self.act_template.format(**act_params)

        recr_act_params = self._default_config_params(node)
        recr_act_params['type'] = node.get_attr('recurrent_activation')
        recr_act_params['n_in'] = node.get_attr('n_out')
        recr_act_params['index'] = str(node.index) + '_rec_act'
        recr_act_config = self.recr_act_template.format(**recr_act_params)

        return act_config + '\n' + recr_act_config + '\n' + lstm_config


class LSTMFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(LSTM, include_header=recurrent_include_list)

    def format(self, node):
        params = self._default_function_params(node)

        if params['pass_initial_states'] == 'true':
            params['h_t'] = node.get_input_variable(node.inputs[1]).type.name
            params['init_state'] = node.get_input_variable(node.inputs[1]).name
            params['init_cell'] = node.get_input_variable(node.inputs[2]).name
            params['hc_t'] = node.get_input_variable(node.inputs[2]).type.name

        types = ['i', 'f', 'c', 'o']
        params['weights'] = ''
        for t in types:
            params['weights'] += f'kernel_{t}_{str(node.index)},'
        for t in types:
            params['weights'] += f'recurrent_kernel_{t}_{str(node.index)},'
        for t in types:
            params['weights'] += 'bias_{}_{}{}'.format(t, str(node.index), ',' if t != 'o' else '')

        if params['pass_initial_states'] == 'true':
            template = lstm_function_initial_state_template
        else:
            template = lstm_function_template

        return template.format(**params)


################################################
# SimpleRNN Template
################################################
simple_rnn_config_template = """struct config{index} : nnet::simpleRNN_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_outputs = {n_outputs};
    static const unsigned n_timesteps = {n_timesteps};
    static const unsigned return_sequences = {return_sequences};

    typedef {accum_t.name} accum_t;
    typedef {weight_t.name} weight_t;
    typedef {bias_t.name} bias_t;
    typedef {recurrent_weight_t.name} recurrent_weight_t;

    typedef {act_t} ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;

    typedef {act_recurrent_t} ACT_CONFIG_RECURRENT_T;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::{recurrent_activation}<x_T, y_T, config_T>;

    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
}};\n"""

simple_rnn_function_template = 'nnet::simple_rnn<{input_t}, {output_t}, {config}>({input}, {output}, {weights});'


class SimpleRNNConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SimpleRNN)
        self.template = simple_rnn_config_template
        self.act_template = activ_config_template
        self.recr_act_template = activ_config_template

    def format(self, node):
        simple_rnn_params = self._default_config_params(node)
        simple_rnn_params['n_in'] = node.get_attr('n_in')
        simple_rnn_params['n_out'] = node.get_attr('n_out')
        simple_rnn_params['n_outputs'] = node.get_attr('n_timesteps') if node.get_attr('return_sequences', False) else '1'
        simple_rnn_params['return_sequences'] = str(node.get_attr('return_sequences')).lower()
        simple_rnn_params['act_t'] = '{}_config{}'.format(node.get_attr('activation'), str(node.index) + '_act')
        simple_rnn_params['act_recurrent_t'] = '{}_config{}'.format(
            node.get_attr('recurrent_activation'), str(node.index) + '_rec_act'
        )
        simple_rnn_params['recurrent_activation'] = 'relu'

        simple_rnn_config = self.template.format(**simple_rnn_params)

        act_params = self._default_config_params(node)
        act_params['type'] = node.get_attr('activation')
        act_params['n_in'] = node.get_attr('n_out')
        act_params['index'] = str(node.index) + '_act'
        act_config = self.act_template.format(**act_params)

        recr_act_params = self._default_config_params(node)
        recr_act_params['type'] = node.get_attr('recurrent_activation')
        recr_act_params['n_in'] = node.get_attr('n_out')
        recr_act_params['index'] = str(node.index) + '_rec_act'
        recr_act_config = self.recr_act_template.format(**recr_act_params)

        return act_config + '\n' + recr_act_config + '\n' + simple_rnn_config


class SimpleRNNFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SimpleRNN, include_header=recurrent_include_list)
        self.template = simple_rnn_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['weights'] = 'w{0}, wr{0}, b{0}'.format(str(node.index))
        return self.template.format(**params)
