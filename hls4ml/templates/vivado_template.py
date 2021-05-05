import numpy as np
import math
from bisect import bisect_left
from queue import Queue
from collections.abc import Iterable

from hls4ml.templates.templates import Backend

dense_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const bool store_weights_in_bram = false;
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    typedef {index_t} index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::{product_type}<x_T, y_T, res_T>;
}};\n"""

batchnorm_config_template = """struct config{index} : nnet::batchnorm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_filt = {n_filt};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    typedef {bias_t} bias_t;
    typedef {scale_t} scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::{product_type}<x_T, y_T, res_T>;
}};\n"""

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
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const unsigned min_width = {min_width};
    static const ap_uint<filt_width> pixels[min_width];
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    typedef {config_t} mult_config;
}};
const ap_uint<config{index}::filt_width> config{index}::pixels[] = {{{instructions}}};\n"""

conv_mult_config_template = """struct config{index}_mult : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    static const unsigned strategy = nnet::{strategy};
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::{product_type}<x_T, y_T, res_T>;
}};\n"""

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
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const unsigned min_height = {min_height};
    static const unsigned min_width = {min_width};
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef {accum_t} accum_t;
    typedef {bias_t} bias_t;
    typedef {weight_t} weight_t;
    typedef {config_t} mult_config;
}};
const ap_uint<config{index}::filt_height * config{index}::filt_width> config{index}::pixels[] = {{{instructions}}};\n"""

sepconv_config_template = """struct config{index} {{
    typedef {depthwise_config} depthwise_config;
    typedef {pointwise_config} pointwise_config;
}};\n"""

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef {table_t} table_t;
}};\n"""

softmax_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::{implementation};
    typedef {exp_table_t} exp_table_t;
    typedef {inv_table_t} inv_table_t;
}};\n"""

pooling1d_config_template = """struct config{index} : nnet::pooling1d_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_filt = {n_filt};
    static const unsigned pool_width = {pool_width};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned stride_width = {stride_width};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
    static const unsigned reuse = {reuse};
    typedef {accum_t} accum_t;
}};\n"""

pooling2d_config_template = """struct config{index} : nnet::pooling2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned pool_height = {pool_height};
    static const unsigned pool_width = {pool_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
    static const unsigned reuse = {reuse};
    typedef {accum_t} accum_t;
}};\n"""

global_pooling1d_config_template = """struct config{index} : nnet::pooling1d_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned stride = {stride};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
    typedef {accum_t} accum_t;
}};\n"""

global_pooling2d_config_template = """struct config{index} : nnet::pooling2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_filt = {n_filt};
    static const nnet::Pool_Op pool_op = nnet::{pool_op};
    static const unsigned reuse = {reuse};
    typedef {accum_t} accum_t;
}};\n"""

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

merge_config_template = """struct config{index} : nnet::merge_config {{
    static const unsigned n_elem = {n_elem};
}};\n"""

dot_config_template = """struct config{index} : nnet::dot_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned reuse_factor = {reuse};
    typedef {accum_t} accum_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::{product_type}<x_T, y_T, res_T>;
}};\n"""

concat_config_template = """struct config{index} : nnet::concat_config {{
    static const unsigned n_elem1_0 = {n_elem1_0};
    static const unsigned n_elem1_1 = {n_elem1_1};
    static const unsigned n_elem1_2 = {n_elem1_2};
    static const unsigned n_elem2_0 = {n_elem2_0};
    static const unsigned n_elem2_1 = {n_elem2_1};
    static const unsigned n_elem2_2 = {n_elem2_2};

    static const unsigned axis = {axis};
}};\n"""

resize_config_template = """struct config{index} : nnet::resize_config {{
    static const unsigned height = {in_height};
    static const unsigned width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned new_height = {out_height};
    static const unsigned new_width = {out_width};
}};\n"""

transpose_config_template = """struct config{index} : nnet::transpose_config {{
    static const unsigned depth = {depth};
    static const unsigned height = {height};
    static const unsigned width = {width};
    static const unsigned perm[3] = {{{perm_str}}};
}};\n"""

garnet_common_config_template = """
    static const unsigned n_vertices = {n_vertices};
    static const unsigned n_vertices_width = {n_vertices_width};
    static const unsigned n_in_features = {n_in_features};
    static const unsigned distance_width = {distance_width};
    static const unsigned output_collapse = {collapse_type};
    static const bool mean_by_nvert = {mean_by_nvert};

    typedef {norm_t} norm_t;
    typedef ap_fixed<{distance_width}, {distance_nint}, AP_TRN, AP_SAT> distance_t;
    typedef {edge_weight_t} edge_weight_t;
    typedef {edge_weight_aggr_t} edge_weight_aggr_t;
    typedef {aggr_t} aggr_t;
    typedef {output_t} output_t;

    static const unsigned reuse_factor = {reuse};
    static const unsigned log2_reuse_factor = {log2_reuse};
"""

garnet_config_template = """struct config{index} : nnet::garnet_config {{"""
garnet_config_template += garnet_common_config_template
garnet_config_template += """
    static const unsigned n_propagate = {n_propagate};
    static const unsigned n_aggregators = {n_aggregators};
    static const unsigned n_out_features = {n_out_features};

    typedef {input_transform_weights_t} input_transform_weights_t;
    typedef {input_transform_biases_t} input_transform_biases_t;
    typedef {aggregator_distance_weights_t} aggregator_distance_weights_t;
    typedef {aggregator_distance_biases_t} aggregator_distance_biases_t;
    typedef {output_transform_weights_t} output_transform_weights_t;
    typedef {output_transform_biases_t} output_transform_biases_t;

    static const input_transform_weights_t (&input_transform_weights)[{input_transform_weights_size}];
    static const input_transform_biases_t (&input_transform_biases)[{input_transform_biases_size}];
    static const aggregator_distance_weights_t (&aggregator_distance_weights)[{aggregator_distance_weights_size}];
    static const aggregator_distance_biases_t (&aggregator_distance_biases)[{aggregator_distance_biases_size}];
    static const output_transform_weights_t (&output_transform_weights)[{output_transform_weights_size}];
    static const output_transform_biases_t (&output_transform_biases)[{output_transform_biases_size}];

    typedef config{index} base_t;
}};

const config{index}::input_transform_weights_t (&config{index}::input_transform_weights)[{input_transform_weights_size}] = {input_transform_weights};
const config{index}::input_transform_biases_t (&config{index}::input_transform_biases)[{input_transform_biases_size}] = {input_transform_biases};
const config{index}::aggregator_distance_weights_t (&config{index}::aggregator_distance_weights)[{aggregator_distance_weights_size}] = {aggregator_distance_weights};
const config{index}::aggregator_distance_biases_t (&config{index}::aggregator_distance_biases)[{aggregator_distance_biases_size}] = {aggregator_distance_biases};
const config{index}::output_transform_weights_t (&config{index}::output_transform_weights)[{output_transform_weights_size}] = {output_transform_weights};
const config{index}::output_transform_biases_t (&config{index}::output_transform_biases)[{output_transform_biases_size}] = {output_transform_biases};
"""

garnet_stack_base_config_template = """struct config{index}_base : nnet::garnet_config {{"""
garnet_stack_base_config_template += garnet_common_config_template
garnet_stack_base_config_template += """
    static const bool is_stack = true;

    typedef config{index}_base base_t;
}};

struct config{index} : config{index}_base {{
    static const unsigned n_sublayers = {n_sublayers};

    template<int L>
    struct sublayer_t : config{index}_base {{}};
}};

{sublayer_configs}
"""

garnet_stack_sublayer_config_template = """template<>
struct config{index}::sublayer_t<{il}> : config{index}_base {{
    static const unsigned n_in_features = {n_in_features};
    static const unsigned n_propagate = {n_propagate};
    static const unsigned n_aggregators = {n_aggregators};
    static const unsigned n_out_features = {n_out_features};

    typedef {input_transform_weights_t} input_transform_weights_t;
    typedef {input_transform_biases_t} input_transform_biases_t;
    typedef {aggregator_distance_weights_t} aggregator_distance_weights_t;
    typedef {aggregator_distance_biases_t} aggregator_distance_biases_t;
    typedef {output_transform_biases_t} output_transform_biases_t;

    static const input_transform_weights_t (&input_transform_weights)[{input_transform_weights_size}];
    static const input_transform_biases_t (&input_transform_biases)[{input_transform_biases_size}];
    static const aggregator_distance_weights_t (&aggregator_distance_weights)[{aggregator_distance_weights_size}];
    static const aggregator_distance_biases_t (&aggregator_distance_biases)[{aggregator_distance_biases_size}];
    static const output_transform_biases_t (&output_transform_biases)[{output_transform_biases_size}];

    typedef config{index}::sublayer_t<{next}> next_layer_t;
}};

const config{index}::sublayer_t<{il}>::input_transform_weights_t (&config{index}::sublayer_t<{il}>::input_transform_weights)[{input_transform_weights_size}] = {input_transform_weights};
const config{index}::sublayer_t<{il}>::input_transform_biases_t (&config{index}::sublayer_t<{il}>::input_transform_biases)[{input_transform_biases_size}] = {input_transform_biases};
const config{index}::sublayer_t<{il}>::aggregator_distance_weights_t (&config{index}::sublayer_t<{il}>::aggregator_distance_weights)[{aggregator_distance_weights_size}] = {aggregator_distance_weights};
const config{index}::sublayer_t<{il}>::aggregator_distance_biases_t (&config{index}::sublayer_t<{il}>::aggregator_distance_biases)[{aggregator_distance_biases_size}] = {aggregator_distance_biases};
const config{index}::sublayer_t<{il}>::output_transform_biases_t (&config{index}::sublayer_t<{il}>::output_transform_biases)[{output_transform_biases_size}] = {output_transform_biases};
"""

garnet_stack_config_template = (garnet_stack_base_config_template, garnet_stack_sublayer_config_template)



dense_function_template = 'nnet::dense<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
batchnorm_function_template = 'nnet::normalize<{input_t}, {output_t}, {config}>({input}, {output}, {scale}, {bias});'
conv1d_function_template = 'nnet::conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
conv2d_function_template = 'nnet::conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
sepconv1d_function_template = 'nnet::separable_conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {d}, {p}, {z}, {b});'
sepconv2d_function_template = 'nnet::separable_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {d}, {p}, {z}, {b});'
depthconv2d_function_template = 'nnet::depthwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {output});'
param_activ_function_template = 'nnet::{activation}<{input_t}, {output_t}, {config}>({input}, {param}, {output});'
pooling1d_function_template = 'nnet::pooling1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
pooling2d_function_template = 'nnet::pooling2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
global_pooling1d_function_template = 'nnet::global_pooling1d<{input_t}, {config}>({input}, {output});'
global_pooling2d_function_template = 'nnet::global_pooling2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
zeropad1d_function_template = 'nnet::zeropad1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
zeropad2d_function_template = 'nnet::zeropad2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
merge_function_template = 'nnet::{merge}<{input1_t}, {input2_t}, {output_t}, {config}>({input1}, {input2}, {output});'
resize_function_template = 'nnet::resize_{algorithm}<{input_t}, {config}>({input}, {output});'
transpose_function_template = 'nnet::transpose{dim}<{input_t}, {config}>({input}, {output});'
garnet_function_template = 'nnet::garnet{impl}<{input_t}, {integer_input_t}, {output_t}, {config}>({input}, {nvtx}, {output});'
garnet_stack_function_template = 'nnet::garnet_stack<{input_t}, {integer_input_t}, {output_t}, {config}>({input}, {nvtx}, {output});'

dense_include_list = ['nnet_utils/nnet_dense.h', 'nnet_utils/nnet_dense_compressed.h', 'nnet_utils/nnet_dense_stream.h']
batchnorm_include_list = ['nnet_utils/nnet_batchnorm.h', 'nnet_utils/nnet_batchnorm_stream.h']
conv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_conv1d_stream.h']
conv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_conv2d_stream.h']
sepconv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_sepconv1d_stream.h']
sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_sepconv2d_stream.h']
activ_include_list = ['nnet_utils/nnet_activation.h', 'nnet_utils/nnet_activation_stream.h']
pooling_include_list = ['nnet_utils/nnet_pooling.h', 'nnet_utils/nnet_pooling_stream.h']
padding_include_list = ['nnet_utils/nnet_padding.h', 'nnet_utils/nnet_padding_stream.h']
merge_include_list = ['nnet_utils/nnet_merge.h', 'nnet_utils/nnet_merge_stream.h']
resize_include_list = ['nnet_utils/nnet_image.h', 'nnet_utils/nnet_image_stream.h']
transpose_include_list = ['nnet_utils/nnet_array.h']
garnet_include_list = ['nnet_utils/nnet_garnet.h']

class VivadoBackend(Backend):
    def __init__(self):
        super(VivadoBackend, self).__init__('Vivado')
        self.register_templates('Dense', dense_function_template, dense_config_template, dense_include_list)
        self.register_templates('BinaryDense'            , dense_function_template,       dense_config_template, dense_include_list)
        self.register_templates('BatchNormalization'     , batchnorm_function_template,   batchnorm_config_template, batchnorm_include_list)
        self.register_templates('Conv1D'                 , conv1d_function_template,      [conv1d_config_template, conv_mult_config_template], conv1d_include_list)
        self.register_templates('Conv2D'                 , conv2d_function_template,      [conv2d_config_template, conv_mult_config_template], conv2d_include_list)
        self.register_templates('Conv2DBatchnorm'        , conv2d_function_template,      [conv2d_config_template, conv_mult_config_template], conv2d_include_list)
        self.register_templates('SeparableConv1D'        , sepconv1d_function_template,   [sepconv_config_template, conv1d_config_template, conv1d_config_template, conv_mult_config_template, conv_mult_config_template], sepconv1d_include_list)
        self.register_templates('SeparableConv2D'        , sepconv2d_function_template,   [sepconv_config_template, conv2d_config_template, conv2d_config_template, conv_mult_config_template, conv_mult_config_template], sepconv2d_include_list)
        self.register_templates('DepthwiseConv2D'        , depthconv2d_function_template, [conv2d_config_template, conv_mult_config_template], sepconv2d_include_list)
        self.register_templates('Activation'             , activ_function_template,       activ_config_template, activ_include_list)
        self.register_templates('ParametrizedActivation' , param_activ_function_template, activ_config_template, activ_include_list)
        self.register_templates('PReLU'                  , param_activ_function_template, activ_config_template, activ_include_list)
        self.register_templates('Softmax'                , activ_function_template,       softmax_config_template, activ_include_list)
        self.register_templates('TernaryTanh'            , activ_function_template,       activ_config_template, activ_include_list)
        self.register_templates('Pooling1D'              , pooling1d_function_template,   pooling1d_config_template, pooling_include_list)
        self.register_templates('Pooling2D'              , pooling2d_function_template,   pooling2d_config_template, pooling_include_list)
        self.register_templates('GlobalPooling1D'        , global_pooling1d_function_template,   global_pooling1d_config_template, pooling_include_list)
        self.register_templates('GlobalPooling2D'        , global_pooling2d_function_template,   global_pooling2d_config_template, pooling_include_list)
        self.register_templates('ZeroPadding1D'          , zeropad1d_function_template,   zeropad1d_config_template, padding_include_list)
        self.register_templates('ZeroPadding2D'          , zeropad2d_function_template,   zeropad2d_config_template, padding_include_list)
        self.register_templates('Merge'                  , merge_function_template,       merge_config_template, merge_include_list)
        self.register_templates('Concatenate'            , merge_function_template,       concat_config_template, merge_include_list)
        self.register_templates('Dot'                    , merge_function_template,       dot_config_template, merge_include_list)
        self.register_templates('Resize'                 , resize_function_template,      resize_config_template, resize_include_list)
        self.register_templates('Transpose'              , transpose_function_template,   transpose_config_template, transpose_include_list)
        self.register_templates('GarNet'                 , garnet_function_template,      garnet_config_template, garnet_include_list)
        self.register_templates('GarNetStack'            , garnet_stack_function_template,garnet_stack_config_template, garnet_include_list)        
    
    def get_valid_reuse_factors(self, layer):
        n_in = 0
        n_out = 0
        if 'Dense' in layer.__class__.__name__:
            n_in = layer.get_attr('n_in')
            n_out = layer.get_attr('n_out')
        elif 'Conv1D' in layer.__class__.__name__:
            n_in = layer.get_attr('n_chan') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')
        elif 'Conv2D' in layer.__class__.__name__:
            n_in = layer.get_attr('n_chan') * layer.get_attr('filt_height') * layer.get_attr('filt_width')
            n_out = layer.get_attr('n_filt')

        max_rf = n_in * n_out
        valid_reuse_factors = []
        for rf in range(1, max_rf + 1):
            _assert = self._check_conditions(n_in, n_out, rf)
            if _assert:
                valid_reuse_factors.append(rf)
        # Avoid using RF=1
        if valid_reuse_factors[0] == 1:
            valid_reuse_factors.pop(0)
        return valid_reuse_factors

    def _check_conditions(self, n_in, n_out, rf):
        multfactor = min(n_in, rf)
        multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
        #
        # THIS ASSERTION IS FOR THE FUNCTIONAL CORRECTNESS OF THE DENSE LAYER
        #
        _assert = (((multiplier_limit % n_out) == 0) or (rf >= n_in))
        _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
        #
        # THIS ASSERTION IS FOR QoR AND EXECUTION TIME OF VIVADO HLS
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

    def set_closest_reuse_factor(self, layer):
        valid_rf = self.get_valid_reuse_factors(layer)
        chosen_rf = layer.reuse_factor
        if chosen_rf not in valid_rf:
            closest_rf = self.get_closest_reuse_factor(valid_rf, chosen_rf)
            print('WARNING: Invalid ReuseFactor={} with "Resource" strategy in layer "{}". Using ReuseFactor={} instead. Valid ReuseFactor(s): {}.'
                .format(chosen_rf, layer.name, closest_rf, ','.join(map(str, valid_rf))))
            layer.reuse_factor = closest_rf

    def convert_precision_string(self, precision):
        '''
        Convert a precision string (e.g. "ap_fixed<16,6>" to the internal IntegerPrecisionTypes etc)
        '''
        from hls4ml.model.hls_layers import IntegerPrecisionType, FixedPrecisionType
        import re
        if isinstance(precision, IntegerPrecisionType) or isinstance(precision, FixedPrecisionType):
            return precision
        bits = re.search('.+<(.+?)>', precision).group(1).split(',')
        sat_mode = None
        round_mode = None
        sat_bits = None
        if 'fixed' in precision:
            W = int(bits[0])
            I = int(bits[1])
            fields = 2
            signed = ~('u' in precision)
        elif 'int' in precision:
            W = int(bits[0])
            I = W
            fields = 1
            signed = ~('u' in precision)
        if len(bits) > fields:
            round_mode = bits[fields]
        if len(bits) > fields+1:
            sat_mode = bits[fields+1]
        if len(bits) > fields+2:
            sat_bits = int(bits[fields+2])
        if 'fixed' in precision:
            return FixedPrecisionType(W, I, signed, round_mode, sat_mode, sat_bits)
        elif 'int' in precision:
            return IntegerPrecisionType(W, signed)

    def product_type(self, data_T, weight_T):
        '''
        Helper function to determine which product implementation to use during inference
        '''
        from hls4ml.model.hls_layers import IntegerPrecisionType, FixedPrecisionType, XnorPrecisionType, ExponentPrecisionType
        assert not isinstance(data_T, ExponentPrecisionType), "Only ExponentPrecisionType (aka 'power of 2') weights are currently supported, not data."
        product = 'mult'
        if isinstance(weight_T, ExponentPrecisionType):
            product = 'weight_exponential'
        else:
            # if binary
            if isinstance(weight_T, XnorPrecisionType) and isinstance(data_T, XnorPrecisionType):
                product = 'both_binary'
            elif isinstance(weight_T, XnorPrecisionType): # data is not xnor-binary
                product = 'weight_binary'
            elif isinstance(weight_T, IntegerPrecisionType) and weight_T.width == 2 and weight_T.signed:
                product = 'weight_ternary'
            else:
                product = 'mult'
        return product

    def compute_conv1d_instructions(self, in_W, in_C, kernel_size=3, stride=1, pad=0):

        # Current limitations
        assert pad == 0

        min_W = (math.ceil(kernel_size / stride) - 1) * stride + kernel_size
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
            windows_int.append((int(''.join(str(p) for p in reversed(windows_bin[i])), 2)))
        
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

        min_H = (math.ceil(kernel_height / stride_height) - 1) * stride_height + kernel_height
        min_W = (math.ceil(kernel_width / stride_width) - 1) * stride_width + kernel_width
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
        if kernel_height == 1 and kernel_width == 1 and stride == 1 and scaled_H == in_H and scaled_W == in_W:
            return (1, 1, map(str, [1]))
        if kernel_height == 3 and kernel_width == 3 and stride == 1 and scaled_H == in_H and scaled_W == in_W:
            return (5, 5, map(str, [1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256]))
        if kernel_height == 5 and kernel_width == 5 and stride == 1 and scaled_H == in_H and scaled_W == in_W:
            return (9, 9, map(str, [1,3,7,15,31,30,28,24,16,33,99,231,495,1023,990,924,792,528,1057,3171,7399,15855,
                             32767,31710,29596,25368,16912,33825,101475,236775,507375,1048575,1014750,947100,
                             811800,541200,1082401,3247203,7576807,16236015,33554431,32472030,30307228,25977624,
                             17318416,1082400,3247200,7576800,16236000,33554400,32472000,30307200,25977600,17318400,
                             1082368,3247104,7576576,16235520,33553408,32471040,30306304,25976832,17317888,1081344,
                             3244032,7569408,16220160,33521664,32440320,30277632,25952256,17301504,1048576,3145728,
                             7340032,15728640,32505856,31457280,29360128,25165824,16777216]))

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
                windows_int.append((int(''.join(str(p) for p in reversed(windows_bin[i * min_W + j])), 2)))
        
        return (min_H, min_W, windows_int)
