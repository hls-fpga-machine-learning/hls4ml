#ifndef NNET_LAYERNORM_H_
#define NNET_LAYERNORM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include <math.h>

#include "hls_math.h"

namespace nnet {

struct layernorm_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float scale_t;

    // Layer Sizes
    static const unsigned n_in = 20;
    static const unsigned seq_len = 4;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <typename CONFIG_T, int N_TABLE> void init_invert_sqr_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Inversion function:
    //   result = 1/sqrt(x)
    float min_val = CONFIG_T::epsilon;
    float max_val = CONFIG_T::table_range;
    float step = max_val / (float)(N_TABLE);
    for (int ii = 0; ii < N_TABLE; ii++) {
        float in_val = min_val + step * ii;
        table_out[ii] = (typename CONFIG_T::table_t)(1.0 / sqrt(in_val));
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void layernorm_1d(data_T data[CONFIG_T::n_in / CONFIG_T::seq_len], res_T res[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=data complete
    #pragma HLS ARRAY_PARTITION variable=res complete
    int inv_range_inv = (int)1 / CONFIG_T::table_range;
    typename CONFIG_T::table_t deno_inver = 0;
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t invert_sqr_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t invert_sqr_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_invert_sqr_table<CONFIG_T, CONFIG_T::table_size>(invert_sqr_table);
        initialized = true;
    }

    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    typename CONFIG_T::mean_t sum_cache = 0;
    typename CONFIG_T::mean_t sum_cache2 = 0;
    typename CONFIG_T::mean_t var, mean, diff;
    typename CONFIG_T::mean_t data_diff[dim];
    typename CONFIG_T::mean_t var_epsilon = (typename CONFIG_T::mean_t)CONFIG_T::epsilon;

    #pragma HLS ARRAY_PARTITION variable=data_diff complete

    const typename CONFIG_T::mean_t k_inv = 1.0 / dim;

LAYERNORM_1D_SUM:
    for (int i = 0; i < dim; ++i) {
        sum_cache += static_cast<typename CONFIG_T::mean_t>(data[i]);
    }
    mean = CONFIG_T::template product<typename CONFIG_T::mean_t, typename CONFIG_T::mean_t>::product(sum_cache, k_inv);

LAYERNORM_1D_VAR:
    for (int i = 0; i < dim; ++i) {
        data_diff[i] = static_cast<typename CONFIG_T::mean_t>(data[i]) - mean;
        diff = data_diff[i] * data_diff[i];
        sum_cache2 += diff;
    }
    var = CONFIG_T::template product<typename CONFIG_T::mean_t, typename CONFIG_T::mean_t>::product(sum_cache2, k_inv);

    int index = (var) * (CONFIG_T::table_size)*inv_range_inv;
    if (CONFIG_T::table_range > 1)
        index = (var) * (CONFIG_T::table_size) / (int)CONFIG_T::table_range;
    if (index < 0)
        index = 0;
    if (index > CONFIG_T::table_size - 1)
        index = CONFIG_T::table_size - 1;
    deno_inver = invert_sqr_table[index];

LAYERNORM_1D_RESULT:
    for (int i = 0; i < dim; ++i) {
        res[i] = data_diff[i] * deno_inver * scale[i] + bias[i];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void layernormalize(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in],
                    typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    data_T in_val[dim];
    res_T outval[dim];
    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=scale,bias

    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete
    #pragma HLS ARRAY_PARTITION variable=in_val complete
    #pragma HLS ARRAY_PARTITION variable=outval complete

LAYERNORM_SEQ_LOOP:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        #pragma HLS PIPELINE
    LAYERNORM_LOAD:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            in_val[i] = data[j * dim + i];
        }
        layernorm_1d<data_T, res_T, CONFIG_T>(in_val, outval, scale, bias);
    LAYERNORM_STORE:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            res[j * dim + i] = outval[i];
        }
    }
}

} // namespace nnet

#endif
