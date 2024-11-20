#ifndef NNET_BATCHNORM_H_
#define NNET_BATCHNORM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include <math.h>

namespace nnet {

struct batchnorm_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float scale_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void normalize(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_in],
    typename CONFIG_T::scale_t scale[CONFIG_T::n_scale_bias],
    typename CONFIG_T::bias_t bias[CONFIG_T::n_scale_bias]
) {
    data_T cache;

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=scale,bias

    // For parallel inputs:
    //   - completely partition arrays -- target fabric
    //   - if we have an unroll factor, limit number of multipliers
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

// Calcuate result
Result:
    for (int ires = 0; ires < CONFIG_T::n_in; ires++) {
        if (CONFIG_T::n_filt == -1) {
            res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[ires]) +
                        bias[ires];
        } else {
            int norm_index = ires % CONFIG_T::n_filt;
            res[ires] =
                CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[norm_index]) +
                bias[norm_index];
        }
    }
}

// ****************************************************
//       Merged Batch Normalization and Quantized Tanh
// ****************************************************
struct batchnorm_quantized_tanh_config {
    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
};

template <class data_T, typename CONFIG_T>
void normalize_binary_tanh(
    data_T data[CONFIG_T::n_in], ap_uint<1> res[CONFIG_T::n_in], data_T threshold[CONFIG_T::n_scale_bias]
) {
    #pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=res complete

    data_T datareg;
    ap_uint<1> cache;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        int norm_index = CONFIG_T::n_filt == -1 ? ii : ii % CONFIG_T::n_filt;
        if (datareg >= threshold[norm_index])
            cache = 1;
        else
            cache = 0;

        res[ii] = cache;
    }
}

template <class data_T, typename CONFIG_T>
void normalize_ternary_tanh(
    data_T data[CONFIG_T::n_in],
    ap_int<2> res[CONFIG_T::n_in],
    data_T threshold_hi[CONFIG_T::n_scale_bias],
    data_T threshold_lo[CONFIG_T::n_scale_bias]
) {
    #pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=res complete

    data_T datareg;
    ap_int<2> cache;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        int norm_index = CONFIG_T::n_filt == -1 ? ii : ii % CONFIG_T::n_filt;
        if (datareg > threshold_hi[norm_index])
            cache = 1;
        else if (datareg <= threshold_lo[norm_index])
            cache = -1;
        else
            cache = 0;

        res[ii] = cache;
    }
}

} // namespace nnet

#endif
