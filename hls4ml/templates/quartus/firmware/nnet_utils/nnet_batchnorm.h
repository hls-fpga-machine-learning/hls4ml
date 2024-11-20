#ifndef NNET_BATCHNORM_H_
#define NNET_BATCHNORM_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"

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

    // Default multiplication
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void normalize(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_in],
    const typename CONFIG_T::scale_t scale[CONFIG_T::n_scale_bias],
    const typename CONFIG_T::bias_t bias[CONFIG_T::n_scale_bias]
) {
// Calcuate result
Result:
    #pragma unroll
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
    data_T data[CONFIG_T::n_in], ac_int<1, false> res[CONFIG_T::n_in], const data_T threshold[CONFIG_T::n_scale_bias]
) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        ac_int<1, false> cache;
        data_T datareg = data[ii];
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
    ac_int<2, true> res[CONFIG_T::n_in],
    const data_T threshold_hi[CONFIG_T::n_scale_bias],
    const data_T threshold_lo[CONFIG_T::n_scale_bias]
) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        ac_int<2, true> cache;
        data_T datareg = data[ii];
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
