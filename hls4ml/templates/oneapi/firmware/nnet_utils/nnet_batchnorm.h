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
    const data_T &data, res_T &res, const typename CONFIG_T::scale_t &scale, const typename CONFIG_T::bias_t &bias
) {
// Calcuate result
Result:
    #pragma unroll
    for (int ires = 0; ires < CONFIG_T::n_in; ires++) {
        if (CONFIG_T::n_filt == -1) {
            res[ires] =
                CONFIG_T::template product<typename data_T::value_type, typename CONFIG_T::scale_t::value_type>::product(
                    data[ires], scale[ires]
                ) +
                bias[ires];
        } else {
            int norm_index = ires % CONFIG_T::n_filt;
            res[ires] =
                CONFIG_T::template product<typename data_T::value_type, typename CONFIG_T::scale_t::value_type>::product(
                    data[ires], scale[norm_index]
                ) +
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

template <class data_T, class res_T, typename CONFIG_T>
void normalize_binary_tanh(const data_T &data, res_T &res, const typename CONFIG_T::threshold_t &threshold) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        ac_int<1, false> cache;
        auto datareg = data[ii];
        int norm_index = CONFIG_T::n_filt == -1 ? ii : ii % CONFIG_T::n_filt;
        if (datareg >= threshold[norm_index])
            cache = 1;
        else
            cache = 0;

        res[ii] = cache;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void normalize_ternary_tanh(
    const data_T &data,
    res_T &res,
    const typename CONFIG_T::threshold_hi_t &threshold_hi,
    const typename CONFIG_T::threshold_lo_t &threshold_lo
) {
    #pragma unroll
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        ac_int<2, true> cache;
        auto datareg = data[ii];
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
