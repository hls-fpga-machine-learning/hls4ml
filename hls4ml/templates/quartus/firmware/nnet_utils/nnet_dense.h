#ifndef NNET_DENSE_LARGE_H_
#define NNET_DENSE_LARGE_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"

namespace nnet {

struct dense_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    static const unsigned reuse_factor = 1;
    static const unsigned block_factor = 1;      // DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    static const unsigned multiplier_limit = 1;  // DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor)
    static const unsigned multiplier_factor = 1; // min n_in, rf
    static const unsigned multiplier_scale = 1;  // M_LIMIT/CONFIG_T::n_out;
    static const unsigned reciprocal = 1;        // 2^35 / 25
    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?

    // Default multiplication
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void dense_rf_gt(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    const typename CONFIG_T::weight_t weights[CONFIG_T::reuse_factor_rounded * CONFIG_T::block_factor_rounded],
    const typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    assert(
        (CONFIG_T::multiplier_limit % CONFIG_T::n_out == 0 || CONFIG_T::reuse_factor >= CONFIG_T::n_in) &&
        "The current Reuse Factor is not allowed"
    );
    assert((CONFIG_T::reuse_factor > CONFIG_T::n_in) && "This function is correct only for RF > N_IN");
    //#pragma ii CONFIG_T::reuse_factor
    hls_register typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
Load:
    #pragma unroll
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }
    hls_register int out_index[CONFIG_T::reuse_factor][CONFIG_T::block_factor];
    hls_register int d_index[CONFIG_T::reuse_factor][CONFIG_T::block_factor];

    #pragma unroll
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        #pragma unroll
        for (int im = 0; im < CONFIG_T::block_factor; im++) {
            uint32 w_index = ir + CONFIG_T::reuse_factor * im;
            out_index[ir][im] = (w_index / CONFIG_T::multiplier_factor).to_int();
            d_index[ir][im] = w_index % CONFIG_T::n_in;
        }
    }
Product1:
    #pragma nofusion
    #pragma speculated_iterations 0
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        hls_register typename CONFIG_T::accum_t tmp_acc[CONFIG_T::block_factor];
    Product2:
        #pragma unroll
        for (int im = 0; im < CONFIG_T::block_factor; im++) {
            uint32 w_index = ir + (CONFIG_T::reuse_factor_rounded)*im;
            if (w_index >= CONFIG_T::reuse_factor_rounded * CONFIG_T::block_factor_rounded)
                continue;
            int data_index = d_index[ir][im];
            // Modified this
            tmp_acc[im] =
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[data_index], weights[w_index]);
        }
        hls_register typename CONFIG_T::accum_t mult[CONFIG_T::multiplier_limit];
    ResetMult:
        #pragma unroll
        for (int imult = 0; imult < CONFIG_T::multiplier_limit; imult++) {
            mult[imult] = 0;
        }
    AccumLoop1:
        #pragma unroll
        for (int im = 0; im < CONFIG_T::block_factor; im++) {
            int o_index = out_index[ir][im];
            if (o_index >= CONFIG_T::n_out)
                continue; // check out of bounds
            mult[o_index] += tmp_acc[im];
        }
    AccumLoop2:
        #pragma unroll
        for (int im = 0; im < CONFIG_T::multiplier_limit; im++) {
            acc[im] += mult[im];
        }
    }
Store:
    #pragma unroll
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]); // acc[jj];
    }
}
template <class data_T, class res_T, typename CONFIG_T>
void dense_rf_lt(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    const typename CONFIG_T::weight_t weights[CONFIG_T::reuse_factor_rounded * CONFIG_T::block_factor_rounded],
    const typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    assert(
        (CONFIG_T::multiplier_limit % CONFIG_T::n_out == 0 || CONFIG_T::reuse_factor >= CONFIG_T::n_in) &&
        "The current Reuse Factor is not allowed"
    );
    assert((CONFIG_T::multiplier_limit == CONFIG_T::block_factor) && "This function is correct only for RF <= N_IN");

    hls_register typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
InitAccum:
    #pragma unroll
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }
ReuseLoop:
    #pragma nofusion
    #pragma speculated_iterations 0
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        hls_register typename CONFIG_T::accum_t mult[CONFIG_T::block_factor];
    MultLoop:
        #pragma unroll
        for (int im = 0, in_index = ir; im < CONFIG_T::block_factor; im++) {
            uint32 w_index = ir + (CONFIG_T::reuse_factor_rounded)*im;
            if (ir + CONFIG_T::reuse_factor * im >= CONFIG_T::n_in * CONFIG_T::n_out)
                continue;
            // Modified this
            mult[im] =
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]);
            in_index += CONFIG_T::reuse_factor;
            if (in_index >= CONFIG_T::n_in)
                in_index = ir;
        }
    AccumLoop:
        #pragma unroll
        for (int im = 0, out_index = 0, acc_step = 0; im < CONFIG_T::block_factor; im++) {
            acc[out_index] += mult[im];
            if (acc_step + 1 >= CONFIG_T::multiplier_scale) {
                acc_step = 0;
                out_index++;
            } else {
                acc_step++;
            }
        }
    }
// Cast to "res_t" type
Result:
    #pragma unroll
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}
template <class data_T, class res_T, typename CONFIG_T>
void dense_resource(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    const typename CONFIG_T::weight_t weights[CONFIG_T::reuse_factor_rounded * CONFIG_T::block_factor_rounded],
    const typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        dense_rf_lt<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_rf_gt<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}
} // namespace nnet
#endif
