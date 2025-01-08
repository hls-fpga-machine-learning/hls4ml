#ifndef NNET_EINSUM_DENSE_H_
#define NNET_EINSUM_DENSE_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense_latency.h"
#include "nnet_dense_resource.h"
#include "nnet_function_stubs.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"

namespace nnet {

struct einsum_dense_config {
    // Internal data type definitions

    typedef void tpose_inp_conf;
    typedef void tpose_out_conf;
    typedef void dense_conf;

    // Layer Sizes
    static const unsigned n_free_data = 1;
    static const unsigned n_free_kernel = 1;
    static const unsigned n_contract = 1;
    static const unsigned n_inplace = 1;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned strategy = latency;
    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1000; // Only useful when n_inplace > 1
    static const bool store_weights_in_bram = false;     // NOT USED

    // Product function to use
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void einsum_dense(
    data_T data[CONFIG_T::n_free_data * CONFIG_T::n_contract * CONFIG_T::n_inplace],
    res_T res[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace],
    typename CONFIG_T::dense_conf::weight_t weights[CONFIG_T::n_free_kernel * CONFIG_T::n_contract * CONFIG_T::n_inplace],
    typename CONFIG_T::dense_conf::bias_t biases[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace]) {
    data_T inp_tpose[CONFIG_T::n_free_data * CONFIG_T::n_contract * CONFIG_T::n_inplace];
    res_T out_tpose[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace];
    res_T out_buffer[CONFIG_T::n_free_kernel];
    #pragma HLS ARRAY_PARTITION variable = inp_tpose complete
    #pragma HLS ARRAY_PARTITION variable = out_tpose complete

    nnet::transpose<data_T, data_T, typename CONFIG_T::tpose_inp_conf>(data, inp_tpose);

    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    for (unsigned l0 = 0; l0 < L0; l0++) {
        #pragma HLS UNROLL factor = CONFIG_T::parallelization_factor
        for (unsigned i = 0; i < I; i++) {
            #pragma HLS UNROLL
            // even w/o explicit distributed arithmetic optimization, latency kernels are partially implemented as such
            // so reusing the same multiplier for different weights doesn't really help... only full unrolling for now
            dense<data_T, res_T, typename CONFIG_T::dense_conf>(&inp_tpose[(i * L0 + l0) * C], out_buffer,
                                                                &weights[(i * L1 * C)], &biases[((i * L0 + l0) * L1)]);
            for (unsigned j = 0; j < L1; j++) {
                #pragma HLS UNROLL
                out_tpose[(i * L0 + l0) * L1 + j] = out_buffer[j];
            }
        }
    }

    nnet::transpose<res_T, res_T, typename CONFIG_T::tpose_out_conf>(out_tpose, res);
}

} // namespace nnet

#endif
