#ifndef NNET_EINSUM_H_
#define NNET_EINSUM_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"

namespace nnet {

struct config_einsum {
    typedef void tpose_inp0_conf;
    typedef void tpose_inp1_conf;
    typedef void tpose_out_conf;

    // Layer Sizes
    static const unsigned n_free0;
    static const unsigned n_free1;
    static const unsigned n_contract;
    static const unsigned n_inplace;

    // Resource reuse info
    static const unsigned io_type;
    static const unsigned strategy;
    static const unsigned reuse_factor;
    static const unsigned multiplier_limit;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <typename data0_T, typename data1_T, typename res_T, typename CONFIG_T>
void einsum(const data0_T data0[CONFIG_T::tpose_inp0_conf::N], const data1_T data1[CONFIG_T::tpose_inp1_conf::N],
            res_T res[CONFIG_T::tpose_out_conf::N]) {

    #pragma HLS PIPELINE II = CONFIG_T::reuse_factor
    #pragma HLS ALLOCATION operation instances = mul limit = CONFIG_T::multiplier_limit

    data0_T tpose_i0[CONFIG_T::tpose_inp0_conf::N];
    data1_T tpose_i1[CONFIG_T::tpose_inp1_conf::N];
    res_T tpose_o[CONFIG_T::tpose_out_conf::N];

    #pragma HLS ARRAY_PARTITION variable = tpose_i0 complete
    #pragma HLS ARRAY_PARTITION variable = tpose_i1 complete
    #pragma HLS ARRAY_PARTITION variable = tpose_o complete

    nnet::transpose<data0_T, data0_T, typename CONFIG_T::tpose_inp0_conf>(data0, tpose_i0);
    nnet::transpose<data1_T, data1_T, typename CONFIG_T::tpose_inp1_conf>(data1, tpose_i1);

    // for l0 in range(L0):
    //     for i in range(I):
    //             output[(i*L0+l0)*L1:(i*L0+l0+1)*L1] = input1[i*L1*C:(i+1)*L1*C].reshape((L1,C)) @
    //             input0[(i*L0+l0)*C:(i*L0+l0+1)*C]

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    typename CONFIG_T::accum_t accum_buf;
    for (unsigned i = 0; i < I; i++) {
        #pragma HLS UNROLL
        for (unsigned l0 = 0; l0 < L0; l0++) {
            #pragma HLS UNROLL
            for (unsigned l1 = 0; l1 < L1; l1++) {
                #pragma HLS UNROLL
                accum_buf = 0;
                for (unsigned c = 0; c < C; c++) {
                    #pragma HLS UNROLL
                    data0_T a = tpose_i0[(i * L0 + l0) * C + c];
                    data1_T b = tpose_i1[i * L1 * C + l1 * C + c];
                    accum_buf += CONFIG_T::template product<data0_T, data1_T>::product(a, b);
                }
                tpose_o[(i * L0 + l0) * L1 + l1] = accum_buf;
            }
        }
    }

    nnet::transpose<res_T, res_T, typename CONFIG_T::tpose_out_conf>(tpose_o, res);
}

} // namespace nnet

#endif
