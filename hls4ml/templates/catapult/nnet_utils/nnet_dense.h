#ifndef NNET_DENSE_H_
#define NNET_DENSE_H_

#include "ac_channel.h"
#include "nnet_common.h"
#include "nnet_dense_latency.h"
#include "nnet_dense_resource.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include <math.h>

namespace nnet {

struct dense_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned strategy = latency;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
    // Product function to use
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void dense(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    //#pragma HLS inline
    if (CONFIG_T::strategy == nnet::latency) {
        dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

} // namespace nnet

#endif
