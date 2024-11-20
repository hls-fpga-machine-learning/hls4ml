#ifndef NNET_DENSE_H_
#define NNET_DENSE_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense_latency.h"
#include "nnet_dense_resource.h"
#include "nnet_function_stubs.h"
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

    template <class data_T, class res_T, class CONFIG_T> using kernel = nnet::DenseKernel<data_T, res_T, CONFIG_T>;

    // Partitioning arrays cyclically to go with roll factors?

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
    #pragma HLS INLINE
    CONFIG_T::template kernel<data_T, res_T, CONFIG_T>::dense(data, res, weights, biases);
}

template <class data_T, class res_T, typename CONFIG_T> class DenseLatency : public DenseKernel<data_T, res_T, CONFIG_T> {
  public:
    static void dense(
        data_T data[CONFIG_T::n_in],
        res_T res[CONFIG_T::n_out],
        typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
        typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
    ) {
        #pragma HLS INLINE
        dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
};

template <class data_T, class res_T, typename CONFIG_T>
class DenseResource_rf_leq_nin : public DenseKernel<data_T, res_T, CONFIG_T> {
  public:
    static void dense(
        data_T data[CONFIG_T::n_in],
        res_T res[CONFIG_T::n_out],
        typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
        typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
    ) {
        #pragma HLS INLINE
        dense_resource_rf_leq_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
};

template <class data_T, class res_T, typename CONFIG_T>
class DenseResource_rf_gt_nin_rem0 : public DenseKernel<data_T, res_T, CONFIG_T> {
  public:
    static void dense(
        data_T data[CONFIG_T::n_in],
        res_T res[CONFIG_T::n_out],
        typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
        typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
    ) {
        #pragma HLS INLINE
        dense_resource_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
};

} // namespace nnet

#endif
