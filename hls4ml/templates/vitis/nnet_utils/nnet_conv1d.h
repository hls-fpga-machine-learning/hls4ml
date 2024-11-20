#ifndef NNET_CONV1D_H_
#define NNET_CONV1D_H_

#include "nnet_common.h"
#include "nnet_conv1d_latency.h"
#include "nnet_conv1d_resource.h"
#include <cstdlib>

namespace nnet {

struct conv1d_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Convolutional parameters
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 0;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 1;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 10; //(N_IN + PAD_LEFT * PAD_RIGHT - (DILATION * (FILT_WIDTH - 1) + 1)) / STRIDE + 1

    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0; // not used yet
};

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    // Inlining helps reduce latency, but may also cause timing issues in some cases, use carefully.
    //#pragma HLS INLINE recursive

    if (CONFIG_T::strategy == nnet::latency) {
        conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        conv_1d_resource_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    assert(CONFIG_T::filt_width == 1);

    // Inlining helps reduce latency, but may also cause timing issues in some cases, use carefully.
    //#pragma HLS INLINE recursive

    // Nothing special to be done for io_parallel implementation
    if (CONFIG_T::strategy == nnet::latency) {
        conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        conv_1d_resource_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

} // namespace nnet

#endif
