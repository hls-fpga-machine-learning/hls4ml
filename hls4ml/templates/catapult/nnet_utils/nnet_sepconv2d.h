#ifndef NNET_SEPARABLE_CONV2D_H_
#define NNET_SEPARABLE_CONV2D_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_chan],
    typename CONFIG_T::weight_t depthwise_weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t depthwise_biases[CONFIG_T::n_chan]
) {
    const int in_height = CONFIG_T::in_height;
    const int in_width = CONFIG_T::in_width;
    const int n_chan = CONFIG_T::n_chan;
    const int filt_height = CONFIG_T::filt_height;
    const int filt_width = CONFIG_T::filt_width;
    const int out_height = CONFIG_T::out_height;
    const int out_width = CONFIG_T::out_width;

    //    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor; (void)ce_reuse_factor;

    //    do {

    //#pragma HLS ARRAY_PARTITION variable=res complete dim=0
    //#pragma HLS ARRAY_PARTITION variable=depthwise_biases complete dim=0
    //#pragma HLS ARRAY_PARTITION variable=depthwise_weights complete dim=0
    for (int h = 0; h < in_height - filt_height + 1; h++) {
        //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind
        for (int w = 0; w < in_width - filt_width + 1; w++) {
            //#pragma HLS UNROLL
            for (int c = 0; c < n_chan; c++) {
                //#pragma HLS UNROLL
                res_T sum = depthwise_biases[c];

                // Apply the filter
                for (int i = 0; i < filt_height; i++) {
                    //#pragma HLS UNROLL
                    for (int j = 0; j < filt_width; j++) {
                        //#pragma HLS UNROLL
                        int data_idx = (h + i) * in_width * n_chan + (w + j) * n_chan + c;
                        int weight_idx = i * filt_width * n_chan + j * n_chan + c;
                        sum += data[data_idx] * depthwise_weights[weight_idx];
                    }
                }

                int res_idx = (h * out_width * n_chan) + w * n_chan + c;
                res[res_idx] = sum;
            }
        }
    }
    //    } while (false);
}

template <class data_T, class dw_res_T, class res_T, typename CONFIG_T>
void separable_conv_2d_cl(
    data_T data
        [CONFIG_T::depthwise_config::in_height * CONFIG_T::depthwise_config::in_width * CONFIG_T::depthwise_config::n_chan],
    res_T
        res[CONFIG_T::pointwise_config::out_height * CONFIG_T::pointwise_config::out_width *
            CONFIG_T::pointwise_config::n_filt],
    typename CONFIG_T::depthwise_config::weight_t depthwise_weights
        [CONFIG_T::depthwise_config::filt_height * CONFIG_T::depthwise_config::filt_width *
         CONFIG_T::depthwise_config::n_chan],
    typename CONFIG_T::pointwise_config::weight_t
        pointwise_weights[CONFIG_T::pointwise_config::n_chan * CONFIG_T::pointwise_config::n_filt],
    typename CONFIG_T::depthwise_config::bias_t depthwise_biases[CONFIG_T::depthwise_config::n_chan],
    typename CONFIG_T::pointwise_config::bias_t pointwise_biases[CONFIG_T::pointwise_config::n_filt]
) {

    //#pragma HLS INLINE region

    dw_res_T depthwise_results
        [CONFIG_T::depthwise_config::out_height * CONFIG_T::depthwise_config::out_width *
         CONFIG_T::depthwise_config::n_chan];
    depthwise_conv_2d_cl<data_T, dw_res_T, typename CONFIG_T::depthwise_config>(
        data, depthwise_results, depthwise_weights, depthwise_biases
    );
    pointwise_conv_2d_cl<dw_res_T, res_T, typename CONFIG_T::pointwise_config>(
        depthwise_results, res, pointwise_weights, pointwise_biases
    );
}

} // namespace nnet

#endif
