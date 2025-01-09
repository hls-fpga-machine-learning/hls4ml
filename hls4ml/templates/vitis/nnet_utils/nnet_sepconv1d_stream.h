#ifndef NNET_SEPARABLE_CONV1D_STREAM_H_
#define NNET_SEPARABLE_CONV1D_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_conv1d_stream.h"
#include "nnet_sepconv_stream.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_1d_buffer_cl(hls::stream<data_T> &data, hls::stream<res_T> &res,
                                 typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan],
                                 typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    if (CONFIG_T::strategy == nnet::latency) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            compute_depthwise_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
        }
    } else {
    ReadInputWidthSerial:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            compute_depthwise_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res,
                          typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan],
                          typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]) {
    assert(CONFIG_T::implementation == conv_implementation::linebuffer &&
           "Only \"linebuffer\" implementation is supported in Vitis HLS.");
    #pragma HLS inline recursive
    depthwise_conv_1d_buffer_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res,
                          typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                          typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_width == 1);

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    if (CONFIG_T::strategy == nnet::latency) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            if (i_iw % CONFIG_T::stride_width == 0) {
                pointwise_mult_buffer<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
            } else {
                data.read();
            }
        }
    } else {
    ReadInputWidthSerial:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            if (i_iw % CONFIG_T::stride_width == 0) {
                pointwise_mult_buffer<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
            } else {
                data.read();
            }
        }
    }
}

template <class data_T, class dw_res_T, class res_T, typename CONFIG_T>
void separable_conv_1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res,
                          typename CONFIG_T::depthwise_config::weight_t
                              depthwise_weights[CONFIG_T::depthwise_config::filt_width * CONFIG_T::depthwise_config::n_chan],
                          typename CONFIG_T::pointwise_config::weight_t
                              pointwise_weights[CONFIG_T::pointwise_config::n_chan * CONFIG_T::pointwise_config::n_filt],
                          typename CONFIG_T::depthwise_config::bias_t depthwise_biases[CONFIG_T::depthwise_config::n_chan],
                          typename CONFIG_T::pointwise_config::bias_t pointwise_biases[CONFIG_T::pointwise_config::n_filt]) {
    assert(CONFIG_T::depthwise_config::implementation == conv_implementation::linebuffer &&
           "Only \"linebuffer\" implementation is supported in Vitis HLS.");
    assert(CONFIG_T::pointwise_config::implementation == conv_implementation::linebuffer &&
           "Only \"linebuffer\" implementation is supported in Vitis HLS.");

    #pragma HLS DATAFLOW

    hls::stream<dw_res_T> depthwise_res;
    const unsigned res_depth = CONFIG_T::depthwise_config::out_width;
    #pragma HLS STREAM variable=depthwise_res depth=res_depth

    depthwise_conv_1d_buffer_cl<data_T, dw_res_T, typename CONFIG_T::depthwise_config>(data, depthwise_res,
                                                                                       depthwise_weights, depthwise_biases);
    pointwise_conv_1d_cl<dw_res_T, res_T, typename CONFIG_T::pointwise_config>(depthwise_res, res, pointwise_weights,
                                                                               pointwise_biases);
}

} // namespace nnet
#endif
