#ifndef NNET_CONV1D_STREAM_H_
#define NNET_CONV1D_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_conv_stream.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    assert(
        CONFIG_T::implementation == conv_implementation::linebuffer &&
        "Only \"linebuffer\" implementation is supported in Vitis HLS."
    );

    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    if (CONFIG_T::strategy == nnet::latency) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            compute_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
        }
    } else {
    ReadInputWidthSerial:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            compute_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
        }
    }
}

} // namespace nnet
#endif
