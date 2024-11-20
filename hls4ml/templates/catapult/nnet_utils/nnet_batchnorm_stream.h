
#ifndef NNET_BATCHNORM_STREAM_H_
#define NNET_BATCHNORM_STREAM_H_

#include "ac_channel.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_types.h"

namespace nnet {

// ****************************************************
//       Streaming Batch Normalization
// ****************************************************

template <class data_T, class res_T, typename CONFIG_T>
void normalize(
    ac_channel<data_T> &data,
    ac_channel<res_T> &res,
    typename CONFIG_T::scale_t scale[CONFIG_T::n_scale_bias],
    typename CONFIG_T::bias_t bias[CONFIG_T::n_scale_bias]
) {
    //#pragma HLS ARRAY_PARTITION variable=scale complete
    //#pragma HLS ARRAY_PARTITION variable=bias complete

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    constexpr unsigned ii = CONFIG_T::n_in / multiplier_limit;
    (void)ii;
    CONFIG_T::template product<typename data_T::value_type, typename CONFIG_T::scale_t>::limit(multiplier_limit);

BatchNormLoop:
    for (unsigned int i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        //#pragma HLS PIPELINE II=ii

        data_T in_data = data.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

    BatchNormpack:
        for (unsigned int j = 0; j < data_T::size; j++) {
            // #pragma HLS UNROLL
            int norm_index;
            if (CONFIG_T::n_filt == -1) {
                norm_index = i * data_T::size + j;
            } else {
                norm_index = j % CONFIG_T::n_filt;
            }
            out_data[j] = CONFIG_T::template product<typename data_T::value_type, typename CONFIG_T::scale_t>::product(
                              in_data[j], scale[norm_index]
                          ) +
                          bias[norm_index];
        }

        res.write(out_data);
    }
}

// ****************************************************
//       Merged Batch Normalization and Quantized Tanh
// ****************************************************
template <class data_T, typename CONFIG_T>
void normalize_binary_tanh(
    ac_channel<data_T> &data,
    ac_channel<nnet::array<ac_int<1, false>, CONFIG_T::n_in>> &res,
    typename data_T::value_type threshold[CONFIG_T::n_in]
) {
    //#pragma HLS ARRAY_PARTITION variable=threshold complete

BinaryNormLoop:
    for (int i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        nnet::array<ac_int<1, false>, CONFIG_T::n_scale_bias> out_data;
        //#pragma HLS DATA_PACK variable=out_data

    BatchNormPack:
        for (int j = 0; j < data_T::size; j++) {
            out_data[j] = (in_data[j] > threshold[i * data_T::size + j]) ? 1 : 0;
        }

        res.write(out_data);
    }
}

template <class data_T, typename CONFIG_T>
void normalize_ternary_tanh(
    ac_channel<data_T> &data,
    ac_channel<nnet::array<ac_int<2, true>, CONFIG_T::n_in>> &res,
    typename data_T::value_type threshold_hi[CONFIG_T::n_in],
    typename data_T::value_type threshold_lo[CONFIG_T::n_in]
) {
    //#pragma HLS ARRAY_PARTITION variable=threshold_hi complete
    //#pragma HLS ARRAY_PARTITION variable=threshold_lo complete

TernaryNormLoop:
    for (int i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        //#pragma HLS PIPELINE

        data_T in_data = data.read();
        nnet::array<ac_int<2, true>, CONFIG_T::n_scale_bias> out_data;
        //#pragma HLS DATA_PACK variable=out_data

    BatchNormPack:
        for (int j = 0; j < data_T::size; j++) {

            int norm_index = i * data_T::size + j;

            if (in_data[j] > threshold_hi[norm_index]) {
                out_data[j] = 1;
            } else if (in_data[j] <= threshold_lo[norm_index]) {
                out_data[j] = -1;
            } else {
                out_data[j] = 0;
            }
        }

        res.write(out_data);
    }
}

} // namespace nnet

#endif
