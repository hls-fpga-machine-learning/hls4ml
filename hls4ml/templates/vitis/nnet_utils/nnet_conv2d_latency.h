#ifndef NNET_CONV2D_LATENCY_H_
#define NNET_CONV2D_LATENCY_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include <cstdlib>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_latency_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    constexpr unsigned mult_n_in = CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=0

    typename CONFIG_T::accum_t mult[mult_n_in * mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=mult complete

    typename CONFIG_T::accum_t acc[mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    // Limit multipliers to control parallelization
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::mult_config::multiplier_limit

PartitionLoop:
    for (int i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

    PixelLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL

            data_T cache;

        // Do the matrix-multiply
        Product1:
            for (int i_in = 0; i_in < mult_n_in; i_in++) {
                #pragma HLS UNROLL
                cache = data_buf[i_pxl][i_in];
            Product2:
                for (int i_out = 0; i_out < mult_n_out; i_out++) {
                    #pragma HLS UNROLL
                    mult[i_in * mult_n_out + i_out] =
                        CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                            cache, weights[i_in * mult_n_out + i_out]
                        );
                }
            }

        // Initialize accumulator with input biases
        ResetAccum:
            for (int i_acc = 0; i_acc < mult_n_out; i_acc++) {
                #pragma HLS UNROLL
                acc[i_acc] = (typename CONFIG_T::accum_t)biases[i_acc];
            }

        // Accumulate multiplication result
        Accum1:
            for (int i_in = 0; i_in < mult_n_in; i_in++) {
                #pragma HLS UNROLL
            Accum2:
                for (int i_out = 0; i_out < mult_n_out; i_out++) {
                    #pragma HLS UNROLL
                    acc[i_out] += mult[i_in * mult_n_out + i_out];
                }
            }

        // Cast to "res_t" type
        Result:
            for (int i_res = 0; i_res < mult_n_out; i_res++) {
                #pragma HLS UNROLL
                res[i_part * CONFIG_T::n_pixels * mult_n_out + i_pxl * mult_n_out + i_res] =
                    cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[i_res]);
            }
        }
    }
}

} // namespace nnet
#endif
