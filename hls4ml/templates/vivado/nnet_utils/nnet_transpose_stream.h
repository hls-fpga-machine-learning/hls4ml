#ifndef NNET_TRANSPOSE_STREAM_H
#define NNET_TRANSPOSE_STREAM_H

#include "hls_stream.h"
#include "nnet_transpose.h"
#include <type_traits>

namespace nnet {

template <typename data_T, typename res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::dims == 2, void>::type transpose(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // #pragma HLS INLINE RECURSIVE
    typename data_T::value_type data_array[CONFIG_T::N];
    #pragma HLS ARRAY_PARTITION variable=data_array complete

    for (int i = 0; i < CONFIG_T::N / data_T::size; i++) {
        #pragma HLS PIPELINE
        data_T in_data = data.read();
        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            data_array[i * data_T::size + j] = typename data_T::value_type(in_data[j]);
        }
    }

    for (int i = 0; i < CONFIG_T::N / res_T::size; i++) {
        #pragma HLS PIPELINE
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = typename res_T::value_type(data_array[j * CONFIG_T::from_shape[1] + i]);
        }
        res.write(out_data);
    }
}

// This sfinae is for vivado_hls, which has some overhead using the transfer_idx in io_stream.
// In vitis both performs exactly the same, thus this is not removed out of convenience.
template <typename data_T, typename res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::dims != 2, void>::type transpose(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // #pragma HLS INLINE RECURSIVE
    typename data_T::value_type data_array[CONFIG_T::N];
    #pragma HLS ARRAY_PARTITION variable=data_array complete

    for (int i = 0; i < CONFIG_T::N / data_T::size; i++) {
        #pragma HLS PIPELINE
        data_T in_data = data.read();
        for (int j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            data_array[i * data_T::size + j] = typename data_T::value_type(in_data[j]);
        }
    }

    for (int i = 0; i < CONFIG_T::N / res_T::size; i++) {
        #pragma HLS PIPELINE
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = typename res_T::value_type(data_array[transfer_idx<CONFIG_T>(i * res_T::size + j)]);
        }
        res.write(out_data);
    }
}

} // namespace nnet
#endif
