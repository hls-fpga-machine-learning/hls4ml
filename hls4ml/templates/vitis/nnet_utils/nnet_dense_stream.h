#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_types.h"
#include <assert.h>
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_latency_wrapper(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_wrapper(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    dense_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

template <class data_T, typename CONFIG_T>
void data_prepare(hls::stream<data_T> &data_stream, typename data_T::value_type data[CONFIG_T::n_in]) {
    #pragma HLS INLINE

    if (CONFIG_T::n_in / data_T::size > 1) {
    DataPrepare:
        for (int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
            #pragma HLS PIPELINE
            data_T data_pack = data_stream.read();
        DataPackPipeline:
            for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
                #pragma HLS UNROLL
                data[i_in * data_T::size + i_pack] = data_pack[i_pack];
            }
        }
    } else {
        data_T data_pack = data_stream.read();
    DataPackSingle:
        for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data[i_pack] = data_pack[i_pack];
        }
    }
}

template <class res_T, typename CONFIG_T>
void res_write(typename res_T::value_type res[CONFIG_T::n_out], hls::stream<res_T> &res_stream) {
    #pragma HLS INLINE

    if (CONFIG_T::n_out / res_T::size > 1) {
    ResWrite:
        for (unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
            #pragma HLS PIPELINE
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        ResPackPipeline:
            for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = res[i_out * res_T::size + i_pack];
            }
            res_stream.write(res_pack);
        }
    } else {
        res_T res_pack;
        PRAGMA_DATA_PACK(res_pack)
    ResPackSingle:
        for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = res[i_pack];
        }
        res_stream.write(res_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense(
    hls::stream<data_T> &data_stream,
    hls::stream<res_T> &res_stream,
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    #pragma HLS INLINE recursive

    typename data_T::value_type data[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=res complete

    data_prepare<data_T, CONFIG_T>(data_stream, data);
    if (CONFIG_T::strategy == nnet::latency) {
        dense_latency_wrapper<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource_wrapper<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(
            data, res, weights, biases
        );
    }
    res_write<res_T, CONFIG_T>(res, res_stream);
}

} // namespace nnet

#endif
