#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_types.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource(
    stream<data_T> &data_stream,
    stream<res_T> &res_stream,
    const typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    hls_register typename data_T::value_type data[CONFIG_T::n_in];
    hls_register typename res_T::value_type res[CONFIG_T::n_out];

DataPrepare:
    #pragma ii 1
    for (int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        data_T data_pack = data_stream.read();
    DataPack:
        #pragma unroll
        for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    dense_resource<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

ResWrite:
    #pragma ii 1
    for (unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
        res_T res_pack;
    ResPack:
        #pragma unroll
        for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }

        res_stream.write(res_pack);
    }
}

} // namespace nnet

#endif
