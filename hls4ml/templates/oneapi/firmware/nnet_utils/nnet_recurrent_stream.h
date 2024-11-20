#ifndef NNET_RECURRENT_STREAM_H_
#define NNET_RECURRENT_STREAM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_recurrent_activation.h"

namespace nnet {
template <class data_pipe, class res_pipe, typename CONFIG_T>
void gru_stream(
    typename CONFIG_T::weight_t weights,
    typename CONFIG_T::recurrent_weight_t recurrent_weights,
    typename CONFIG_T::bias_t bias,
    typename CONFIG_T::recurrent_bias_t recurrent_bias
) {

    using data_T = typename ExtractPipeType<data_pipe>::value_type;
    using res_T = typename ExtractPipeType<res_pipe>::value_type;
    using h_T = array<typename res_T::value_type, CONFIG_T::n_units>;

    constexpr auto datasize = std::tuple_size<data_T>{};
    constexpr auto ressize = std::tuple_size<res_T>{};

    [[intel::fpga_register]] h_T h;
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_units; i++) {
        h[i] = 0;
    }

    [[intel::fpga_register]] data_T x;

DataPropagation:
    for (int i_in = 0; i_in < CONFIG_T::n_timesteps * CONFIG_T::n_in / datasize; i_in++) {
        auto data_pack = data_pipe::read();

    DataPack:
        #pragma unroll
        for (int i_pack = 0; i_pack < datasize; i_pack++) {
            x[i_pack] = data_pack[i_pack];
        }

        nnet::gru_cell<data_T, h_T, CONFIG_T>(x, h, weights, recurrent_weights, bias, recurrent_bias);

        if (CONFIG_T::return_sequences) {
            res_T res_pack;

        ResPackRetSeq:
            #pragma unroll
            for (int i_pack = 0; i_pack < ressize; i_pack++) {
                res_pack[i_pack] = h[i_pack];
            }

            res_pipe::write(res_pack);
        }
    }

    if (!CONFIG_T::return_sequences) {
        res_T res_pack;

    ResPackNoRetSeq:
        #pragma unroll
        for (int i_pack = 0; i_pack < ressize; i_pack++) {
            res_pack[i_pack] = h[i_pack];
        }

        res_pipe::write(res_pack);
    }
}

} // namespace nnet

#endif
