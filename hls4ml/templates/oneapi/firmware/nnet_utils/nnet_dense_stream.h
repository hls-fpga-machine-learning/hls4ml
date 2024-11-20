#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_types.h"

namespace nnet {

// Note:  DataPack logic removed, at least in the initial version
template <class data_pipe, class res_pipe, typename CONFIG_T>
void dense_resource_stream(typename CONFIG_T::weight_t weights, typename CONFIG_T::bias_t biases) {

    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type res;
    [[intel::fpga_register]] auto data = data_pipe::read();
    dense_resource<
        typename ExtractPipeType<data_pipe>::value_type,
        typename ExtractPipeType<res_pipe>::value_type,
        CONFIG_T>(data, res, weights, biases);
    res_pipe::write(res);
}

} // namespace nnet

#endif
