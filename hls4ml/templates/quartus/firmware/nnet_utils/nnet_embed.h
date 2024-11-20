#ifndef NNET_EMBED_H_
#define NNET_EMBED_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

namespace nnet {

struct embed_config {
    // Internal data type definitions
    typedef float embeddings_t;

    // (Default layer sizes, overwritten form the backend
    static const unsigned n_in = 10;
    static const unsigned n_out = 16;
    static const unsigned vocab_size = 50;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
};

template <class data_T, class res_T, typename CONFIG_T>
void embedding(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::embeddings_t embeddings[CONFIG_T::vocab_size * CONFIG_T::n_out]
) {

    /*
     * Can store embeddings[] in a register, but a large multiiplexer
     * is created due to a non-constant access pattern
     */

InputSequence:
    #pragma ii CONFIG_T::reuse_factor
    #pragma unroll
    for (int j = 0; j < CONFIG_T::n_in; j++) {
    DenseEmbedding:
        #pragma unroll
        for (int i = 0; i < CONFIG_T::n_out; i++) {
            res[j * CONFIG_T::n_out + i] = embeddings[data[j].to_uint() * CONFIG_T::n_out + i];
        }
    }
}

} // namespace nnet
#endif
