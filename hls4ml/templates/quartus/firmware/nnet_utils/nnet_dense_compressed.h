#ifndef NNET_COMPRESSED_LAYER_H_
#define NNET_COMPRESSED_LAYER_H_

#include "nnet_common.h"
#include "nnet_dense.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_compressed(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    const typename CONFIG_T::weight_t weights[CONFIG_T::n_nonzeros],
    const typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {

    hls_register typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

InitAccum:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        acc[i] = (typename CONFIG_T::accum_t)(biases[i]);
    }

    hls_register int out_index[CONFIG_T::reuse_factor][CONFIG_T::compressed_block_factor];
    hls_register data_T inputs[CONFIG_T::reuse_factor][CONFIG_T::compressed_block_factor];

    #pragma unroll
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        #pragma unroll
        for (int im = 0; im < CONFIG_T::compressed_block_factor; im++) {
            uint32 w = ir + CONFIG_T::reuse_factor * im;
            inputs[ir][im] = data[weights[w].row_index];
            out_index[ir][im] = weights[w].col_index;
        }
    }
ReuseLoop:
    #pragma nofusion
    #pragma speculated_iterations 0
    for (int ir = 0; ir < CONFIG_T::reuse_factor; ir++) {
        hls_register typename CONFIG_T::accum_t mult[CONFIG_T::compressed_block_factor];
    CompressedMultLoop:
        #pragma unroll
        for (int im = 0; im < CONFIG_T::compressed_block_factor; im++) {
            uint32 w = ir + CONFIG_T::reuse_factor * im;
            // if (w >= CONFIG_T::reuse_factor*CONFIG_T::compressed_block_factor) continue;
            typename CONFIG_T::accum_t prod = mult[im] =
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(inputs[0][im], weights[w].weight);
            #pragma unroll
            for (int is = 0; is < CONFIG_T::reuse_factor - 1; is++) {
                inputs[is][im] = inputs[is + 1][im];
            }
        }
        hls_register typename CONFIG_T::accum_t tmp_acc[CONFIG_T::n_out];
    ResetMult:
        #pragma unroll
        for (int tacc = 0; tacc < CONFIG_T::n_out; tacc++) {
            tmp_acc[tacc] = 0;
        }
    AccumLoop1:
        #pragma unroll
        for (int im = 0; im < CONFIG_T::compressed_block_factor; im++) {
            int col = out_index[ir][im];
            tmp_acc[col] += mult[im];
        }
    AccumLoop2:
        #pragma unroll
        for (int im = 0; im < CONFIG_T::n_out; im++) {
            acc[im] += tmp_acc[im];
        }
    }

// Cast to "res_t" type
ResultLoop:
    #pragma unroll
    for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
        res[i] = cast<data_T, res_T, CONFIG_T>(acc[i]);
    }
}

} // namespace nnet

#endif
