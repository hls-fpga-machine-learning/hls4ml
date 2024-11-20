#ifndef NNET_CONV1D_RESOURCE_H_
#define NNET_CONV1D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include <cstdint>

namespace nnet {

enum class conv1d_implementation { combination, im2col, winograd };

// ****************************************************************
//      im2col - General-purpose 1D Convolution algorithm
// ****************************************************************

template <class data_T, typename CONFIG_T>
void im2col_1d_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::impl_filt_width * CONFIG_T::n_chan],
    const int col
) {
    // im2col can be unrolled fully, since number of parallel executions = filt_w x n_chann ~ O(100) and very little DSP
    // usage

    hls_register int index = 0;

KernelLoop:
    #pragma unroll
    for (int kernel_col = 0; kernel_col < CONFIG_T::impl_filt_width; kernel_col++) {
    ChannelLoop:
        #pragma unroll
        for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
            hls_register int index_data =
                (col * CONFIG_T::stride_width + kernel_col - CONFIG_T::pad_left) * CONFIG_T::n_chan + channel;
            if (index_data >= 0 && index_data < CONFIG_T::in_width * CONFIG_T::n_chan) {
                data_col[index++] = data[index_data];
            } else {
                data_col[index++] = 0;
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_im2col_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::impl_filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    // im2col performs no filter transformations; therefore, filter size remains constant
    assert(CONFIG_T::filt_width == CONFIG_T::impl_filt_width);

    // Unroll factor for loop traversing input image, derived from parallelization_factor
    static constexpr int pf = MIN(CONFIG_T::parallelization_factor, CONFIG_T::out_width);

ColLoop:
    #pragma unroll pf
    #pragma ii CONFIG_T::reuse_factor
    for (int i = 0; i < CONFIG_T::out_width; i++) {
        // Loop variables should always be declared in the deepest scope available
        // See Intel's HLS - Loop Best Practices
        // https://www.intel.com/content/www/us/en/docs/programmable/683152/22-2/declare-variables-in-the-deepest-scope.html

        hls_register data_T data_col[CONFIG_T::impl_filt_width * CONFIG_T::n_chan];
        im2col_1d_cl<data_T, CONFIG_T>(data, data_col, i);

        hls_register res_T res_col[CONFIG_T::n_filt];
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);

    // Unroll fully, since
    // (1) n_filt is usually low in io_parallel (< 32)
    // (2) no complex operations handled in loop, this loop performs a simple register writing operation
    FiltLoop:
        #pragma unroll
        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            res[i * CONFIG_T::n_filt + j] = res_col[j];
        }
    }
}

// ****************************************************************
//       1D Convolution for 3x1 kernels from Winograd's algoirithm
// ****************************************************************

// Explicity transofrmed input (B'dB) needed for Winograd convolution, as explained by Lavin & Gray (2015)
template <typename data_T, typename res_T>
inline void winograd_transform_input_tile_3x1_kernel(const data_T I[4], res_T D[4]) {
    D[0] = I[0] - I[2];
    D[1] = I[1] + I[2];
    D[2] = -I[1] + I[2];
    D[3] = I[1] - I[3];
}

template <class data_T, class res_T, typename CONFIG_T>
void winograd_conv1d_3x1_kernel_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::impl_filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    // Ensure Winograd conditions are met
    assert(CONFIG_T::filt_width == 3);
    assert(CONFIG_T::stride_width == 1);
    assert(CONFIG_T::out_width > 2);

    // Unroll factor for loop traversing input image, derived from parallelization_factor
    static constexpr int pf = MIN(CONFIG_T::parallelization_factor, CONFIG_T::out_width);

    // Initialise result to bias
    // Unroll fully, as loop performs a simple operation - assigning the outputs to a constant value
    #pragma unroll
    for (int i = 0; i < CONFIG_T::out_width; i++) {
        int offset = CONFIG_T::n_filt * i;
        #pragma unroll
        for (int f = 0; f < CONFIG_T::n_filt; f++) {
            res[offset + f] = static_cast<res_T>(biases[f]);
        }
    }

WidthLoop:
    #pragma unroll pf
    for (int col = 0; col < CONFIG_T::out_width; col += 2) {
    ChannelLoop:
        #pragma unroll
        for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
            // Get current 4x1 tile
            hls_register data_T T[16];
            hls_register uint8_t p = 0;

            #pragma unroll
            for (int c = col - (int)CONFIG_T::pad_left; c < col + 4 - (int)CONFIG_T::pad_left; c++) {
                if (c < CONFIG_T::in_width && c >= 0) {
                    T[p++] = data[c * CONFIG_T::n_chan + channel];
                } else {
                    T[p++] = 0;
                }
            }

            // Transform input tile
            hls_register typename CONFIG_T::accum_t D[4];
            winograd_transform_input_tile_3x1_kernel<data_T, typename CONFIG_T::accum_t>(T, D);

            #pragma unroll
            for (int filter = 0; filter < CONFIG_T::n_filt; filter++) {
                hls_register int filter_offset = 4 * (CONFIG_T::n_chan * filter + channel);

                // Hadamard product between transformed input tile and kernel
                hls_register typename CONFIG_T::accum_t Y[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    Y[i] = static_cast<typename CONFIG_T::accum_t>(D[i] * weights[filter_offset + i]);
                }

                // Explicitly transform intermediate result Z = A'YA and save to output
                res[CONFIG_T::n_filt * col + filter] += static_cast<res_T>(Y[0] + Y[1] + Y[2]);
                if ((col + 1) < CONFIG_T::out_width)
                    res[CONFIG_T::n_filt * (col + 1) + filter] += static_cast<res_T>(Y[1] - Y[2] - Y[3]);
            }
        }
    }
}

// ****************************************************************
//       1D Convolution for 1x1 kernels using optimized im2col
// ****************************************************************

template <class data_T, typename CONFIG_T>
void im2col_1d_pointwise_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], data_T data_col[CONFIG_T::n_chan], const int col
) {
    // pointwise_im2col can be unrolled fully, only one loop with n_chan iterations

    hls_register int index = 0;

ChannelLoop:
    #pragma unroll
    for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
        hls_register int index_data = (col * CONFIG_T::stride_width - CONFIG_T::pad_left) * CONFIG_T::n_chan + channel;
        if (index_data >= 0 && index_data < CONFIG_T::in_width * CONFIG_T::n_chan) {
            data_col[index++] = data[index_data];
        } else {
            data_col[index++] = 0;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_resource_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    assert(CONFIG_T::filt_width == 1);

    // Unroll factor for loop traversing input image, derived from parallelization_factor
    static constexpr int pf = MIN(CONFIG_T::parallelization_factor, CONFIG_T::out_width);

ColLoop:
    #pragma unroll pf
    #pragma ii CONFIG_T::reuse_factor
    for (int col = 0; col < CONFIG_T::out_width; col++) {
        // Loop variables should always be declared in the deepest scope available
        // See Intel's HLS - Loop Best Practices
        // https://www.intel.com/content/www/us/en/docs/programmable/683152/22-2/declare-variables-in-the-deepest-scope.html

        hls_register data_T data_col[CONFIG_T::n_chan];
        im2col_1d_pointwise_cl<data_T, CONFIG_T>(data, data_col, col);

        hls_register res_T res_col[CONFIG_T::n_filt];
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);

    // Unroll fully, since
    // (1) n_filt is usually low in io_parallel (< 32)
    // (2) no complex operations handled in loop, this loop performs a simple register writing operation
    FiltLoop:
        #pragma unroll
        for (int k = 0; k < CONFIG_T::n_filt; k++) {
            res[col * CONFIG_T::n_filt + k] = res_col[k];
        }
    }
}

// ****************************************************************
//      Top-level function - handles different implementations
// ****************************************************************
template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_resource_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::impl_filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    static constexpr bool winograd_conditions =
        // Winograd's minimal filtering algorithm not applicable to stride != 1
        CONFIG_T::stride_width == 1 &&

            // Intel HLS will fail to pipeline the entire component if the Winograd loop only runs once
            CONFIG_T::out_width > 2 &&

            // Verify user opted for Winograd
            CONFIG_T::implementation == nnet::conv1d_implementation::combination ||
        CONFIG_T::implementation == nnet::conv1d_implementation::winograd;

    if (CONFIG_T::filt_width == 3 && winograd_conditions) {
        winograd_conv1d_3x1_kernel_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        conv_1d_im2col_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

} // namespace nnet
#endif
