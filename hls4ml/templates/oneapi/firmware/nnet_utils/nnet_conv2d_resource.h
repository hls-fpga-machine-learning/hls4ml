#ifndef NNET_CONV2D_RESOURCE_H_
#define NNET_CONV2D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"

namespace nnet {

enum class conv2d_implementation { combination, im2col, winograd };

// ****************************************************************
//      im2col - General-purpose 2D Convolution algorithm
// ****************************************************************

template <class data_T, class data_col_T, typename CONFIG_T>
void im2col_2d_cl(const data_T &data, data_col_T &data_col, const int row, const int col) {
    // im2col can be unrolled fully, since number of parallel executions = filt_h x filt_w x n_chann ~ O(100) and very little
    // DSP usage

    [[intel::fpga_register]] int index = 0;

FiltHeightLoop:
    #pragma unroll
    for (int kernel_row = 0; kernel_row < CONFIG_T::impl_filt_height; kernel_row++) {
        [[intel::fpga_register]] int input_row =
            -CONFIG_T::pad_top + kernel_row * CONFIG_T::dilation_height + row * CONFIG_T::stride_height;

    FiltWidthLoop:
        #pragma unroll
        for (int kernel_col = 0; kernel_col < CONFIG_T::impl_filt_width; kernel_col++) {
            [[intel::fpga_register]] int input_col =
                -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation_width + col * CONFIG_T::stride_width;

        ChannelLoop:
            #pragma unroll
            for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
                if (input_row >= 0 && input_row < CONFIG_T::in_height && input_col >= 0 && input_col < CONFIG_T::in_width) {
                    data_col[index++] =
                        data[input_row * CONFIG_T::in_width * CONFIG_T::n_chan + input_col * CONFIG_T::n_chan + channel];
                } else {
                    data_col[index++] = 0;
                }
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_im2col_cl(
    const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights, const typename CONFIG_T::bias_t &biases
) {
    // im2col performs no filter transformations; therefore, filter size remains constant
    assert(CONFIG_T::filt_height == CONFIG_T::impl_filt_height && CONFIG_T::filt_width == CONFIG_T::impl_filt_width);

    // Unroll factors for loop traversing input image, derived from parallelization_factor
    // Outer loop only gets unrolled after inner loop is fully unrolled
    static constexpr int pfc = MIN(CONFIG_T::parallelization_factor, CONFIG_T::out_width);
    static constexpr int pfr = MIN((CONFIG_T::parallelization_factor / pfc), CONFIG_T::out_height);

    using data_col_T =
        array<typename data_T::value_type, CONFIG_T::impl_filt_height * CONFIG_T::impl_filt_width * CONFIG_T::n_chan>;
    using res_col_T = array<typename res_T::value_type, CONFIG_T::n_filt>;

HeightLoop:
    #pragma unroll pfr
    for (int i = 0; i < CONFIG_T::out_height; i++) {
    WidthLoop:
        #pragma unroll pfc
        [[intel::initiation_interval(CONFIG_T::reuse_factor)]] for (int j = 0; j < CONFIG_T::out_width; j++) {
            // Loop variables should always be declared in the deepest scope available
            // See Intel's HLS - Loop Best Practices
            // https://www.intel.com/content/www/us/en/docs/programmable/683152/22-2/declare-variables-in-the-deepest-scope.html

            [[intel::fpga_register]] data_col_T data_col;
            im2col_2d_cl<data_T, data_col_T, CONFIG_T>(data, data_col, i, j);

            [[intel::fpga_register]] res_col_T res_col;
            dense_resource<data_col_T, res_col_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);

        // Unroll fully, since
        // (1) n_filt is usually low in io_parallel (< 32)
        // (2) no complex operations handled in loop, this loop performs a simple register writing operation
        FiltLoop:
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_filt; k++) {
                res[i * CONFIG_T::out_width * CONFIG_T::n_filt + j * CONFIG_T::n_filt + k] = res_col[k];
            }
        }
    }
}

// ****************************************************************
//       2D Convolution for 3x3 kernels from Winograd's algoirithm
// ****************************************************************

// Explicity transofrmed input (B'dB) needed for Winograd calculation, as explained by Lavin & Gray, 2015
template <typename data_T, typename res_T>
inline void winograd_transform_input_tile_3x3_kernel(const data_T I[16], res_T D[16]) {
    D[0] = I[0] - I[2] - I[8] + I[10];
    D[1] = I[1] + I[2] - I[9] - I[10];
    D[2] = -I[1] + I[2] + I[9] - I[10];
    D[3] = I[1] - I[3] - I[9] + I[11];

    D[4] = I[4] - I[6] + I[8] - I[10];
    D[5] = I[5] + I[6] + I[9] + I[10];
    D[6] = -I[5] + I[6] - I[9] + I[10];
    D[7] = I[5] - I[7] + I[9] - I[11];

    D[8] = -I[4] + I[6] + I[8] - I[10];
    D[9] = -I[5] - I[6] + I[9] + I[10];
    D[10] = I[5] - I[6] - I[9] + I[10];
    D[11] = -I[5] + I[7] + I[9] - I[11];

    D[12] = I[4] - I[6] - I[12] + I[14];
    D[13] = I[5] + I[6] - I[13] - I[14];
    D[14] = I[6] - I[5] + I[13] - I[14];
    D[15] = I[5] - I[7] - I[13] + I[15];
}

template <class data_T, class res_T, typename CONFIG_T>
void winograd_conv2d_3x3_kernel_cl(
    const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights, const typename CONFIG_T::bias_t &biases
) {
    // Ensure Winograd conditions are met
    assert(CONFIG_T::filt_height == 3 && CONFIG_T::filt_width == 3);
    assert(CONFIG_T::stride_height == 1 && CONFIG_T::stride_width == 1);
    assert(CONFIG_T::pad_left == CONFIG_T::pad_right && CONFIG_T::pad_top == CONFIG_T::pad_bottom);
    assert(CONFIG_T::out_height > 2 && CONFIG_T::out_width > 2);

    // Unroll factor for loop traversing input image, derived from parallelization_factor
    // Outer loop only gets unrolled after inner loop is fully unrolled
    static constexpr int pfc = MIN(CONFIG_T::parallelization_factor, DIV_ROUNDUP(CONFIG_T::out_width, 2));
    static constexpr int pfr = MIN((CONFIG_T::parallelization_factor / pfc), DIV_ROUNDUP(CONFIG_T::out_height, 2));

    // Initialise result to bias
    // Unroll fully, as loop performs a simple operation - assigning the outputs to a constant value
    #pragma unroll
    for (int i = 0; i < CONFIG_T::out_height * CONFIG_T::out_width; i++) {
        int offset = CONFIG_T::n_filt * i;
        #pragma unroll
        for (int f = 0; f < CONFIG_T::n_filt; f++) {
            res[offset + f] = static_cast<typename res_T::value_type>(biases[f]);
        }
    }

HeightLoop:
    #pragma unroll pfr
    for (int row = 0; row < CONFIG_T::out_height; row += 2) {
    WidthLoop:
        #pragma unroll pfc
        for (int col = 0; col < CONFIG_T::out_width; col += 2) {
        ChannelLoop:
            #pragma unroll
            for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
                // Get current 4x4 tile
                [[intel::fpga_register]] typename data_T::value_type T[16];
                [[intel::fpga_register]] typename CONFIG_T::accum_t D[16];
                [[intel::fpga_register]] uint8_t p = 0;

                #pragma unroll
                for (int r = row - (int)CONFIG_T::pad_top; r < row + 4 - (int)CONFIG_T::pad_top; r++) {
                    #pragma unroll
                    for (int c = col - (int)CONFIG_T::pad_left; c < col + 4 - (int)CONFIG_T::pad_left; c++) {
                        if (r < CONFIG_T::in_height && r >= 0 && c < CONFIG_T::in_width && c >= 0) {
                            T[p++] = data[r * CONFIG_T::in_width * CONFIG_T::n_chan + c * CONFIG_T::n_chan + channel];
                        } else {
                            T[p++] = 0;
                        }
                    }
                }

                // Transform input tile
                winograd_transform_input_tile_3x3_kernel<typename data_T::value_type, typename CONFIG_T::accum_t>(T, D);

                #pragma unroll
                for (int filter = 0; filter < CONFIG_T::n_filt; filter++) {
                    [[intel::fpga_register]] int filter_offset = 16 * (CONFIG_T::n_chan * filter + channel);

                    // Hadamard product between transformed input tile and kernel
                    [[intel::fpga_register]] typename CONFIG_T::accum_t Y[16];
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        Y[i] = static_cast<typename CONFIG_T::accum_t>(D[i] * weights[filter_offset + i]);
                    }

                    // Explicitly transform intermediate result Z = A'YA and save to output
                    res[CONFIG_T::n_filt * (row * CONFIG_T::out_width + col) + filter] +=
                        static_cast<typename res_T::value_type>(
                            Y[0] + Y[1] + Y[2] + Y[4] + Y[5] + Y[6] + Y[8] + Y[9] + Y[10]
                        );
                    if ((col + 1) < CONFIG_T::out_height)
                        res[CONFIG_T::n_filt * (row * CONFIG_T::out_width + (col + 1)) + filter] +=
                            static_cast<typename res_T::value_type>(
                                Y[1] - Y[2] - Y[3] + Y[5] - Y[6] - Y[7] + Y[9] - Y[10] - Y[11]
                            );
                    if ((row + 1) < CONFIG_T::out_width)
                        res[CONFIG_T::n_filt * ((row + 1) * CONFIG_T::out_width + col) + filter] +=
                            static_cast<typename res_T::value_type>(
                                Y[4] + Y[5] + Y[6] - Y[8] - Y[9] - Y[10] - Y[12] - Y[13] - Y[14]
                            );
                    if ((row + 1) < (CONFIG_T::out_width) && (col + 1) < CONFIG_T::out_height)
                        res[CONFIG_T::n_filt * ((row + 1) * CONFIG_T::out_width + (col + 1)) + filter] +=
                            static_cast<typename res_T::value_type>(
                                Y[5] - Y[6] - Y[7] - Y[9] + Y[10] + Y[11] + Y[15] - Y[13] + Y[14]
                            );
                }
            }
        }
    }
}

// ****************************************************************
//       2D Convolution for 1x1 kernels using optimized im2col
// ****************************************************************

template <class data_T, class data_col_T, typename CONFIG_T>
void im2col_2d_pointwise_cl(const data_T &data, data_col_T &data_col, const int row, const int col) {
    // pointwise_im2col can be unrolled fully, only one loop with n_chan iterations

    [[intel::fpga_register]] int index = 0;

ChannelLoop:
    #pragma unroll
    for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {

        [[intel::fpga_register]] int input_row = -CONFIG_T::pad_top + row * CONFIG_T::stride_height;
        [[intel::fpga_register]] int input_col = -CONFIG_T::pad_left + col * CONFIG_T::stride_width;

        if (input_row >= 0 && input_row < CONFIG_T::in_height && input_col >= 0 && input_col < CONFIG_T::in_width) {
            data_col[index++] =
                data[input_row * CONFIG_T::in_width * CONFIG_T::n_chan + input_col * CONFIG_T::n_chan + channel];
        } else {
            data_col[index++] = 0;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_2d_resource_cl(
    const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights, const typename CONFIG_T::bias_t &biases
) {
    assert(CONFIG_T::filt_height == 1 && CONFIG_T::filt_width == 1);

    // Unroll factors for loop traversing input image, derived from parallelization_factor
    // Outer loop only gets unrolled after inner loop is fully unrolled
    static constexpr int pfc = MIN(CONFIG_T::parallelization_factor, CONFIG_T::out_width);
    static constexpr int pfr = MIN((CONFIG_T::parallelization_factor / pfc), CONFIG_T::out_height);

    using data_col_T = array<typename data_T::value_type, CONFIG_T::n_chan>;
    using res_col_T = array<typename res_T::value_type, CONFIG_T::n_filt>;

HeightLoop:
    #pragma unroll pfr
    for (int row = 0; row < CONFIG_T::out_height; row++) {
    WidthLoop:
        #pragma unroll pfc
        [[intel::initiation_interval(CONFIG_T::reuse_factor)]] for (int col = 0; col < CONFIG_T::out_width; col++) {
            // Loop variables should always be declared in the deepest scope available
            // See Intel's HLS - Loop Best Practices
            // https://www.intel.com/content/www/us/en/docs/programmable/683152/22-2/declare-variables-in-the-deepest-scope.html

            [[intel::fpga_register]] data_col_T data_col;
            im2col_2d_pointwise_cl<data_T, data_col_T, CONFIG_T>(data, data_col, row, col);

            [[intel::fpga_register]] res_col_T res_col;
            dense_resource<data_col_T, res_col_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);

        FiltLoop:
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_filt; k++) {
                res[row * CONFIG_T::out_width * CONFIG_T::n_filt + col * CONFIG_T::n_filt + k] = res_col[k];
            }
        }
    }
}

// ****************************************************************
//      Top-level function - handles different implementations
// ****************************************************************
template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_resource_cl(
    const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights, const typename CONFIG_T::bias_t &biases
) {
    static constexpr bool winograd_conditions =
        // Winograd's minimal filtering algorithm not applicable to stride != 1
        CONFIG_T::stride_height == 1 && CONFIG_T::stride_width == 1 &&

        // Intel HLS will fail to pipeline the entire component if the Winograd loop only runs once
        CONFIG_T::out_height > 2 && CONFIG_T::out_width > 2 &&

        // Verify user opted for Winograd
        (CONFIG_T::implementation == nnet::conv2d_implementation::combination ||
         CONFIG_T::implementation == nnet::conv2d_implementation::winograd);

    if (CONFIG_T::filt_height == 3 && CONFIG_T::filt_width == 3 && winograd_conditions) {
        winograd_conv2d_3x3_kernel_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        conv_2d_im2col_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

} // namespace nnet

#endif
