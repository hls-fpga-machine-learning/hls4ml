#ifndef NNET_CONV2D_STREAM_H_
#define NNET_CONV2D_STREAM_H_

#include "nnet_dense.h"
#include "nnet_types.h"

namespace nnet {

/*
 * void kernel_shift(shift_buffer, kernel_window)
 *
 * Args:
 *   shift_buffer - array elements popped from the line the buffer during the shift line buffer operation
 *   kernel_window - array of values from the input curently being convolved with the kernel
 *
 * Values from shift_buffer are inserted into kernel_window, updating the values to be convolved
 */
template <class data_T, class data_window_T, typename CONFIG_T>
void kernel_shift_2d(
    typename data_T::value_type shift_buffer[CONFIG_T::filt_height][CONFIG_T::n_chan], data_window_T &kernel_window
) {
/*
 * Manually shift kernel_window by one step to the left
 * Not possible to use nnet::shift_reg<T, N> as the kernel window is convolved with the kernel weights using dense matrix
 * multiplication Dense matrix multiplication is only implemented for arrays However, provided certain timing constrains are
 * met, Intel HLS automatically infers a shift operation and implements kernel_window as a shift register To verify, see
 * synthesis report in report.html > Area Analysis of System
 */
KernelShiftWidth:
    #pragma unroll
    for (int col = 0; col < CONFIG_T::filt_width - 1; col++) {
    KernelShiftHeight:
        #pragma unroll
        for (int row = 0; row < CONFIG_T::filt_height; row++) {
        KernelShiftChannel:
            #pragma unroll
            for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
                kernel_window[row * CONFIG_T::filt_width * CONFIG_T::n_chan + col * CONFIG_T::n_chan + channel] =
                    kernel_window[row * CONFIG_T::filt_width * CONFIG_T::n_chan + (col + 1) * CONFIG_T::n_chan + channel];
            }
        }
    }

// Insert shift_buffer values into the last column of the kernel window
KernelPushHeight:
    #pragma unroll
    for (int col = 0; col < CONFIG_T::filt_height; col++) {
    KernelPushChannel:
        #pragma unroll
        for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
            kernel_window
                [(CONFIG_T::filt_width - 1) * CONFIG_T::n_chan + col * CONFIG_T::filt_width * CONFIG_T::n_chan + channel] =
                    shift_buffer[col][channel];
        }
    }
}

/*
 * void shift_line_buffer(in_element, line_buffer, shift_buffer)
 *
 * Args:
 *   in_element - current elements from input image, data_T type is usually nnet::array, size of array corresponds to number
 * of channels line_buffer - chained array of shift registers, one for each row of the kernel and channel shift_buffer -
 * array elements popped from the line the buffer during the shift operation
 *
 * Values from in_element are inserted into the line buffer, causing all other elements to be shifted by one
 * Popped elements are later used to update the kernel window, during the kernel_shift operation
 */
template <class data_T, typename CONFIG_T>
void shift_line_buffer_2d(
    const data_T &in_elem,
    nnet::shift_reg<typename data_T::value_type, CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right>
        line_buffer[MAX(CONFIG_T::filt_height - 1, 1)][CONFIG_T::n_chan],
    typename data_T::value_type shift_buffer[CONFIG_T::filt_height][CONFIG_T::n_chan]
) {
// For every channel, insert the incoming pixel at end of the shift buffer
UpdateBuffer:
    #pragma unroll
    for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
        shift_buffer[CONFIG_T::filt_height - 1][channel] = in_elem[channel];
    }

// Shift line buffer and save popped values to shift buffer
LineBufferDataIn:
    #pragma unroll
    for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
    LineBufferShift:
        #pragma unroll
        for (unsigned col = 1; col < CONFIG_T::filt_height; col++) {
            // Shift the line buffer, return the popped pixel
            typename data_T::value_type pop =
                line_buffer[col - 1][channel].shift(shift_buffer[CONFIG_T::filt_height - col][channel]);

            // Place popped pixed into the shift buffer, one row above
            shift_buffer[CONFIG_T::filt_height - col - 1][channel] = pop;
        }
    }
}

/*
 * void compute_output_buffer(in_element, res_stream, line_buffer, kernel_window, weights, biases)
 *
 * Args:
 *   in_element - current elements from input image, data_T type is usually nnet::array, size of array corresponds to number
 * of channels res_stream - output stream, passed by reference to allow direct writing line_buffer - chained array of shift
 * registers, one for each row of the kernel and channel kernel_window - array of values from the input curently convolved
 * with the kernel weights - Conv1D/Conv2D layer weights biases - Conv1D/Conv2D layer biases
 *
 * Function executes 4 steps:
 *   (1) Shift line buffer - updates the contents of the chained shift registers, inserting the new inputs and removing last
 * elements (2) Kernel shift - updates the elements of the kernel window, by storing the new inputs and popped elements from
 * the line buffer (3) Matrix mulitplication - performs dense matrix multiplication between the current input window and
 * kernel weights (4) Counter housekeeping - keeps track of current pixel and stride
 */
template <class data_T, class data_window_T, class res_pipe, typename CONFIG_T>
void compute_output_buffer_2d(
    const data_T &in_elem,
    nnet::shift_reg<typename data_T::value_type, CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right>
        line_buffer[MAX(CONFIG_T::filt_height - 1, 1)][CONFIG_T::n_chan],
    data_window_T &kernel_window,
    const typename CONFIG_T::weight_t &weights,
    const typename CONFIG_T::bias_t &biases,
    int &pX,
    int &pY,
    int &sX,
    int &sY
) {

    using res_T = typename ExtractPipeType<res_pipe>::value_type;

    // Thresholds
    constexpr int lShiftX = CONFIG_T::filt_width - 1;
    constexpr int lShiftY = CONFIG_T::filt_height - 1;

    // Step 1 - Shift line buffer
    [[intel::fpga_register]] typename data_T::value_type shift_buffer[CONFIG_T::filt_height][CONFIG_T::n_chan];
    nnet::shift_line_buffer_2d<data_T, CONFIG_T>(in_elem, line_buffer, shift_buffer);

    // Step 2 - Kernel shift
    nnet::kernel_shift_2d<data_T, data_window_T, CONFIG_T>(shift_buffer, kernel_window);

    // Check to see if we have a full kernel
    if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > (lShiftY - 1) && pX > (lShiftX - 1)) {
        // Step 3 - Dense matrix multiplication
        [[intel::fpga_register]] res_T res_out;
        dense_resource<data_window_T, res_T, typename CONFIG_T::mult_config>(kernel_window, res_out, weights, biases);

        // Write result to output stream
        [[intel::fpga_register]] res_T res_pack;
    CastLoop:
        #pragma unroll
        for (int channel = 0; channel < CONFIG_T::n_filt; channel++) {
            res_pack[channel] = res_out[channel];
        }
        res_pipe::write(res_pack);
    }

    // Reached end of image
    if ((pX + 1) == (CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right) &&
        (pY + 1) == (CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom)) {
        pX = 0;
        sX = 0;
        pY = 0;
        sY = 0;
        // Reached end of row
    } else if ((pX + 1) == (CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right)) {
        pX = 0;
        sX = 0;
        pY++;
        sY = ((sY - lShiftY) == 0) ? (sY - CONFIG_T::stride_height + 1) : (sY + 1);
        // Same row, same colum, therefore, move to the right
    } else {
        pX++;
        sX = ((sX - lShiftX) == 0) ? (sX - CONFIG_T::stride_width + 1) : (sX + 1);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T>
void conv_2d_cl_stream(typename CONFIG_T::weight_t weights, typename CONFIG_T::bias_t biases) {

    using data_arr_T = typename ExtractPipeType<data_pipe>::value_type;
    using data_element_T = typename data_arr_T::value_type;
    using data_window_T = array<data_element_T, CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan>;

    // Line buffer and kernel window
    [[intel::fpga_register]] nnet::shift_reg<data_element_T, CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right>
        line_buffer[MAX(CONFIG_T::filt_height - 1, 1)][CONFIG_T::n_chan];
    [[intel::fpga_register]] data_window_T kernel_window;

    // An array of length CONFIG_T::n_chan, with elements set to zero (padding for each channel)
    constexpr auto padds = zero_array<data_arr_T>();

    // move former static variables outside the function calls
    // X position pixel
    int pX = 0;
    // Y position pixel
    int pY = 0;
    // X strides
    int sX = 0;
    // Y strides
    int sY = 0;

// Padding above input image
PaddingTopHeight:
    [[intel::loop_coalesce(2)]] for (int row = 0; row < CONFIG_T::pad_top; row++) {
    PaddingTopWidth:
        for (int col = 0; col < CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right; col++) {
            compute_output_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(
                padds, line_buffer, kernel_window, weights, biases, pX, pY, sX, sY
            );
        }
    }

ReadInputHeight:
    [[intel::loop_coalesce(2)]] for (int row = 0; row < CONFIG_T::in_height; row++) {
    // Input image left-side padding
    PaddingLeftWidth:
        for (int col = 0; col < CONFIG_T::pad_left; col++) {
            compute_output_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(
                padds, line_buffer, kernel_window, weights, biases, pX, pY, sX, sY
            );
        }

    // Read input image
    ReadInputWidth:
        for (int col = 0; col < CONFIG_T::in_width; col++) {
            compute_output_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(
                data_pipe::read(), line_buffer, kernel_window, weights, biases, pX, pY, sX, sY
            );
        }

    // Input image right-side padding
    PaddingRightWidth:
        for (int col = 0; col < CONFIG_T::pad_right; col++) {
            compute_output_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(
                padds, line_buffer, kernel_window, weights, biases, pX, pY, sX, sY
            );
        }
    }

// Padding below input image
PaddingBottomHeight:
    [[intel::loop_coalesce(2)]] for (int row = 0; row < CONFIG_T::pad_bottom; row++) {
    PaddingBottomWidth:
        for (int col = 0; col < CONFIG_T::pad_left + CONFIG_T::in_width + CONFIG_T::pad_right; col++) {
            compute_output_buffer_2d<data_arr_T, data_window_T, res_pipe, CONFIG_T>(
                padds, line_buffer, kernel_window, weights, biases, pX, pY, sX, sY
            );
        }
    }
}

} // namespace nnet

#endif
