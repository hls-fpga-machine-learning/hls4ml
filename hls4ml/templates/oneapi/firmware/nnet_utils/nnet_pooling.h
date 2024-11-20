#ifndef NNET_POOLING_H_
#define NNET_POOLING_H_

#include "nnet_common.h"

namespace nnet {

// Returns the maximum value from an array of size N
template <typename T, int N, typename accum_t> accum_t max(T x[N]) {
    [[intel::fpga_register]] T y = x[0];

    // Due to loop dependencies, pipelining & unrolling is not possible
    // Explictily disabling pipeline significantly reduces resource usage
    [[intel::disable_loop_pipelining]] for (int i = 1; i < N; i++) {
        if (x[i] > y)
            y = x[i];
    }

    return y;
}

// Returns the mean value of an array of size N
template <typename T, int N, typename accum_t> accum_t avg(T x[N], unsigned length) {
    [[intel::fpga_register]] accum_t y = 0;

    // Due to loop dependencies, pipelining & unrolling is not possible
    // Explictily disabling pipeline significantly reduces resource usage
    [[intel::disable_loop_pipelining]] for (int i = 0; i < N; i++) { y += x[i]; }

    y /= length;
    return y;
}

// Enumeration for pooling functions
enum Pool_Op { Max, Average };
template <typename T, int N, Pool_Op op, typename accum_t> accum_t pool_op(T x[N], unsigned length) {
    switch (op) {
    case Max:
        return max<T, N, accum_t>(x);
    case Average:
        return avg<T, N, accum_t>(x, length);
    }
}

template <typename T, int N, Pool_Op op, typename accum_t> accum_t pool_op(T (&x)[N]) {
    return pool_op<T, N, op, accum_t>(x, N);
}

/*
 * In Tensorflow, pooling ignores the value in the padded cells
 * For Avg pooling, return 0 (the divisior is modified to the area overlapping the unpadded image.)
 * For ax pooling, return the most negative value for the type.
 */
template <typename T, Pool_Op op> inline T pad_val() {
    switch (op) {
    case Max: {
        T x = 0;
        x[x.width - 1] = 1;
        return x;
    }
    case Average:
        return 0;
    }
}

struct pooling1d_config {
    // Pooling paramaters
    static const unsigned pool_width = 2;
    static const unsigned stride_width = 2;

    // I/O sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = (n_in - pool_width) / stride_width + 1;
    static const unsigned n_filt = 4;

    // Padding
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;

    // Pooling function
    static const Pool_Op pool_op = Max;
};

template <class data_T, class res_T, typename CONFIG_T> void pooling1d_cl(const data_T &data, res_T &res) {
    // Add padding and reduce input width to area covered by pooling function
    static constexpr int full_padded_width = CONFIG_T::n_in + CONFIG_T::pad_left + CONFIG_T::pad_right;
    static constexpr int restricted_padded_width = full_padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width;

FiltLoop:
    #pragma unroll
    [[intel::disable_loop_pipelining]] for (int filt = 0; filt < CONFIG_T::n_filt; filt++) {
    InputWidthLoop:
        #pragma unroll
        [[intel::disable_loop_pipelining]] for (int inp_col = 0; inp_col < restricted_padded_width;
                                                inp_col += CONFIG_T::stride_width) {
            [[intel::fpga_register]] typename data_T::value_type pool[CONFIG_T::pool_width];

            // Keep track of number of pixels in image vs padding region; needed for rescaling Average Pooling
            [[intel::fpga_register]] unsigned img_overlap = 0;

        PoolWidthLoop:
            #pragma unroll
            [[intel::disable_loop_pipelining]] for (int pool_col = 0; pool_col < CONFIG_T::stride_width; pool_col++) {
                if (inp_col + pool_col < CONFIG_T::pad_left ||
                    inp_col + pool_col >= (full_padded_width - CONFIG_T::pad_right)) {
                    // Add padding
                    pool[pool_col] = pad_val<typename data_T::value_type, CONFIG_T::pool_op>();
                    if (CONFIG_T::count_pad)
                        img_overlap++;
                } else {
                    // Current element is from input image
                    pool[pool_col] = data[(inp_col + pool_col - CONFIG_T::pad_left) * CONFIG_T::n_filt + filt];
                    img_overlap++;
                }
            }

            // Pooling operation
            res[(inp_col / CONFIG_T::stride_width) * CONFIG_T::n_filt + filt] = static_cast<typename res_T::value_type>(
                pool_op<typename data_T::value_type, CONFIG_T::pool_width, CONFIG_T::pool_op, typename CONFIG_T::accum_t>(
                    pool, img_overlap
                )
            );
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T> void global_pooling1d_cl(const data_T &data, res_T &res) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);

FiltLoop:
    #pragma unroll
    [[intel::disable_loop_pipelining]] for (int filt = 0; filt < CONFIG_T::n_filt; filt++) {
        [[intel::fpga_register]] typename data_T::value_type pool[CONFIG_T::n_in];

    InputWidthLoop:
        #pragma unroll
        [[intel::disable_loop_pipelining]] for (int col = 0; col < CONFIG_T::n_in; col++) {
            pool[col] = data[col * CONFIG_T::n_filt + filt];
        }

        res[filt] = static_cast<typename res_T::value_type>(
            pool_op<typename data_T::value_type, CONFIG_T::n_in, CONFIG_T::pool_op, typename CONFIG_T::accum_t>(pool)
        );
    }
}

struct pooling2d_config {
    // Pooling parameters
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    // I/O sizes
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_filt = 4;

    static const unsigned out_height = (in_height - pool_height) / stride_height + 1;
    static const unsigned out_width = (in_width - pool_width) / stride_width + 1;

    // Padding
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;

    // Pooling function
    static const Pool_Op pool_op = Max;
};

template <class data_T, class res_T, typename CONFIG_T> void pooling2d_cl(const data_T &data, res_T &res) {
    // Add padding and reduce input width to area covered by pooling function
    static constexpr int full_padded_width = CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right;
    static constexpr int full_padded_height = CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom;
    static constexpr int restricted_padded_width = full_padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width;
    static constexpr int restricted_padded_height = full_padded_height / CONFIG_T::stride_height * CONFIG_T::stride_height;

FiltLoop:
    #pragma unroll
    [[intel::disable_loop_pipelining]] for (int filt = 0; filt < CONFIG_T::n_filt; filt++) {
    InputHeightLoop:
        #pragma unroll
        [[intel::disable_loop_pipelining]] for (int inp_col = 0; inp_col < restricted_padded_height;
                                                inp_col += CONFIG_T::stride_height) {
        InputWidthLoop:
            #pragma unroll
            [[intel::disable_loop_pipelining]] for (int inp_width = 0; inp_width < restricted_padded_width;
                                                    inp_width += CONFIG_T::stride_width) {
                [[intel::fpga_register]] typename data_T::value_type pool[CONFIG_T::pool_height * CONFIG_T::pool_width];

                // Keep track of number of pixels in image vs padding region; needed for rescaling Average Pooling
                [[intel::fpga_register]] unsigned img_overlap = 0;

            PoolHeightLoop:
                #pragma unroll
                [[intel::disable_loop_pipelining]] for (int pool_col = 0; pool_col < CONFIG_T::stride_height; pool_col++) {
                PoolWidthLoop:
                    #pragma unroll
                    [[intel::disable_loop_pipelining]] for (int pool_row = 0; pool_row < CONFIG_T::stride_width;
                                                            pool_row++) {
                        if (inp_col + pool_col < CONFIG_T::pad_top ||
                            inp_col + pool_col >= (full_padded_height - CONFIG_T::pad_bottom) ||
                            inp_width + pool_row < CONFIG_T::pad_left ||
                            inp_width + pool_row >= (full_padded_width - CONFIG_T::pad_right)) {
                            // Add padding
                            pool[pool_col * CONFIG_T::stride_width + pool_row] =
                                pad_val<typename data_T::value_type, CONFIG_T::pool_op>();
                            if (CONFIG_T::count_pad)
                                img_overlap++;
                        } else {
                            // Current element is from input image
                            pool[pool_col * CONFIG_T::stride_width + pool_row] = data
                                [(inp_col + pool_col - CONFIG_T::pad_top) * CONFIG_T::in_width * CONFIG_T::n_filt +
                                 (inp_width + pool_row - CONFIG_T::pad_left) * CONFIG_T::n_filt + filt];
                            img_overlap++;
                        }
                    }
                }

                // Pooling operation
                res[(inp_col / CONFIG_T::stride_height) * CONFIG_T::out_width * CONFIG_T::n_filt +
                    (inp_width / CONFIG_T::stride_width) * CONFIG_T::n_filt + filt] =
                    static_cast<typename res_T::value_type>(pool_op<
                                                            typename data_T::value_type,
                                                            CONFIG_T::pool_height * CONFIG_T::pool_width,
                                                            CONFIG_T::pool_op,
                                                            typename CONFIG_T::accum_t>(pool, img_overlap));
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T> void global_pooling2d_cl(const data_T &data, res_T &res) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0);
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height);

FiltLoop:
    #pragma unroll
    [[intel::disable_loop_pipelining]] for (int filt = 0; filt < CONFIG_T::n_filt; filt++) {
        [[intel::fpga_register]] typename data_T::value_type pool[CONFIG_T::in_height * CONFIG_T::in_width];

    InputLoop:
        #pragma unroll
        [[intel::disable_loop_pipelining]] for (int i = 0; i < CONFIG_T::in_height * CONFIG_T::in_width; i++) {
            pool[i] = data[i * CONFIG_T::n_filt + filt];
        }

        res[filt] = static_cast<typename res_T::value_type>(pool_op<
                                                            typename data_T::value_type,
                                                            CONFIG_T::in_height * CONFIG_T::in_width,
                                                            CONFIG_T::pool_op,
                                                            typename CONFIG_T::accum_t>(pool));
    }
}

} // namespace nnet

#endif
