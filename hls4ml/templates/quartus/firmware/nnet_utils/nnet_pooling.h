#ifndef NNET_POOLING_H_
#define NNET_POOLING_H_

#include "nnet_common.h"

namespace nnet {

// Returns the maximum value from an array of size N
template <typename T, int N> T max(T x[N]) {
    hls_register T y = x[0];

    // Due to loop dependencies, pipelining & unrolling is not possible
    // Explictily disabling pipeline significantly reduces resource usage
    #pragma disable_loop_pipelining
    for (int i = 1; i < N; i++) {
        if (x[i] > y)
            y = x[i];
    }

    return y;
}

// Returns the mean value of an array of size N
template <typename T, int N> T avg(T (&x)[N]) {
    hls_register T y = 0;

    // Due to loop dependencies, pipelining & unrolling is not possible
    // Explictily disabling pipeline significantly reduces resource usage
    #pragma disable_loop_pipelining
    for (int i = 0; i < N; i++) {
        y += x[i];
    }

    y /= N;
    return y;
}

// Returns the mean value of an array of size N
// Overload of the above function; using a wider accumulator than the input to avoid overflow
template <int W, int N> ac_int<W, true> avg(ac_int<W, true> (&x)[N]) {
    hls_register ac_int<W + ceillog2(N), true> tmp = 0;

    // Due to loop dependencies, pipelining & unrolling is not possible
    // Explictily disabling pipeline significantly reduces resource usage
    #pragma disable_loop_pipelining
    for (int i = 0; i < N; i++) {
        tmp += x[i];
    }

    tmp /= N;

    // Cast back to original type
    ac_int<W, true> y = static_cast<ac_int<W, true>>(tmp);
    return tmp;
}

// Returns the mean value of an array of size N
// Overload of the above function; using a wider accumulator than the input to avoid overflow
template <int W, int I, int N> ac_fixed<W, I, true> avg(ac_fixed<W, I, true> (&x)[N]) {
    hls_register ac_fixed<W + ceillog2(N), I + ceillog2(N), true> tmp = 0;

    // Due to loop dependencies, pipelining & unrolling is not possible
    // Explictily disabling pipeline significantly reduces resource usage
    #pragma disable_loop_pipelining
    for (int i = 0; i < N; i++) {
        tmp += x[i];
    }

    tmp /= N;

    // Cast back to original type
    ac_fixed<W, I, true> y = tmp;
    return y;
}

// Enumeration for pooling functions
enum Pool_Op { Max, Average };
template <typename T, int N, Pool_Op op> T pool_op(T (&x)[N]) {
    switch (op) {
    case Max:
        return max<T, N>(x);
    case Average:
        return avg(x);
    }
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

template <class data_T, class res_T, typename CONFIG_T>
void pooling1d_cl(data_T data[CONFIG_T::n_in * CONFIG_T::n_filt], res_T res[CONFIG_T::n_out * CONFIG_T::n_filt]) {
    // Add padding and reduce input width to area covered by pooling function
    static constexpr int full_padded_width = CONFIG_T::n_in + CONFIG_T::pad_left + CONFIG_T::pad_right;
    static constexpr int restricted_padded_width = full_padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width;

FiltLoop:
    #pragma unroll
    #pragma disable_loop_pipelining
    for (int filt = 0; filt < CONFIG_T::n_filt; filt++) {
    InputWidthLoop:
        #pragma unroll
        #pragma disable_loop_pipelining
        for (int inp_col = 0; inp_col < restricted_padded_width; inp_col += CONFIG_T::stride_width) {
            hls_register data_T pool[CONFIG_T::pool_width];

            // Keep track of number of pixels in image vs padding region; needed for rescaling Average Pooling
            hls_register unsigned img_overlap = 0;

        PoolWidthLoop:
            #pragma unroll
            #pragma disable_loop_pipelining
            for (int pool_col = 0; pool_col < CONFIG_T::stride_width; pool_col++) {
                if (inp_col + pool_col < CONFIG_T::pad_left ||
                    inp_col + pool_col >= (full_padded_width - CONFIG_T::pad_right)) {
                    // Add padding
                    pool[pool_col] = pad_val<data_T, CONFIG_T::pool_op>();
                    if (CONFIG_T::count_pad)
                        img_overlap++;
                } else {
                    // Current element is from input image
                    pool[pool_col] = data[(inp_col + pool_col - CONFIG_T::pad_left) * CONFIG_T::n_filt + filt];
                    img_overlap++;
                }
            }

            // Pooling operation
            res[(inp_col / CONFIG_T::stride_width) * CONFIG_T::n_filt + filt] =
                static_cast<res_T>(pool_op<data_T, CONFIG_T::pool_width, CONFIG_T::pool_op>(pool));

            // If the pool op is Average, the zero-padding needs to be removed from the results
            if (CONFIG_T::pool_op == Average)
                res[(inp_col / CONFIG_T::stride_width) * CONFIG_T::n_filt + filt] *=
                    (static_cast<data_T>(CONFIG_T::pool_width) / img_overlap);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling1d_cl(data_T data[CONFIG_T::n_in * CONFIG_T::n_filt], res_T res[CONFIG_T::n_filt]) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);

FiltLoop:
    #pragma unroll
    #pragma disable_loop_pipelining
    for (int filt = 0; filt < CONFIG_T::n_filt; filt++) {
        hls_register data_T pool[CONFIG_T::n_in];

    InputWidthLoop:
        #pragma unroll
        #pragma disable_loop_pipelining
        for (int col = 0; col < CONFIG_T::n_in; col++) {
            pool[col] = data[col * CONFIG_T::n_filt + filt];
        }

        res[filt] = static_cast<res_T>(pool_op<data_T, CONFIG_T::n_in, CONFIG_T::pool_op>(pool));
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

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt]
) {
    // Add padding and reduce input width to area covered by pooling function
    static constexpr int full_padded_width = CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right;
    static constexpr int full_padded_height = CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom;
    static constexpr int restricted_padded_width = full_padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width;
    static constexpr int restricted_padded_height = full_padded_height / CONFIG_T::stride_height * CONFIG_T::stride_height;

FiltLoop:
    #pragma unroll
    #pragma disable_loop_pipelining
    for (int filt = 0; filt < CONFIG_T::n_filt; filt++) {
    InputHeightLoop:
        #pragma unroll
        #pragma disable_loop_pipelining
        for (int inp_col = 0; inp_col < restricted_padded_height; inp_col += CONFIG_T::stride_height) {
        InputWidthLoop:
            #pragma unroll
            #pragma disable_loop_pipelining
            for (int inp_width = 0; inp_width < restricted_padded_width; inp_width += CONFIG_T::stride_width) {
                hls_register data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];

                // Keep track of number of pixels in image vs padding region; needed for rescaling Average Pooling
                hls_register unsigned img_overlap = 0;

            PoolHeightLoop:
                #pragma unroll
                #pragma disable_loop_pipelining
                for (int pool_col = 0; pool_col < CONFIG_T::stride_height; pool_col++) {
                PoolWidthLoop:
                    #pragma unroll
                    #pragma disable_loop_pipelining
                    for (int pool_row = 0; pool_row < CONFIG_T::stride_width; pool_row++) {
                        if (inp_col + pool_col < CONFIG_T::pad_top ||
                            inp_col + pool_col >= (full_padded_height - CONFIG_T::pad_bottom) ||
                            inp_width + pool_row < CONFIG_T::pad_left ||
                            inp_width + pool_row >= (full_padded_width - CONFIG_T::pad_right)) {
                            // Add padding
                            pool[pool_col * CONFIG_T::stride_width + pool_row] = pad_val<data_T, CONFIG_T::pool_op>();
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
                    static_cast<res_T>(pool_op<data_T, CONFIG_T::pool_height * CONFIG_T::pool_width, CONFIG_T::pool_op>(pool)
                    );

                // If the pool op is Average, the zero-padding needs to be removed from the results
                if (CONFIG_T::pool_op == Average)
                    res[(inp_col / CONFIG_T::stride_height) * CONFIG_T::out_width * CONFIG_T::n_filt +
                        (inp_width / CONFIG_T::stride_width) * CONFIG_T::n_filt + filt] *=
                        (static_cast<data_T>(CONFIG_T::pool_height) * static_cast<data_T>(CONFIG_T::pool_width) / img_overlap
                        );
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling2d_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt], res_T res[CONFIG_T::n_filt]
) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0);
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height);

FiltLoop:
    #pragma unroll
    #pragma disable_loop_pipelining
    for (int filt = 0; filt < CONFIG_T::n_filt; filt++) {
        hls_register data_T pool[CONFIG_T::in_height * CONFIG_T::in_width];

    InputLoop:
        #pragma unroll
        #pragma disable_loop_pipelining
        for (int i = 0; i < CONFIG_T::in_height * CONFIG_T::in_width; i++) {
            pool[i] = data[i * CONFIG_T::n_filt + filt];
        }

        res[filt] = static_cast<res_T>(pool_op<data_T, CONFIG_T::in_height * CONFIG_T::in_width, CONFIG_T::pool_op>(pool));
    }
}

} // namespace nnet

#endif
