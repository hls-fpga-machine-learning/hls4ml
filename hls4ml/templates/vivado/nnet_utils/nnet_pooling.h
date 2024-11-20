#ifndef NNET_POOLING_H_
#define NNET_POOLING_H_

#include "nnet_helpers.h"
#include <iostream>

namespace nnet {

// Return the maximum value from an array
template <typename T, int N, typename accum_t> accum_t max(T x[N]) {
    T y = x[0];
    for (int i = 1; i < N; i++) {
        y = x[i] > y ? x[i] : y;
    }
    return y;
}

// Return the mean value of an array
template <typename T, int N, typename accum_t> accum_t avg(T (&x)[N], unsigned length) {
    accum_t y = 0;
    for (int i = 0; i < N; i++) {
        y += x[i];
    }
    y /= length;
    return y;
}

// Enumeration for pooling operation (max, avg, l2norm pooling)
enum Pool_Op { Max, Average }; // L2Norm };
template <typename T, int N, Pool_Op op, typename accum_t> accum_t pool_op(T (&x)[N], unsigned length) {
    switch (op) {
    case Max:
        return max<T, N, accum_t>(x);
    case Average:
        return avg<T, N, accum_t>(x, length);
        // case L2Norm: return l2norm<T, N>(x);
    }
}

template <typename T, int N, Pool_Op op, typename accum_t> accum_t pool_op(T (&x)[N]) {
    return pool_op<T, N, op, accum_t>(x, N);
}

template <typename T, Pool_Op op> T pad_val() {
    /*---
     *- In Tensorflow, pooling ignores the value in the padded cells
     *- For Avg pooling, return 0 (the divisior is modified to the
     *- area overlapping the unpadded image.
     *- For max pooling, return the most negative value for the type.
     *- TODO this is not really generic, it assumes fixed point or integer T
    ---*/
    switch (op) {
    case Max: {
        T x = 0;
        x[x.width - 1] = 1;
        return x;
        break;
    }
    case Average:
        return 0;
    }
}

struct pooling1d_config {
    // IO size
    static const unsigned n_in = 10;
    static const unsigned pool_width = 2;
    static const unsigned stride_width = 2;
    static const unsigned n_out = (n_in - pool_width) / stride_width + 1;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;
    // Pooling function
    static const Pool_Op pool_op = Max;
};

template <typename CONFIG_T> constexpr int pool_op_limit_1d() {
    return CONFIG_T::n_in * CONFIG_T::n_filt / CONFIG_T::reuse_factor;
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling1d_cl(data_T data[CONFIG_T::n_in * CONFIG_T::n_filt], res_T res[CONFIG_T::n_out * CONFIG_T::n_filt]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // TODO partition the arrays according to the reuse factor
    const int limit = pool_op_limit_1d<CONFIG_T>();
    #pragma HLS ALLOCATION function instances=CONFIG_T::pool_op limit=limit
    // Add any necessary padding

    // Add padding and reduce input width to area covered by pooling function
    static constexpr int full_padded_width = CONFIG_T::n_in + CONFIG_T::pad_left + CONFIG_T::pad_right;
    static constexpr int restricted_padded_width = full_padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width;

    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        // Loop over input image x in steps of stride
        for (int ii = 0; ii < restricted_padded_width; ii += CONFIG_T::stride_width) {
            unsigned overlap_pixel = 0;
            data_T pool[CONFIG_T::pool_width];
            #pragma HLS ARRAY_PARTITION variable=pool complete dim=0

            for (int jj = 0; jj < CONFIG_T::pool_width; jj++) {
                if (ii + jj >= CONFIG_T::pad_left && ii + jj < CONFIG_T::n_in + CONFIG_T::pad_left) {
                    pool[jj] = data[(ii + jj - CONFIG_T::pad_left) * CONFIG_T::n_filt + ff];
                    overlap_pixel++;
                } else
                    pool[jj] = pad_val<data_T, CONFIG_T::pool_op>();
            }

            int patch_size = CONFIG_T::count_pad ? CONFIG_T::stride_width : overlap_pixel;

            res[(ii / CONFIG_T::stride_width) * CONFIG_T::n_filt + ff] =
                pool_op<data_T, CONFIG_T::pool_width, CONFIG_T::pool_op, typename CONFIG_T::accum_t>(pool, patch_size);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling1d_cl(data_T data[CONFIG_T::n_in * CONFIG_T::n_filt], res_T res[CONFIG_T::n_filt]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);

    // TODO partition the arrays according to the reuse factor
    const int limit = pool_op_limit_1d<CONFIG_T>();
    #pragma HLS ALLOCATION function instances=CONFIG_T::pool_op limit=limit

    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        data_T pool[CONFIG_T::n_in];
        #pragma HLS ARRAY_PARTITION variable=pool complete dim=0
        for (int jj = 0; jj < CONFIG_T::n_in; jj++) {
            pool[jj] = data[jj * CONFIG_T::n_filt + ff];
        }
        // do the pooling
        res[ff] = pool_op<data_T, CONFIG_T::n_in, CONFIG_T::pool_op, typename CONFIG_T::accum_t>(pool);
    }
}

struct pooling2d_config {
    // IO size
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_filt = 4;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
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
    // Reuse factor
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef float accum_t;
};

template <typename CONFIG_T> constexpr int pool_op_limit() {
    return (CONFIG_T::out_height * CONFIG_T::out_width) * CONFIG_T::n_filt / CONFIG_T::reuse_factor;
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt]
) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // TODO partition the arrays according to the reuse factor
    const int limit = pool_op_limit<CONFIG_T>();
    #pragma HLS ALLOCATION function instances=CONFIG_T::pool_op limit=limit

    // Add padding and reduce input width to area covered by pooling function
    static constexpr int full_padded_width = CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right;
    static constexpr int full_padded_height = CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom;
    static constexpr int restricted_padded_width = full_padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width;
    static constexpr int restricted_padded_height = full_padded_height / CONFIG_T::stride_height * CONFIG_T::stride_height;

    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {

        // Loop over input image y in steps of stride
        for (int ii = 0; ii < restricted_padded_height; ii += CONFIG_T::stride_height) {
            // Loop over input image x in steps of stride
            for (int jj = 0; jj < restricted_padded_width; jj += CONFIG_T::stride_width) {
                data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
                #pragma HLS ARRAY_PARTITION variable=pool complete dim=0

                unsigned overlap_pixel = 0;

                // Loop over pool window y
                for (int kk = 0; kk < CONFIG_T::stride_height; kk++) {
                    // Loop over pool window x
                    for (int ll = 0; ll < CONFIG_T::stride_width; ll++) {
                        bool cond1 = ii + kk >= CONFIG_T::pad_top && ii + kk < CONFIG_T::in_height + CONFIG_T::pad_top;
                        bool cond2 = jj + ll >= CONFIG_T::pad_left && jj + ll < CONFIG_T::in_width + CONFIG_T::pad_left;
                        if (cond1 && cond2) {
                            unsigned data_idx =
                                ((ii + kk - CONFIG_T::pad_top) * CONFIG_T::in_width + (jj + ll - CONFIG_T::pad_left)) *
                                    CONFIG_T::n_filt +
                                ff;
                            pool[kk * CONFIG_T::stride_width + ll] = data[data_idx];
                            overlap_pixel++;
                        } else
                            pool[kk * CONFIG_T::stride_width + ll] = pad_val<data_T, CONFIG_T::pool_op>();
                    }
                }

                int patch_size = CONFIG_T::count_pad ? CONFIG_T::stride_width * CONFIG_T::stride_height : overlap_pixel;

                res[(ii / CONFIG_T::stride_height) * CONFIG_T::out_width * CONFIG_T::n_filt +
                    (jj / CONFIG_T::stride_width) * CONFIG_T::n_filt + ff] =
                    pool_op<
                        data_T,
                        CONFIG_T::pool_height * CONFIG_T::pool_width,
                        CONFIG_T::pool_op,
                        typename CONFIG_T::accum_t>(pool, patch_size);
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_cf(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_filt],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt]
) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // TODO partition the arrays according to the reuse factor
    const int limit = pool_op_limit<CONFIG_T>();
    #pragma HLS ALLOCATION function instances=CONFIG_T::pool_op limit=limit
    // Add padding and reduce input width to area covered by pooling function
    static constexpr int full_padded_width = CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right;
    static constexpr int full_padded_height = CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom;
    static constexpr int restricted_padded_width = full_padded_width / CONFIG_T::stride_width * CONFIG_T::stride_width;
    static constexpr int restricted_padded_height = full_padded_height / CONFIG_T::stride_height * CONFIG_T::stride_height;

    for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        // Loop over input image y in steps of stride
        for (int ii = 0; ii < restricted_padded_height; ii += CONFIG_T::stride_height) {
            // Loop over input image x in steps of stride
            for (int jj = 0; jj < restricted_padded_width; jj += CONFIG_T::stride_width) {
                data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
                #pragma HLS ARRAY_PARTITION variable=pool complete dim=0
                // Keep track of number of pixels in image vs padding region
                unsigned img_overlap = 0;
                // Loop over pool window y
                for (int kk = 0; kk < CONFIG_T::stride_height; kk++) {
                    // Loop over pool window x
                    for (int ll = 0; ll < CONFIG_T::stride_width; ll++) {
                        if (ii + kk < CONFIG_T::pad_top || ii + kk >= (full_padded_height - CONFIG_T::pad_bottom) ||
                            jj + ll < CONFIG_T::pad_left || jj + ll >= (full_padded_width - CONFIG_T::pad_right)) {
                            // Add padding
                            pool[kk * CONFIG_T::stride_width + ll] = pad_val<data_T, CONFIG_T::pool_op>();
                            if (CONFIG_T::count_pad)
                                img_overlap++;
                        } else {
                            pool[kk * CONFIG_T::stride_width + ll] = data
                                [(ii + kk - CONFIG_T::pad_top) * CONFIG_T::in_width +
                                 ff * CONFIG_T::in_width * CONFIG_T::in_height + ll + jj - CONFIG_T::pad_left];
                            img_overlap++;
                        }
                    }
                }
                // do the pooling
                // TODO in the case of average pooling, need to reduce height * width to area of pool window
                // not overlapping padding region
                res[(ii / CONFIG_T::stride_height) * CONFIG_T::out_width + (jj / CONFIG_T::stride_width) +
                    ff * CONFIG_T::out_height * CONFIG_T::out_width] =
                    pool_op<
                        data_T,
                        CONFIG_T::pool_height * CONFIG_T::pool_width,
                        CONFIG_T::pool_op,
                        typename CONFIG_T::accum_t>(pool);
                // If the pool op is Average, the zero-padding needs to be removed from the results
                if (CONFIG_T::pool_op == Average) {
                    data_T rescale =
                        static_cast<data_T>(CONFIG_T::pool_height) * static_cast<data_T>(CONFIG_T::pool_width) / img_overlap;
                    res[(ii / CONFIG_T::stride_height) * CONFIG_T::out_width + (jj / CONFIG_T::stride_width) +
                        ff * CONFIG_T::out_height * CONFIG_T::out_width] *= rescale;
                }
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

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    const int limit = pool_op_limit<CONFIG_T>();
    #pragma HLS ALLOCATION instances=pool_op limit=limit function

FiltLoop:
    for (int filt = 0; filt < CONFIG_T::n_filt; filt++) {
        data_T pool[CONFIG_T::in_height * CONFIG_T::in_width];

    InputLoop:
        for (int i = 0; i < CONFIG_T::in_height * CONFIG_T::in_width; i++) {
            pool[i] = data[i * CONFIG_T::n_filt + filt];
        }

        res[filt] = static_cast<res_T>(
            pool_op<data_T, CONFIG_T::in_height * CONFIG_T::in_width, CONFIG_T::pool_op, typename CONFIG_T::accum_t>(pool)
        );
    }
}

} // namespace nnet

#endif
