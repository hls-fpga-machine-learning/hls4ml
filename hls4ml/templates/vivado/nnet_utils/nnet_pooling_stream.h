#ifndef NNET_POOLING_STREAM_H_
#define NNET_POOLING_STREAM_H_

#include "ap_shift_reg.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_conv_stream.h"
#include "nnet_pooling.h"
#include "utils/x_hls_utils.h"

namespace nnet {

// *************************************************
//       Max/average pooling
// *************************************************

template <class T, int N, class CONFIG_T> T reduce_pool(T x[N]) {
    #pragma HLS INLINE
    if (CONFIG_T::pool_op == Max) {
        Op_max<T> op_max;
        return reduce<T, N, Op_max<T>>(x, op_max);
    } else {
        Op_add<T> op_add;
        T sum = reduce<T, N, Op_add<T>>(x, op_add);
        return sum / N;
    }
}

template <unsigned TABLE_SIZE, unsigned POOL_SIZE> void init_pool_table(unsigned table[TABLE_SIZE]) {
    for (unsigned ii = 0; ii < TABLE_SIZE; ii++) {
        table[ii] = ii % POOL_SIZE;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void compute_pool_encoded_2d(
    const unsigned h_idx,
    const unsigned w_idx,
    const data_T &in_elem,
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::pool_height * CONFIG_T::pool_width * CONFIG_T::n_filt],
    hls::stream<res_T> &res,
    res_T &res_pack,
    unsigned &outputs_ready
) {
    // Nearest H without unused pixels on the right
    constexpr unsigned nH =
        ((CONFIG_T::in_height - CONFIG_T::pool_height) / CONFIG_T::stride_height) * CONFIG_T::stride_height +
        CONFIG_T::pool_height;
    // Scaled H that behaves like original H
    constexpr unsigned sH =
        (DIV_ROUNDUP(CONFIG_T::pool_height, CONFIG_T::stride_height) - 1) * CONFIG_T::stride_height + CONFIG_T::pool_height;
    // Nearest W without unused pixels on the right
    constexpr unsigned nW = ((CONFIG_T::in_width - CONFIG_T::pool_width) / CONFIG_T::stride_width) * CONFIG_T::stride_width +
                            CONFIG_T::pool_width;
    // Scaled W that behaves like original W
    constexpr unsigned sW =
        (DIV_ROUNDUP(CONFIG_T::pool_width, CONFIG_T::stride_width) - 1) * CONFIG_T::stride_width + CONFIG_T::pool_width;

#ifdef __SYNTHESIS__
    bool initialized = false;
    unsigned pool_table_height[CONFIG_T::in_height];
    unsigned pool_table_width[CONFIG_T::in_width];
#else
    static bool initialized = false;
    static unsigned pool_table_height[CONFIG_T::in_height];
    static unsigned pool_table_width[CONFIG_T::in_width];
#endif
    if (!initialized) {
        init_pool_table<CONFIG_T::in_height, CONFIG_T::pool_height>(pool_table_height);
        init_pool_table<CONFIG_T::in_width, CONFIG_T::pool_width>(pool_table_width);
        initialized = true;
    }

    #pragma HLS INLINE

    if (data_T::size / CONFIG_T::n_filt > 1) {
        #pragma HLS ARRAY_PARTITION variable=pool_table_height complete
        #pragma HLS ARRAY_PARTITION variable=pool_table_width complete
    }

    typename CONFIG_T::accum_t pool_window[CONFIG_T::pool_height * CONFIG_T::pool_width];
    #pragma HLS ARRAY_PARTITION variable=pool_window complete

    const unsigned sh_idx = pool_table_height[h_idx] * CONFIG_T::pool_width;
    const unsigned wp_idx = w_idx * (data_T::size / CONFIG_T::n_filt);

PixelLoop:
    for (unsigned p = 0; p < data_T::size / CONFIG_T::n_filt; p++) {
        #pragma HLS PIPELINE

        ap_uint<CONFIG_T::pool_height *CONFIG_T::pool_width> filt_mask = 0;
        if ((h_idx < nH) && (wp_idx + p < nW)) {
            filt_mask = sh_idx + pool_table_width[wp_idx + p] + 1;
        }

    CopyDataFilt:
        for (unsigned c = 0; c < CONFIG_T::n_filt; c++) {
            if (filt_mask > 0)
                data_window[c * CONFIG_T::pool_height * CONFIG_T::pool_width + filt_mask.to_uint() - 1].write(
                    in_elem[p * CONFIG_T::n_filt + c]
                );
        }

        if (filt_mask == CONFIG_T::pool_height * CONFIG_T::pool_width) {
        FiltLoop:
            for (unsigned c = 0; c < CONFIG_T::n_filt; c++) {
            PoolLoop:
                for (unsigned f = 0; f < CONFIG_T::pool_height * CONFIG_T::pool_width; f++) {
                    pool_window[f] = data_window[c * CONFIG_T::pool_height * CONFIG_T::pool_width + f].read();
                }
                if (res_T::size / CONFIG_T::n_filt ==
                    1) { // Saves resources if we don't pack output, compiler will remove the else branch
                    res_pack[c] =
                        reduce_pool<typename CONFIG_T::accum_t, CONFIG_T::pool_height * CONFIG_T::pool_width, CONFIG_T>(
                            pool_window
                        );
                } else {
                    res_pack[outputs_ready * CONFIG_T::n_filt + c] =
                        reduce_pool<typename CONFIG_T::accum_t, CONFIG_T::pool_height * CONFIG_T::pool_width, CONFIG_T>(
                            pool_window
                        );
                }
            }
            if (res_T::size / CONFIG_T::n_filt ==
                1) { // Saves resources if we don't pack output, compiler will remove the else branch
                res.write(res_pack);
            } else {
                if (outputs_ready == (res_T::size / CONFIG_T::n_filt) - 1) {
                    res.write(res_pack);
                    outputs_ready = 0;
                } else {
                    outputs_ready++;
                }
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_encoded_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height && CONFIG_T::pool_width == CONFIG_T::stride_width);

    res_T res_pack;
    PRAGMA_DATA_PACK(res_pack)
    unsigned outputs_ready = 0;

    hls::stream<typename data_T::value_type> data_window[CONFIG_T::pool_height * CONFIG_T::pool_width * CONFIG_T::n_filt];
    constexpr int win_depth = CONFIG_T::pool_height * CONFIG_T::out_width;
    for (unsigned i_out = 0; i_out < CONFIG_T::pool_height * CONFIG_T::pool_width * CONFIG_T::n_filt; i_out++) {
        #pragma HLS STREAM variable=data_window[i_out] depth=win_depth
    }

    constexpr int pack_factor = data_T::size / CONFIG_T::n_filt;

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (pack_factor); i_iw++) {
            #pragma HLS LOOP_FLATTEN
            if (res_T::size / CONFIG_T::n_filt == 1) {
                #pragma HLS PIPELINE II=pack_factor
            }
            compute_pool_encoded_2d<data_T, res_T, CONFIG_T>(
                i_ih, i_iw, data.read(), data_window, res, res_pack, outputs_ready
            );
        }
    }
}

// *************************************************
//       Line Buffer Implementation (Phil's)
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void compute_pool_buffer_2d(
    const data_T &in_elem,
    ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::pool_height - 1, 1)]
                                                                             [CONFIG_T::n_filt],
    hls::stream<res_T> &res
) {
    #pragma HLS INLINE
    const static int lShiftX = CONFIG_T::pool_width - 1;
    const static int lShiftY = CONFIG_T::pool_height - 1;
    static int pX = 0; // pixel X
    static int pY = 0; // pixel Y
    static int sX = 0; // stride X
    static int sY = 0; // stride Y

    typename CONFIG_T::accum_t pool_window[CONFIG_T::pool_height * CONFIG_T::pool_width];
    #pragma HLS ARRAY_PARTITION variable=pool_window complete

    static typename data_T::value_type kernel_data[CONFIG_T::pool_height * CONFIG_T::pool_width * CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable = kernel_data complete dim = 0

    res_T res_pack;
    PRAGMA_DATA_PACK(res_pack)

    // Add pixel into line buffer, return pooling kernels
    nnet::shift_line_buffer<data_T, CONFIG_T>(in_elem, line_buffer, kernel_data);

    // Can compute pooling output
    if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
    FiltLoop:
        for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
        #pragma HLS PIPELINE

        // Retrieve data for current channel
        PoolLoop:
            for (unsigned i_ihw = 0; i_ihw < CONFIG_T::pool_height * CONFIG_T::pool_width; i_ihw++) {
                pool_window[i_ihw] = kernel_data[i_ihw * CONFIG_T::n_filt + i_ic];
            }

            // Compute Pooling
            res_pack[i_ic] =
                reduce_pool<typename CONFIG_T::accum_t, CONFIG_T::pool_height * CONFIG_T::pool_width, CONFIG_T>(pool_window);
        }

        // Write to output
        res.write(res_pack);
    }

    // Counter Housekeeping
    if (pX + 1 == CONFIG_T::in_width) // Includes padding, end of line (padded)
    {
        pX = 0;
        sX = 0;
        if (pY + 1 == CONFIG_T::in_height) { // Reached bottom of image
            pY = 0;
            sY = 0;
        } else { // Next line
            pY = pY + 1;
            // Update stride (threshold) ? subtract stride : increment stride
            sY = ((sY - lShiftY) == 0) ? sY - CONFIG_T::stride_height + 1 : sY + 1;
        }
    } else {
        pX = pX + 1;
        // Update stride (threshold) ? subtract stride : increment stride
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_buffer_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height && CONFIG_T::pool_width == CONFIG_T::stride_width);

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::pool_height - 1, 1)]
                                                                                    [CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable = line_buffer complete dim = 2

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE

            compute_pool_buffer_2d<data_T, res_T, CONFIG_T>(data.read(), line_buffer, res);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling2d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    #pragma HLS inline recursive
    switch (CONFIG_T::implementation) {
    case conv_implementation::linebuffer:
        pooling2d_buffer_cl<data_T, res_T, CONFIG_T>(data, res);
        break;
    case conv_implementation::encoded:
        pooling2d_encoded_cl<data_T, res_T, CONFIG_T>(data, res);
        break;
    }
}

// *************************************************
//                  Pooling 1D
// *************************************************

template <class data_T, class res_T, typename CONFIG_T>
void compute_pool_encoded_1d(
    const unsigned w_idx,
    const data_T &in_elem,
    hls::stream<typename data_T::value_type> data_window[CONFIG_T::pool_width * CONFIG_T::n_filt],
    hls::stream<res_T> &res,
    res_T &res_pack,
    unsigned &outputs_ready
) {
    // Nearest W without unused pixels on the right
    constexpr unsigned nW =
        ((CONFIG_T::n_in - CONFIG_T::pool_width) / CONFIG_T::stride_width) * CONFIG_T::stride_width + CONFIG_T::pool_width;
    // Scaled W that behaves like original W
    constexpr unsigned sW =
        (DIV_ROUNDUP(CONFIG_T::pool_width, CONFIG_T::stride_width) - 1) * CONFIG_T::stride_width + CONFIG_T::pool_width;

#ifdef __SYNTHESIS__
    bool initialized = false;
    unsigned pool_table_width[CONFIG_T::n_in];
#else
    static bool initialized = false;
    static unsigned pool_table_width[CONFIG_T::n_in];
#endif
    if (!initialized) {
        init_pool_table<CONFIG_T::n_in, CONFIG_T::pool_width>(pool_table_width);
        initialized = true;
    }

    #pragma HLS INLINE

    if (data_T::size / CONFIG_T::n_filt > 1) {
        #pragma HLS ARRAY_PARTITION variable=pool_table_width complete
    }

    typename CONFIG_T::accum_t pool_window[CONFIG_T::pool_width];
    #pragma HLS ARRAY_PARTITION variable=pool_window complete

    const unsigned wp_idx = w_idx * (data_T::size / CONFIG_T::n_filt);

PixelLoop:
    for (unsigned p = 0; p < data_T::size / CONFIG_T::n_filt; p++) {
        #pragma HLS PIPELINE

        ap_uint<CONFIG_T::pool_width> filt_mask = 0;
        if (wp_idx + p < nW) {
            filt_mask = pool_table_width[wp_idx + p] + 1;
        }

    CopyDataFilt:
        for (unsigned c = 0; c < CONFIG_T::n_filt; c++) {
            if (filt_mask > 0)
                data_window[c * CONFIG_T::pool_width + filt_mask.to_uint() - 1].write(in_elem[p * CONFIG_T::n_filt + c]);
        }

        if (filt_mask == CONFIG_T::pool_width) {
        FiltLoop:
            for (unsigned c = 0; c < CONFIG_T::n_filt; c++) {
            PoolLoop:
                for (unsigned f = 0; f < CONFIG_T::pool_width; f++) {
                    pool_window[f] = data_window[c * CONFIG_T::pool_width + f].read();
                }
                if (res_T::size / CONFIG_T::n_filt ==
                    1) { // Saves resources if we don't pack output, compiler will remove the else branch
                    res_pack[c] = reduce_pool<typename CONFIG_T::accum_t, CONFIG_T::pool_width, CONFIG_T>(pool_window);
                } else {
                    res_pack[outputs_ready * CONFIG_T::n_filt + c] =
                        reduce_pool<typename CONFIG_T::accum_t, CONFIG_T::pool_width, CONFIG_T>(pool_window);
                }
            }
            if (res_T::size / CONFIG_T::n_filt ==
                1) { // Saves resources if we don't pack output, compiler will remove the else branch
                res.write(res_pack);
            } else {
                if (outputs_ready == (res_T::size / CONFIG_T::n_filt) - 1) {
                    res.write(res_pack);
                    outputs_ready = 0;
                } else {
                    outputs_ready++;
                }
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling1d_encoded_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);

    res_T res_pack;
    PRAGMA_DATA_PACK(res_pack)
    unsigned outputs_ready = 0;

    hls::stream<typename data_T::value_type> data_window[CONFIG_T::pool_width * CONFIG_T::n_filt];
    constexpr int win_depth = CONFIG_T::n_out;
    for (unsigned i_out = 0; i_out < CONFIG_T::pool_width * CONFIG_T::n_filt; i_out++) {
        #pragma HLS STREAM variable=data_window[i_out] depth=win_depth
    }

    constexpr int pack_factor = data_T::size / CONFIG_T::n_filt;

ReadInputWidth:
    for (unsigned i_iw = 0; i_iw < CONFIG_T::n_in / (pack_factor); i_iw++) {
        #pragma HLS LOOP_FLATTEN
        if (res_T::size / CONFIG_T::n_filt == 1) {
            #pragma HLS PIPELINE II=pack_factor
        }
        compute_pool_encoded_1d<data_T, res_T, CONFIG_T>(i_iw, data.read(), data_window, res, res_pack, outputs_ready);
    }
}

// *************************************************
//       Line Buffer Implementation (Phil's) 1D
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void compute_pool_buffer_1d(const data_T &in_elem, hls::stream<res_T> &res) {
    #pragma HLS INLINE
    const static int lShiftX = CONFIG_T::pool_width - 1;
    // Counters
    static int pX = 0;
    static int sX = 0;

    typename CONFIG_T::accum_t pool_window[CONFIG_T::pool_width];
    #pragma HLS ARRAY_PARTITION variable=pool_window complete

    static typename data_T::value_type kernel_data[CONFIG_T::pool_width * CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable = kernel_data complete dim = 0

    res_T res_pack;
    PRAGMA_DATA_PACK(res_pack)

    // Add pixel into line buffer, return pooling kernels
    // 1D case line buffer not necessary. Put directly into the kernel_data buffer
    nnet::kernel_shift_1d<data_T, CONFIG_T>(in_elem, kernel_data);

    // Can compute pooling output
    if ((sX - lShiftX) == 0 && pX > lShiftX - 1) {
    FiltLoop:
        for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
        #pragma HLS PIPELINE

        // Retrieve data for current channel
        PoolLoop:
            for (unsigned i_iw = 0; i_iw < CONFIG_T::pool_width; i_iw++) {
                pool_window[i_iw] = kernel_data[i_iw * CONFIG_T::n_filt + i_ic];
            }

            // Compute Pooling
            res_pack[i_ic] = reduce_pool<typename CONFIG_T::accum_t, CONFIG_T::pool_width, CONFIG_T>(pool_window);
        }

        // Write to output
        res.write(res_pack);
    }

    // Counter Housekeeping
    if (pX + 1 == CONFIG_T::n_in) // Includes padding, end of line (padded)
    {
        pX = 0;
        sX = 0;
    } else {
        pX = pX + 1;
        // Update stride (threshold) ? subtract stride : increment stride
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling1d_buffer_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

ReadInputWidth:
    for (unsigned i_iw = 0; i_iw < CONFIG_T::n_in; i_iw++) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE
        compute_pool_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pooling1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    #pragma HLS inline recursive
    switch (CONFIG_T::implementation) {
    case conv_implementation::linebuffer:
        pooling1d_buffer_cl<data_T, res_T, CONFIG_T>(data, res);
        break;
    case conv_implementation::encoded:
        pooling1d_encoded_cl<data_T, res_T, CONFIG_T>(data, res);
        break;
    }
}

// *************************************************
//       Global max/average pooling
// *************************************************

template <class T, int N, class CONFIG_T> T reduce_global_pool(T x, T y[N]) {
    #pragma HLS INLINE
    if (CONFIG_T::pool_op == Max) {
        Op_max<T> op_max;
        T y_max = reduce<T, N, Op_max<T>>(y, op_max);
        return (x > y_max) ? x : y_max;
    } else {
        Op_add<T> op_add;
        T y_sum = reduce<T, N, Op_add<T>>(y, op_add);
        return x + y_sum;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void compute_global_pool(const data_T &in_elem, typename CONFIG_T::accum_t data_window[CONFIG_T::n_filt]) {
PoolFilt:
    for (unsigned c = 0; c < CONFIG_T::n_filt; c++) {
        #pragma HLS UNROLL

        typename CONFIG_T::accum_t data_pack[data_T::size / CONFIG_T::n_filt];
        #pragma HLS ARRAY_PARTITION variable=data_pack complete dim=0

    PixelLoop:
        for (unsigned p = 0; p < data_T::size / CONFIG_T::n_filt; p++) {
            #pragma HLS UNROLL
            data_pack[p] = in_elem[p * CONFIG_T::n_filt + c];
        }
        data_window[c] = reduce_global_pool<typename CONFIG_T::accum_t, data_T::size / CONFIG_T::n_filt, CONFIG_T>(
            data_window[c], data_pack
        );
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling2d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_height == CONFIG_T::stride_height && CONFIG_T::pool_width == CONFIG_T::stride_width);

    typename CONFIG_T::accum_t data_window[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=data_window complete

    typename CONFIG_T::accum_t init = 0;
    if (CONFIG_T::pool_op == Max) {
        init = hls::numeric_limits<typename CONFIG_T::accum_t>::min();
    }

PoolInitLoop:
    for (unsigned i_init = 0; i_init < CONFIG_T::n_filt; i_init++) {
        #pragma HLS UNROLL
        data_window[i_init] = init;
    }

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_filt); i_iw++) {
            #pragma HLS LOOP_FLATTEN
            compute_global_pool<data_T, res_T, CONFIG_T>(data.read(), data_window);
        }
    }

    if (CONFIG_T::pool_op == Max) {
    MaxPoolRes:
        for (unsigned i_res = 0; i_res < CONFIG_T::n_filt / res_T::size; i_res++) {
            #pragma HLS PIPELINE

            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        MaxPoolPack:
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = data_window[i_pack];
            }
            res.write(res_pack);
        }
    } else {
    AvgPoolRes:
        for (unsigned i_res = 0; i_res < CONFIG_T::n_filt / res_T::size; i_res++) {
            #pragma HLS PIPELINE

            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        AvgPoolPack:
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = data_window[i_pack] / (CONFIG_T::in_height * CONFIG_T::in_width);
            }
            res.write(res_pack);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void global_pooling1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::pool_width == CONFIG_T::stride_width);

    typename CONFIG_T::accum_t data_window[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=data_window complete

    typename CONFIG_T::accum_t init = 0;
    if (CONFIG_T::pool_op == Max) {
        init = hls::numeric_limits<typename CONFIG_T::accum_t>::min();
    }

PoolInitLoop:
    for (unsigned i_init = 0; i_init < CONFIG_T::n_filt; i_init++) {
        #pragma HLS UNROLL
        data_window[i_init] = init;
    }

ReadInput:
    for (unsigned i_iw = 0; i_iw < CONFIG_T::n_in / (data_T::size / CONFIG_T::n_filt); i_iw++) {
        #pragma HLS LOOP_FLATTEN
        compute_global_pool<data_T, res_T, CONFIG_T>(data.read(), data_window);
    }

    if (CONFIG_T::pool_op == Max) {
    MaxPoolRes:
        for (unsigned i_res = 0; i_res < CONFIG_T::n_filt / res_T::size; i_res++) {
            #pragma HLS PIPELINE

            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        MaxPoolPack:
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = data_window[i_pack];
            }
            res.write(res_pack);
        }
    } else {
    AvgPoolRes:
        for (unsigned i_res = 0; i_res < CONFIG_T::n_filt / res_T::size; i_res++) {
            #pragma HLS PIPELINE

            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        AvgPoolPack:
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = data_window[i_pack] / CONFIG_T::n_in;
            }
            res.write(res_pack);
        }
    }
}

} // namespace nnet

#endif
