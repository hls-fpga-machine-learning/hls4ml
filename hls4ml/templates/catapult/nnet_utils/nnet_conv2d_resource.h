#ifndef NNET_CONV2D_RESOURCE_H_
#define NNET_CONV2D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"

namespace nnet {

template <class data_T, typename CONFIG_T>
void im2col_2d(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col
        [CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::out_height * CONFIG_T::out_width]
) {
    const int output_h = (CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom -
                          (CONFIG_T::dilation_height * (CONFIG_T::filt_height - 1) + 1)) /
                             CONFIG_T::stride_height +
                         1;
    const int output_w = (CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right -
                          (CONFIG_T::dilation_width * (CONFIG_T::filt_width - 1) + 1)) /
                             CONFIG_T::stride_width +
                         1;
    const int channel_size = CONFIG_T::in_height * CONFIG_T::in_width;

    for (int channel = CONFIG_T::n_chan; channel--; data += channel_size) {
        for (int kernel_row = 0; kernel_row < CONFIG_T::filt_height; kernel_row++) {
            for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
                int input_row = -CONFIG_T::pad_top + kernel_row * CONFIG_T::dilation_height;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (input_row < 0 || input_row > CONFIG_T::in_height) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation_width;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                                *(data_col++) = data[input_row * CONFIG_T::in_width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += CONFIG_T::stride_width;
                        }
                    }
                    input_row += CONFIG_T::stride_height;
                }
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_full(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    data_T data_conv
        [CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::out_height * CONFIG_T::out_width];
    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    ////#pragma HLS ARRAY_PARTITION variable=data_conv complete
    //#pragma HLS ARRAY_PARTITION variable=data_col complete
    //#pragma HLS ARRAY_PARTITION variable=res_col complete

    im2col_2d<data_T, CONFIG_T>(data, data_conv);

    for (int i = 0; i < CONFIG_T::out_height * CONFIG_T::out_width; i++) {
        for (int j = 0; j < CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan; j++) {
            data_col[j] = data[j * CONFIG_T::out_height * CONFIG_T::out_width + i];
        }
        dense<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            // res[i * CONFIG_T::n_filt + j] = res_col[j];
            res[j * CONFIG_T::out_height * CONFIG_T::out_width + i] = res_col[j]; // Transposed order
        }
    }
}

template <class data_T, typename CONFIG_T>
void im2col_2d_cf(
    data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
    data_T data_col[CONFIG_T::n_chan * CONFIG_T::filt_height * CONFIG_T::filt_width],
    const int row,
    const int col
) {
    const int channel_size = CONFIG_T::in_height * CONFIG_T::in_width;
    int index = 0;
    for (int channel = CONFIG_T::n_chan; channel--; data += channel_size) {
        for (int kernel_row = 0; kernel_row < CONFIG_T::filt_height; kernel_row++) {
            int input_row = -CONFIG_T::pad_top + kernel_row * CONFIG_T::dilation_height + row * CONFIG_T::stride_height;
            for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
                if (input_row < 0 || input_row > CONFIG_T::in_height) {
                    data_col[index++] = 0;
                } else {
                    int input_col =
                        -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation_width + col * CONFIG_T::stride_width;
                    if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                        //*(data_col++) = data[input_row * CONFIG_T::in_width + input_col];
                        data_col[index++] = data[input_row * CONFIG_T::in_width + input_col];
                    } else {
                        //*(data_col++) = 0;
                        data_col[index++] = 0;
                    }
                    input_col += CONFIG_T::stride_width;
                }
            }
            input_row += CONFIG_T::stride_height;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_resource_cf(
    data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    const int nin = CONFIG_T::n_chan * CONFIG_T::filt_width;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin * nout, rufactor);

    ////#pragma HLS function_instantiate variable=weights,biases
    ////#pragma HLS RESOURCE         variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose
    /// correctly
    ////#pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    ////#pragma HLS ARRAY_PARTITION variable=biases complete

    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    //#pragma HLS ARRAY_PARTITION variable=data_col complete
    //#pragma HLS ARRAY_PARTITION variable=res_col complete

HeightLoop:
    for (int i = 0; i < CONFIG_T::out_height; i++) {
    WidthLoop:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            //#pragma HLS PIPELINE
            im2col_2d_cf<data_T, CONFIG_T>(data, data_col, i, j);
            dense<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        FiltLoop:
            for (int k = 0; k < CONFIG_T::n_filt; k++) {
                // res[i * CONFIG_T::out_width * CONFIG_T::n_filt + j * CONFIG_T::n_filt + k] = res_col[k];
                res[k * CONFIG_T::out_height * CONFIG_T::out_width + i * CONFIG_T::out_width + j] =
                    res_col[k]; // Transposed order
            }
        }
    }
}

template <class data_T, typename CONFIG_T>
void im2col_2d_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    const int row,
    const int col
) {
    int index = 0;
    for (int kernel_row = 0; kernel_row < CONFIG_T::filt_height; kernel_row++) {
        int input_row = -CONFIG_T::pad_top + kernel_row * CONFIG_T::dilation_height + row * CONFIG_T::stride_height;
        for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
            for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
                if (input_row < 0 || input_row >= CONFIG_T::in_height) {
                    data_col[index++] = 0;
                } else {
                    int input_col =
                        -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation_width + col * CONFIG_T::stride_width;
                    if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                        data_col[index++] =
                            data[input_row * CONFIG_T::in_width * CONFIG_T::n_chan + input_col * CONFIG_T::n_chan + channel];
                    } else {
                        data_col[index++] = 0;
                    }
                }
            }
        }
    }
}

template <class data_T, typename CONFIG_T>
void im2col_2d_pointwise_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::n_chan],
    const int row,
    const int col
) {
    int index = 0;
    int input_row = -CONFIG_T::pad_top + row * CONFIG_T::stride_height;

ChannelLoop:
    for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
        if (input_row < 0 || input_row >= CONFIG_T::in_height) {
            data_col[index++] = 0;
        } else {
            int input_col = -CONFIG_T::pad_left + col * CONFIG_T::stride_width;
            if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                data_col[index++] =
                    data[input_row * CONFIG_T::in_width * CONFIG_T::n_chan + input_col * CONFIG_T::n_chan + channel];
            } else {
                data_col[index++] = 0;
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_resource_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    const int nin = CONFIG_T::n_chan * CONFIG_T::filt_width;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin * nout, rufactor);

    ////#pragma HLS function_instantiate variable=weights,biases
    ////#pragma HLS RESOURCE         variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose
    /// correctly
    ////#pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    ////#pragma HLS ARRAY_PARTITION variable=biases complete

    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    //#pragma HLS ARRAY_PARTITION variable=data_col complete
    //#pragma HLS ARRAY_PARTITION variable=res_col complete

HeightLoop:
    for (int i = 0; i < CONFIG_T::out_height; i++) {
    WidthLoop:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            //#pragma HLS PIPELINE
            im2col_2d_cl<data_T, CONFIG_T>(data, data_col, i, j);
            dense<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        FiltLoop:
            for (int k = 0; k < CONFIG_T::n_filt; k++) {
                res[i * CONFIG_T::out_width * CONFIG_T::n_filt + j * CONFIG_T::n_filt + k] = res_col[k];
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_2d_resource_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    assert(CONFIG_T::filt_height == 1 && CONFIG_T::filt_width == 1);

    const int nin = CONFIG_T::n_chan;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin * nout, rufactor);

    ////#pragma HLS function_instantiate variable=weights,biases
    ////#pragma HLS RESOURCE         variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose
    /// correctly
    ////#pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    ////#pragma HLS ARRAY_PARTITION variable=biases complete

    data_T data_col[CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    //#pragma HLS ARRAY_PARTITION variable=data_col complete
    //#pragma HLS ARRAY_PARTITION variable=res_col complete

HeightLoop:
    for (int i = 0; i < CONFIG_T::out_height; i++) {
    WidthLoop:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            //#pragma HLS PIPELINE
            im2col_2d_pointwise_cl<data_T, CONFIG_T>(data, data_col, i, j);
            dense<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        FiltLoop:
            for (int k = 0; k < CONFIG_T::n_filt; k++) {
                res[i * CONFIG_T::out_width * CONFIG_T::n_filt + j * CONFIG_T::n_filt + k] = res_col[k];
            }
        }
    }
}

} // namespace nnet
#endif
