#ifndef NNET_PADDING_STREAM_H_
#define NNET_PADDING_STREAM_H_

#include <math.h>

namespace nnet {

template <class res_T, typename CONFIG_T> void fill_zero(hls::stream<res_T> &res) {
    #pragma HLS INLINE
    res_T res_part;
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
        res_part[c] = 0;
    }
    res.write(res_part);
}

template <class data_T, class res_T, typename CONFIG_T> void fill_data(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    #pragma HLS INLINE
    data_T data_part = data.read();
    res_T res_part;
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
        res_part[c] = data_part[c];
    }
    res.write(res_part);
}

template <class data_T, class res_T, typename CONFIG_T>
void zeropad1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
PadLeft:
    for (int i = 0; i < CONFIG_T::pad_left; i++) {
        fill_zero<res_T, CONFIG_T>(res);
    }

CopyMain:
    for (int i = 0; i < CONFIG_T::in_width; i++) {
        fill_data<data_T, res_T, CONFIG_T>(data, res);
    }

PadRight:
    for (int i = 0; i < CONFIG_T::pad_right; i++) {
        fill_zero<res_T, CONFIG_T>(res);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {

PadTop:
    for (int i = 0; i < CONFIG_T::pad_top; i++) {
    PadTopWidth:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
    }

PadMain:
    for (int i = 0; i < CONFIG_T::in_height; i++) {
        #pragma HLS PIPELINE II=1
    PadLeft:
        for (int j = 0; j < CONFIG_T::pad_left; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
    CopyMain:
        for (int j = 0; j < CONFIG_T::in_width; j++) {
            fill_data<data_T, res_T, CONFIG_T>(data, res);
        }
    PadRight:
        for (int j = 0; j < CONFIG_T::pad_right; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
    }

PadBottom:
    for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
    PadBottomWidth:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
    }
}

} // namespace nnet

#endif
