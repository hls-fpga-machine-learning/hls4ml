//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_CONV2D_H_
#define NNET_CONV2D_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet {

struct conv2d_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Convolutional parameters
    static const unsigned pad_top = 4;
    static const unsigned pad_bottom = 5;
    static const unsigned pad_left = 4;
    static const unsigned pad_right = 5;
    static const unsigned in_height = 128;
    static const unsigned in_width = 128;
    static const unsigned n_chan = 9;
    static const unsigned filt_height = 10;
    static const unsigned filt_width = 10;
    static const unsigned n_filt = 4;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 128;
    static const unsigned out_width = 128;

    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0; // not used yet
};


//Computes multiplier limit
//This function should not be synthesized into firmware
template<typename CONFIG_T>
    int compute_multiplier_limit_conv2d(
    typename CONFIG_T::weight_t  weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt]
)
{
    int n_mult = 0;

    for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
            for(int ff = 0; ff < CONFIG_T::n_filt; ff++){
                for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                    for(int fh = 0; fh < CONFIG_T::filt_height; fh++){
                        for(int fw = 0; fw < CONFIG_T::filt_width; fw++){

                                int index_weight = fh*CONFIG_T::filt_width*CONFIG_T::n_chan*CONFIG_T::n_filt
                                                 + fw*CONFIG_T::n_chan*CONFIG_T::n_filt
                                                 + cc*CONFIG_T::n_filt
                                                  + ff;

                                if ((oh*CONFIG_T::stride_height+fh) < CONFIG_T::pad_top
                                || (oh*CONFIG_T::stride_height+fh) >= (CONFIG_T::pad_top+CONFIG_T::in_height)
                                || (ow*CONFIG_T::stride_width+fw) < CONFIG_T::pad_left
                                || (ow*CONFIG_T::stride_width+fw) >= (CONFIG_T::pad_left+CONFIG_T::in_width)) {
                                    //padded - do nothing
                                    continue;
                                } else {
                                    if (weights[index_weight] > 1e-20 || weights[index_weight] < -1e-20) {
                                          n_mult++;
                                    }
                                }

                        }//end mult loop
                    }//end channel loop
                }//end filter width loop
            }//end filter height loop
        }//end output width loop
    }//end output height loop

    return ceil( float(n_mult) / float(CONFIG_T::reuse_factor) );

}//end compute_n_mult


template<class data_T, class res_T, typename CONFIG_T>
void conv_2d(
    data_T data[CONFIG_T::in_height*CONFIG_T::in_width*CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height*CONFIG_T::out_width*CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{

    typename CONFIG_T::accum_t mult[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt * CONFIG_T::n_chan * CONFIG_T::filt_height * CONFIG_T::filt_width];
    typename CONFIG_T::accum_t acc[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    // Parallel mode
    #pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=biases complete dim=0

    // Limit multipliers to control parallelization
    const int multiplier_limit = compute_multiplier_limit_conv2d<CONFIG_T>(weights);
    #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    // Convolve, saving all multiplication results to accumulate later
    ConvOutHeight: for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        ConvOutWidth: for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
            ConvFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++){
                ConvChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                    ConvFiltHeight: for(int fh = 0; fh < CONFIG_T::filt_height; fh++){
                        ConvFiltWidth: for(int fw = 0; fw < CONFIG_T::filt_width; fw++){

                            int index_mult = oh*CONFIG_T::out_width*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + ow*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                                + ff*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                               + cc*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + fh*CONFIG_T::filt_width
                                                + fw;

                                int index_weight = fh*CONFIG_T::filt_width*CONFIG_T::n_chan*CONFIG_T::n_filt
                                                 + fw*CONFIG_T::n_chan*CONFIG_T::n_filt
                                                 + cc*CONFIG_T::n_filt
                                                  + ff;

                                if ((oh*CONFIG_T::stride_height+fh) < CONFIG_T::pad_top
                                || (oh*CONFIG_T::stride_height+fh) >= (CONFIG_T::pad_top+CONFIG_T::in_height)
                                || (ow*CONFIG_T::stride_width+fw) < CONFIG_T::pad_left
                                || (ow*CONFIG_T::stride_width+fw) >= (CONFIG_T::pad_left+CONFIG_T::in_width)) {
                                    mult[index_mult] = 0;
                                } else {
                                    int index_data = (oh*CONFIG_T::stride_height+fh-CONFIG_T::pad_top)*CONFIG_T::in_width*CONFIG_T::n_chan
                                                   + (ow*CONFIG_T::stride_width+fw-CONFIG_T::pad_left)*CONFIG_T::n_chan
                                                   + cc;
                                    mult[index_mult] = data[index_data] * weights[index_weight];
                                }

                        }//end mult loop
                    }//end channel loop
                  }//end filter width loop
            }//end filter height loop
        }//end output width loop
    }//end output height loop


    // Initialize accumulator with input biases
    for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
            for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                acc[oh*CONFIG_T::out_width*CONFIG_T::n_filt + ow*CONFIG_T::n_filt + ff]=biases[ff];
            }
        }
    }


    // Accumulate multiplication result
    AccumOutHeight: for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        AccumOutWidth: for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
            AccumFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                //Do "dot product" sum within filter and sum over channels
                AccumChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                    AccumDotHeight: for(int fh = 0; fh < CONFIG_T::filt_height; fh++){
                        AccumDotWidth: for(int fw = 0; fw < CONFIG_T::filt_width; fw++){

                            int index_mult = oh*CONFIG_T::out_width*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + ow*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                                + ff*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                               + cc*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + fh*CONFIG_T::filt_width
                                                + fw;
                            int index_acc = oh*CONFIG_T::out_width*CONFIG_T::n_filt
                                          + ow*CONFIG_T::n_filt
                                          + ff;

                            acc[index_acc] += mult[index_mult];

                        }//end dot product filter width loop
                    }//end dot product filter height loop
                }//end n channel loop
            }//end n filter loop
        }//end output width loop
    }//end output height loop

     // Cast to "res_t" type
    for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
              for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                int index = oh*CONFIG_T::out_width*CONFIG_T::n_filt + ow*CONFIG_T::n_filt + ff;
                res[index] = (res_T)(acc[index]);
            }
        }
    }

}//end conv2d


template<class data_T, int N1, int N2, int N3>
void flatten(
    data_T data[N1][N2][N3],
    data_T res[N1*N2*N3]
)
{
    for(int i1=0; i1<N1; i1++){
        for(int i2=0; i2<N2; i2++){
            for(int i3=0; i3<N3; i3++){
                res[i1*N2*N3+i2*N3+i3] = data[i1][i2][i3];
            }//i3
        }//i2
    }//i1
}


template<class data_T, int N1, int N2, int N3>
void unflatten(
    data_T data[N1*N2*N3],
    data_T res[N1][N2][N3]
)
{
    for(int i1=0; i1<N1; i1++){
        for(int i2=0; i2<N2; i2++){
            for(int i3=0; i3<N3; i3++){
                res[i1][i2][i3] = data[i1*N2*N3+i2*N3+i3];
            }//i3
        }//i2
    }//i1
}


}//end namespace

#endif
