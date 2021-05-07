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

#ifndef NNET_LARGE_LAYER_H_
#define NNET_LARGE_LAYER_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "hls_stream.h"
#include <math.h>
#include <assert.h>

// This is a substitute for "ceil(n/(float)d)".
#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_large(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit/CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");

    #pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    InitAccum:
    for(int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        typename CONFIG_T::accum_t tmpmult[block_factor];
        #pragma HLS ARRAY_PARTITION variable=tmpmult complete

        MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int in_index = w_index % nin;
            if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue; // check out of bounds
                //tmpmult[im] = product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(data[in_index], weights[w_index]);
            	//The commented line above is the original code of Vladimir. But it was not working for GNN so we opted to following line
            	tmpmult[im] = data[in_index] * weights[w_index]; //Also suggested by Vladimir
            }

            typename CONFIG_T::accum_t mult[multiplier_limit];
            #pragma HLS ARRAY_PARTITION variable=mult complete

        ResetMult:
        for (int imult = 0; imult < multiplier_limit; imult++) {
            #pragma HLS UNROLL
            mult[imult] = 0;
        }

        AccumLoop1:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int out_index = w_index / multfactor;
            if (out_index >= multiplier_limit) continue; // check out of bounds
                mult[out_index] += tmpmult[im];
            }

        AccumLoop2:
        for (int im = 0; im < multiplier_limit; im++) {
            #pragma HLS UNROLL
            int out_index = im/multscale;
            acc[out_index] += mult[im];
        }
    }

    // Cast to "res_t" type
    Result:
    for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        #pragma HLS UNROLL
        //res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    	//The line above was original.
    	res[ires] = (res_T) (acc[ires]);
    }
}
/// GNN Dense-Batch addition
template<class data_T, class res_T, typename CONFIG_T>
void dense_batch(
		data_T    data[CONFIG_T::n_batch][CONFIG_T::n_in],
		res_T     res[CONFIG_T::n_batch][CONFIG_T::n_out],
        typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
        typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
	data_T data_temp[CONFIG_T::n_in];
    res_T res_temp[CONFIG_T::n_out];
    for (int bb = 0; bb < CONFIG_T::n_batch; bb++) {
    	for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
    		data_temp[ii] = data[bb][ii];
    	}
    	//#pragma HLS ALLOCATION instances=compute_layer limit=10
        dense_large<data_T, res_T, CONFIG_T>(data_temp, res_temp, weights, biases);
        for (int ii = 0; ii < CONFIG_T::n_out; ii++) {
          res[bb][ii] = res_temp[ii];
        }
    }
}
}

#endif
