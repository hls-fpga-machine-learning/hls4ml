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

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_rf_leq_nin(
    data_T data[CONFIG_T::n_in],//1->3
    res_T  res[CONFIG_T::n_out],//1->4
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],//1->12
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {//1->4

    const int rufactor = CONFIG_T::reuse_factor;//at first 1
    const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);//at first 3 and 1 so 1
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor);//at first, 12,1 so 12
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);//at first, 12,1 so 12
    const int multscale = multiplier_limit/CONFIG_T::n_out;//at first 12/4=3
    const int nin = CONFIG_T::n_in;//1->3
    const int nout = CONFIG_T::n_out;//1->4

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");//(12%4=0||1>!=3)
    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_IN");//12==12

    #pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete
	//#pragma HLS PIPELINE //New Addition// commenting does not work
    InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {//1
        #pragma HLS PIPELINE II=1 rewind

        int w_index = ir;
        int in_index = ir;
        int out_index = 0;
        int acc_step = 0;

        MultLoop:
        for (int im = 0; im < block_factor; im++) {//12
            #pragma HLS UNROLL
        	/*---Following are C++11 features. Not suppported by Vivado. Commenting...---*/
            //acc[out_index] += product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(data[in_index], weights[w_index]);
        	acc[out_index] += (data[in_index] * weights[w_index]);
            // Increment w_index
            w_index += rufactor;
            // Increment in_index
            in_index += rufactor;
            if (in_index >= nin) {
                in_index = ir;
            }
            // Increment out_index
            if (acc_step + 1 >= multscale) {//3
                acc_step = 0;
                out_index++;
            } else {
                acc_step++;
            }
        }
    }

    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        /*---Following are C++11 features. Not suppported by Vivado. Commenting...---*/
        //res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
        res[ires] = (res_T) (acc[ires]); //res_T may need a checking
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_rf_gt_nin_rem0(
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
    assert((rufactor > nin && rufactor % nin == 0) && "This function is correct only for RF > N_IN && RF % N_IN == 0");

    #pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    int w_index;
    int in_index = 0;
    int out_index;
    int outstep = 0;
    const int outscale = rufactor / nin;

    int outidx[rufactor];
    IndexLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        if ((ir + 1) % nin == 0) {
            outstep++;
        }
    }

    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        w_index = ir;
        out_index = outidx[ir]/*outstep*/;

        MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
        	/*---Following are C++11 features. Not suppported by Vivado. Commenting...---*/
            //acc[out_index] += product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(data[in_index], weights[w_index]);
        	acc[out_index] += (data[in_index] * weights[w_index]);
            w_index += rufactor;
            if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out) break; // check out of bounds
            out_index += outscale;
        }

        in_index++;
        if (in_index >= nin) {
            in_index = 0;
            //outstep++; // This causes a huge increase in scheduling and RTL generation times, hence the above workaround.
        }
    }

    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        /*---Following are C++11 features. Not suppported by Vivado. Commenting...---*/
    	//res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
        res[ires] = (res_T) (acc[ires]); //res_T may need a checking
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large_rf_gt_nin(
    data_T data[CONFIG_T::n_in],//1->3
    res_T  res[CONFIG_T::n_out],//1->4
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],//1->12
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {//1->4

    const int rufactor = CONFIG_T::reuse_factor;//7
    const int multfactor = MIN(CONFIG_T::n_in,CONFIG_T::reuse_factor);//(3,7)->3
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, multfactor);//4
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);//2
    const int multscale = multiplier_limit/CONFIG_T::n_out;//1
    const int nin = CONFIG_T::n_in;//3
    const int nout = CONFIG_T::n_out;//4

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin) && "This function is correct only for RF > N_IN");

    #pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        typename CONFIG_T::accum_t tmpmult[block_factor];//2
        #pragma HLS ARRAY_PARTITION variable=tmpmult complete

        MultLoop:
        for (int im = 0; im < block_factor; im++) {//2
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int in_index = w_index % nin;
            if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue; // check out of bounds
            /*---Following are C++11 features. Not suppported by Vivado. Commenting...---*/
            //tmpmult[im] = product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(data[in_index], weights[w_index]);
            tmpmult[im] = (data[in_index] * weights[w_index]);
        }

        typename CONFIG_T::accum_t mult[multiplier_limit];
        #pragma HLS ARRAY_PARTITION variable=mult complete

        ResetMult:
        for (int imult = 0; imult < multiplier_limit; imult++) {//4
            #pragma HLS UNROLL
            mult[imult] = 0;
        }

        AccumLoop1:
        for (int im = 0; im < block_factor; im++) {//2
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int out_index = w_index / multfactor;
            if (out_index >= multiplier_limit) continue; // check out of bounds
            mult[out_index] += tmpmult[im];
        }

        AccumLoop2:
        for (int im = 0; im < multiplier_limit; im++) {//4
            #pragma HLS UNROLL
            //int out_index = im/multscale; // This is the general case
            //acc[out_index] += mult[im];
            acc[im] += mult[im]; // If RF > N_IN then multiplier_limit == n_out
        }
    }

    // Cast to "res_t" type
    Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {//4
        #pragma HLS UNROLL
    	/*---Following are C++11 features. Not suppported by Vivado. Commenting...---*/
    	//res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    	res[ires] = (res_T) (acc[ires]); //res_T may need a checking
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_large(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]) {

    #pragma HLS INLINE region

    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        dense_large_rf_leq_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
        std::cout<<"dense_large_rf_leq_nin"<<std::endl;
    } else if (CONFIG_T::reuse_factor % CONFIG_T::n_in == 0) {
        dense_large_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
        std::cout<<"dense_large_rf_gt_nin_rem0"<<std::endl;
    } else {
        dense_large_rf_gt_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
        std::cout<<"dense_large_rf_gt_nin"<<std::endl;
    }
}
/// GNN Dense-Batch addition
template<class data_T, class res_T, typename CONFIG_T>
void dense_batch(
		data_T    data[CONFIG_T::n_batch][CONFIG_T::n_in],//1->9x3
		res_T     res[CONFIG_T::n_batch][CONFIG_T::n_out],//1->9x4
        typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],//W1->12
        typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])//->b1->4
{
	std::cout<<CONFIG_T::n_in<<std::endl;//1->3
	data_T data_temp[CONFIG_T::n_in];//1->3
    res_T res_temp[CONFIG_T::n_out];//1->4
    //New Test to reduce latency
    //#pragma HLS PIPELINE //Speeds up but resuse factor does not work
    for (int bb = 0; bb < CONFIG_T::n_batch; bb++) {//1->9
     	//New Test to reduce latency.
		#pragma HLS PIPELINE
    	for (int ii = 0; ii < CONFIG_T::n_in; ii++) {//1->3
    		data_temp[ii] = data[bb][ii];
    	}
    	//#pragma HLS ALLOCATION instances=compute_layer limit=10
        dense_large<data_T, res_T, CONFIG_T>(data_temp, res_temp, weights, biases);
        for (int ii = 0; ii < CONFIG_T::n_out; ii++) {//1->4
          res[bb][ii] = res_temp[ii];
        }
    }
}
}

#endif
