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

#ifndef NNET_DENSE_H_
#define NNET_DENSE_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct dense_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_batch = 1;
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
};

/* ---
 * 3 different methods to perform the product of input and weight, depending on the
 * types of each. Use std::enable_if<>::type for the return type since partial
 * template specification is not allowed by c++
 * --- */
/*
template<class data_T, class weight_T, class ret_T>
inline typename std::enable_if<std::is_same<data_T, ap_uint<1>>::value
        and std::is_same<weight_T, ap_uint<1>>::value, ap_uint<1>>::type
product(ap_uint<1> a, ap_uint<1> w){
    // specialisation for 1-bit weights and incoming data
    #pragma HLS inline off
    return a == w;
}

template<class data_T, class weight_T, class ret_T>
inline typename std::enable_if<(not std::is_same<data_T, ap_uint<1>>::value)
        and std::is_same<weight_T, ap_uint<1>>::value, ret_T>::type
product(data_T a, ap_uint<1> w){
    // Specialisation for 1-bit weights, arbitrary data
    #pragma HLS inline off
    return w == 0 ? (data_T) -a : a;
}

template<class data_T, class weight_T, class ret_T>
inline typename std::enable_if<(not std::is_same<data_T, ap_uint<1>>::value)
        and (not std::is_same<weight_T, ap_uint<1>>::value), ret_T>::type
product(data_T a, weight_T w){
    // 'Normal' product
    #pragma HLS inline off
    return a * w;
}

template<class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<std::is_same<data_T, ap_uint<1>>::value
        and std::is_same<typename CONFIG_T::weight_t, ap_uint<1>>::value, ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>>::type
cast(typename CONFIG_T::accum_t x){
  return (ap_int<nnet::ceillog2(CONFIG_T::n_in) + 2>) (x - CONFIG_T::n_in / 2) * 2;
}

template<class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<(not std::is_same<data_T, ap_uint<1>>::value), res_T>::type
cast(typename CONFIG_T::accum_t x){
  return (res_T) x;
}
*/
template<class data_T, class res_T, typename CONFIG_T>
void dense(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    if (CONFIG_T::io_type == io_parallel){
        // For parallel inputs:
        //   - completely partition arrays -- target fabric
        //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
		
        int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
        //#pragma HLS ALLOCATION instances=product limit=multiplier_limit function //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
		#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    } else if (CONFIG_T::io_type == io_serial){
        // Only reduce cycle_factor if n_out is evenly divisible by reuse_factor
        // Otherwise, HLS wont be happy
        int cycle_factor = CONFIG_T::n_out;
        float reused_cycle = CONFIG_T::n_out / CONFIG_T::reuse_factor;
        if (reused_cycle == ceil(reused_cycle)){
            // Dont use "ceil" here; as of 2018.2, HLS crashes mysteriously
            cycle_factor = cycle_factor / CONFIG_T::reuse_factor;
        }
        #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=mult cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS DATAFLOW
        #pragma HLS STREAM variable=mult depth=1
        #pragma HLS STREAM variable=acc depth=1
        if (CONFIG_T::store_weights_in_bram){
            #pragma HLS RESOURCE variable=weights core=ROM_2P_BRAM
        }
    }

    // Do the matrix-multiply
    Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        cache = data[ii];
        Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                //#pragma HLS ALLOCATION instances=product limit=multiplier_limit function //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
				#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
        int index = ii*CONFIG_T::n_out+jj;
        //mult[index] = product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(cache, weights[index]); //mult[index] = cache * weights[index];
        mult[index] = cache * weights[index];
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
        int index = ii*CONFIG_T::n_out+jj;
        acc[jj] += mult[index];
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = (res_T) (acc[ires]);
		//res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

//template<class data_T, class res_T, typename CONFIG_T>
//void dense_batch(
//    data_T    data[CONFIG_T::n_batch][CONFIG_T::n_in],
//    res_T     res[CONFIG_T::n_batch][CONFIG_T::n_out],
//    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
//    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
//{
//  data_T data_temp[CONFIG_T::n_in];
//  res_T res_temp[CONFIG_T::n_out];
//  for (int bb = 0; bb < CONFIG_T::n_batch; bb++) {
//    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
//      data_temp[ii] = data[bb][ii];
//    }
//    //#pragma HLS ALLOCATION instances=compute_layer limit=10
//    dense<data_T, res_T, CONFIG_T>(data_temp, res_temp, weights, biases);
//    for (int ii = 0; ii < CONFIG_T::n_out; ii++) {
//      res[bb][ii] = res_temp[ii];
//    }
//  }
//}
}

#endif
