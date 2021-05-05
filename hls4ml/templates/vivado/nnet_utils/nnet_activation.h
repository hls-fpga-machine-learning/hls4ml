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

#ifndef NNET_ACTIVATION_H_
#define NNET_ACTIVATION_H_

#include <cmath>
#include "ap_fixed.h"
#include "nnet_common.h"

namespace nnet {

struct activ_config
{
    // IO size
    static const unsigned n_in = 10;

    // Internal info
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef ap_fixed<18,8> table_t;
};

// *************************************************
//       LINEAR Activation -- See Issue 53
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  linear(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        res[ii] = data[ii];
    }
}



// *************************************************
//       RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  relu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        datareg = data[ii];
        if (datareg > 0) res[ii] = datareg;
        else res[ii] = 0;
    }
}

template<class data_T, class res_T, int MAX_INT, typename CONFIG_T>
void  relu_max(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        datareg = data[ii];
        if (datareg < 0) res[ii] = 0;
        else if (datareg > MAX_INT) res[ii] = MAX_INT;
        else res[ii] = datareg;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  relu6(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    relu_max<data_T, res_T, 6, CONFIG_T>(data, res);
}

template<class data_T, class res_T, typename CONFIG_T>
void  relu1(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    relu_max<data_T, res_T, 1, CONFIG_T>(data, res);
}

// *************************************************
//       Sigmoid Activation
// *************************************************
inline float sigmoid_fcn_float(float input) {
    return 1.0 / (1 + std::exp(-input));
}

template<typename CONFIG_T, int N_TABLE>
void init_sigmoid_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default logistic sigmoid function:
    //   result = 1/(1+e^(-x))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = sigmoid_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(sigmoid_table);
        initialized = true;
    }

    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_round = data[ii]*CONFIG_T::table_size/16;
        index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) sigmoid_table[index];
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

enum class softmax_implementation {latency=0, legacy=1, stable=2};

inline float exp_fcn_float(float input) {
    return std::exp(input);
}

template<class data_T, typename CONFIG_T>
inline float softmax_real_val_from_idx(unsigned i){
    // Treat the index as the top N bits
    static constexpr int N = ceillog2(CONFIG_T::table_size); // number of address bits for table
    data_T x(0);
    x(x.width-1, x.width-N) = i;
    return (float) x;
}

template<class data_T, typename CONFIG_T>
inline unsigned softmax_idx_from_real_val(data_T x){
    // Slice the top N bits to get an index into the table
    static constexpr int N = ceillog2(CONFIG_T::table_size); // number of address bits for table
    ap_uint<N> y = x(x.width-1, x.width-N); // slice the top N bits of input
    return (unsigned) y(N-1, 0);
}

template<class data_T, typename CONFIG_T>
void init_exp_table(typename CONFIG_T::exp_table_t table_out[CONFIG_T::table_size]){
    // The template data_T is the data type used to address the table
    for(unsigned i = 0; i < CONFIG_T::table_size; i++){
        // Slicing bits for address is going to round towards 0, so take the central value
        float x = softmax_real_val_from_idx<data_T, CONFIG_T>(i);
        typename CONFIG_T::exp_table_t exp_x = exp_fcn_float(x);
        table_out[i] = exp_x;
    }
}

template<class data_T, typename CONFIG_T>
void init_invert_table(typename CONFIG_T::inv_table_t table_out[CONFIG_T::table_size]){
    // The template data_T is the data type used to address the table
    for(unsigned i = 0; i < CONFIG_T::table_size; i++){
        float x = softmax_real_val_from_idx<data_T, CONFIG_T>(i);
        typename CONFIG_T::inv_table_t inv_x = 1 / x;
        table_out[i] = inv_x;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]){
    #pragma HLS pipeline
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<data_T, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
	#pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);
    for(unsigned i = 0; i < CONFIG_T::n_in; i++){
        #pragma HLS unroll
    	unsigned x = softmax_idx_from_real_val<data_T, CONFIG_T>(data[i]);
        exp_res[i] = exp_table[x];
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    exp_sum = reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum = invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];
    for(unsigned i = 0; i < CONFIG_T::n_in; i++){
        #pragma HLS unroll
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]){
    #pragma HLS pipeline
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<data_T, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    // Find the max and compute all delta(x_i, x_max)
    Op_max<data_T> op_max;
    data_T x_max = reduce<data_T, CONFIG_T::n_in, Op_max<data_T>>(data, op_max);

    // For the diffs, use the same type as the input but force rounding and saturation
    ap_fixed<data_T::width, data_T::iwidth,AP_RND,AP_SAT> d_xi_xmax[CONFIG_T::n_in];
    for(unsigned i = 0; i < CONFIG_T::n_in; i++){
        #pragma HLS unroll
        d_xi_xmax[i] = data[i] - x_max;
    }

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
	#pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);
    for(unsigned i = 0; i < CONFIG_T::n_in; i++){
        #pragma HLS unroll
    	unsigned x = softmax_idx_from_real_val<data_T, CONFIG_T>(d_xi_xmax[i]);
        exp_res[i] = exp_table[x];
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    exp_sum = reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum = invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];
    for(unsigned i = 0; i < CONFIG_T::n_in; i++){
        #pragma HLS unroll
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

template<typename CONFIG_T, int N_TABLE>
void init_exp_table_legacy(typename CONFIG_T::table_t table_out[N_TABLE])
{
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = exp_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<typename CONFIG_T, int N_TABLE>
void init_invert_table_legacy(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Inversion function:
    //   result = 1/x
    for (int ii = 0; ii < N_TABLE; ii++) {
      // First, convert from table index to X-value (signed 8-bit, range 0 to +64)
	float in_val = 64.0*ii/float(N_TABLE);
        // Next, compute lookup table function
	if (in_val > 0.0) table_out[ii] = 1.0/in_val;
	else table_out[ii] = 0.0;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  softmax_legacy(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_invert_table_legacy<CONFIG_T, CONFIG_T::table_size>(invert_table);
        initialized = true;
    }

    if (CONFIG_T::io_type == io_parallel){
        // Note: This is going to be a resource hog to run with pipeline, but hey, whatever
        #pragma HLS PIPELINE
    }

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[CONFIG_T::n_in];// different, independent, fixed point precision
    typename CONFIG_T::table_t exp_diff_res;// different, independent, fixed point precision
    data_T data_cache[CONFIG_T::n_in];
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
      data_cache[ii] = data[ii];
      exp_res[ii] = 0;
    }
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
      if (CONFIG_T::io_type == io_serial){
          #pragma HLS PIPELINE
      }
      for (int jj=0; jj<CONFIG_T::n_in; jj++) {
	if (ii==jj) exp_diff_res = 1;
	else {
	  data_round = (data_cache[jj]-data_cache[ii])*CONFIG_T::table_size/16;
	  index = data_round + 8*CONFIG_T::table_size/16;
	  if (index < 0)   index = 0;
	  if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
	  exp_diff_res = exp_table[index];
	}
	exp_res[ii] += exp_diff_res;
      }
    }

    //Second loop to invert
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
      int exp_res_index = exp_res[ii]*CONFIG_T::table_size/64;
      if (exp_res_index < 0)   exp_res_index = 0;
      if (exp_res_index > CONFIG_T::table_size-1) exp_res_index = CONFIG_T::table_size-1;
      //typename CONFIG_T::table_t exp_res_invert = invert_table[exp_res_index];
      res[ii] = (res_T) invert_table[exp_res_index];
    }

}

template<class data_T, class res_T, typename CONFIG_T>
void softmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]){
    #pragma HLS inline
    switch(CONFIG_T::implementation){
    case softmax_implementation::latency:
        softmax_latency<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::stable:
        softmax_stable<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::legacy:
        softmax_legacy<data_T, res_T, CONFIG_T>(data, res);
        break;
    }    
}

// *************************************************
//       TanH Activation
// *************************************************
template<typename CONFIG_T, int N_TABLE>
void init_tanh_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Implement tanh lookup
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
        float in_val = 2*4.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = tanh(in_val);
        //std::cout << "Tanh:  Lookup table Index: " <<  ii<< " In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void  tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>(tanh_table);
        initialized = true;
    }

    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_round = data[ii]*CONFIG_T::table_size/8;
        index = data_round + 4*CONFIG_T::table_size/8;
        //std::cout << "Input: "  << data[ii] << " Round: " << data_round << " Index: " << index << std::endl;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) tanh_table[index];
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  hard_sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    data_T datareg;
    data_T slope = (data_T) 0.2;
    data_T shift = (data_T) 0.5;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        datareg = slope * data[ii] + shift;
        if (datareg > 1) datareg = 1;
        else if (datareg < 0) datareg = 0;
        res[ii] = datareg;
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  leaky_relu(data_T data[CONFIG_T::n_in], data_T alpha, res_T res[CONFIG_T::n_in])
{
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        datareg = data[ii];
        if (datareg > 0) res[ii] = datareg;
        else res[ii] = alpha * datareg;
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  thresholded_relu(data_T data[CONFIG_T::n_in], data_T theta, res_T res[CONFIG_T::n_in])
{
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        datareg = data[ii];
        if (datareg > theta) res[ii] = datareg;
        else res[ii] = 0;
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
inline float softplus_fcn_float(float input) {
    return std::log(std::exp(input) + 1.);
}

template<typename CONFIG_T, int N_TABLE>
void init_softplus_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default softplus function:
    //   result = log(exp(x) + 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = softplus_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  softplus(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t softplus_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t softplus_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_softplus_table<CONFIG_T, CONFIG_T::table_size>(softplus_table);
        initialized = true;
    }

    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_round = data[ii]*CONFIG_T::table_size/16;
        index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) softplus_table[index];
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
inline float softsign_fcn_float(float input) {
    return input / (std::abs(input) + 1.);
}

template<typename CONFIG_T, int N_TABLE>
void init_softsign_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default softsign function:
    //   result = x / (abs(x) + 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2*8.0*(ii-float(N_TABLE)/2.0)/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = softsign_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  softsign(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t softsign_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t softsign_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_softsign_table<CONFIG_T, CONFIG_T::table_size>(softsign_table);
        initialized = true;
    }

    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_round = data[ii]*CONFIG_T::table_size/16;
        index = data_round + 8*CONFIG_T::table_size/16;
        if (index < 0)   index = 0;
        if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
        res[ii] = (res_T) softsign_table[index];
    }
}

// *************************************************
//       ELU Activation
// *************************************************
inline float elu_fcn_float(float input) {
    return std::exp(input) - 1.;
}

template<typename CONFIG_T, int N_TABLE>
void init_elu_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default ELU function:
    //   result = alpha * (e^(x) - 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to 0)
        float in_val = -8.0*ii/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = elu_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  elu(data_T data[CONFIG_T::n_in], const res_T alpha, res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t elu_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t elu_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_elu_table<CONFIG_T, CONFIG_T::table_size>(elu_table);
        initialized = true;
    }

    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    data_T datareg;
    // Index into the lookup table based on data
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = datareg;
        } else {
            index = datareg*CONFIG_T::table_size/-8;
            if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            res[ii] = alpha * elu_table[index];
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  elu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
	elu<data_T, res_T, CONFIG_T>(data, 1.0, res);
}

// *************************************************
//       SELU Activation
// *************************************************
inline float selu_fcn_float(float input) {
    return 1.0507009873554804934193349852946 * (1.6732632423543772848170429916717 * (std::exp(input) - 1.));
}

template<typename CONFIG_T, int N_TABLE>
void init_selu_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    // Default SELU function:
    //   result = 1.05 * (1.673 * (e^(x) - 1))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to 0)
        float in_val = -8.0*ii/float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = selu_fcn_float(in_val);
        //std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void  selu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t selu_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t selu_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_selu_table<CONFIG_T, CONFIG_T::table_size>(selu_table);
        initialized = true;
    }

    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    data_T datareg;
    // Index into the lookup table based on data
    int index;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = res_T(1.0507009873554804934193349852946) * datareg;
        } else {
            index = datareg*CONFIG_T::table_size/-8;
            if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            res[ii] = selu_table[index];
        }
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  prelu(data_T data[CONFIG_T::n_in], data_T alpha[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
    }

    data_T datareg;
    for (int ii=0; ii<CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        datareg = data[ii];
        if (datareg > 0) res[ii] = datareg;
        else res[ii] = alpha[ii] * datareg;
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  binary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{

 if (CONFIG_T::io_type == io_parallel){
     #pragma HLS PIPELINE
 }
  
 data_T datareg;   
 res_T cache; 
 for (int ii=0; ii<CONFIG_T::n_in; ii++) {

  if (CONFIG_T::io_type == io_serial){
      #pragma HLS PIPELINE
  }
  datareg = data[ii];	 
  if( datareg > 0 ) cache = 1;
  else cache = -1;
  
  res[ii] = (res_T) cache;
 
 }
 
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void  ternary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{

 if (CONFIG_T::io_type == io_parallel){
     #pragma HLS PIPELINE
 }
  
 data_T datareg;   
 res_T cache; 
 for (int ii=0; ii<CONFIG_T::n_in; ii++) {

  if (CONFIG_T::io_type == io_serial){
      #pragma HLS PIPELINE
  }
  datareg = 2*data[ii];
  if( datareg > 1 ) cache = 1;
  else if( datareg > -1 && datareg <= 1) cache=0;
  else cache = -1;
  
  res[ii] = (res_T) cache;
 
 }
 
}

}

#endif
