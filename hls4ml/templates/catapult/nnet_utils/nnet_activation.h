
// Change History:
//   2022-06-30  dgburnette - Cleaned up code to separate AC Math from LUT code.
//                            Added LUT dump to text file.
//                            Activation functions not implemented in AC Math will assert.
//   2022-06-28  dgburnette - Replaced AP Types with AC Datatypes.
//                            Commented out all Vivado pragmas.
//                            Added Catapult hierarchy pragmas.
//                            Started replacement of activation functions with
//                            AC Math piecewise-linear versions.

#ifndef NNET_ACTIVATION_H_
#define NNET_ACTIVATION_H_

// Define this macro to switch the implementations of certain activiation functions
// from the original HLS4ML look-up table approach to using the piecewise-linear approximation
// functions in AC Math.
#define USE_AC_MATH 1

#if !defined(USE_AC_MATH) && !defined(__SYNTHESIS__)
// Define a macro that causes the look-up table generation code to dump out text files
// of the array contents.
// #define BUILD_TABLE_FILE 1
#endif

#include "ac_fixed.h"
#include "ac_std_float.h"
#include "nnet_common.h"
#include <ac_math/ac_elu_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_relu.h>
#include <ac_math/ac_selu_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math/ac_softmax_pwl.h>
#include <ac_math/ac_softplus_pwl.h>
#include <ac_math/ac_softsign_pwl.h>
#include <ac_math/ac_tanh_pwl.h>
#include <cmath>

namespace nnet {

struct activ_config {
    // IO size
    static const unsigned n_in = 10;

    // Internal info
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef ac_fixed<18, 8, true> table_t;
};

// *************************************************
//       LINEAR Activation -- See Issue 53
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void linear(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        res[ii] = data[ii];
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void relu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
#ifndef USE_AC_MATH
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = 0;
#else
        ac_math::ac_relu(datareg, res[ii]);
#endif
    }
}

template <class data_T, class res_T, int MAX_INT, typename CONFIG_T>
void relu_max(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE
    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg < 0)
            res[ii] = 0;
        else if (datareg > MAX_INT)
            res[ii] = MAX_INT;
        else
            res[ii] = datareg;
    }
}

template <class data_T, class res_T, typename CONFIG_T> void relu6(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    relu_max<data_T, res_T, 6, CONFIG_T>(data, res);
}

template <class data_T, class res_T, typename CONFIG_T> void relu1(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    relu_max<data_T, res_T, 1, CONFIG_T>(data, res);
}

// *************************************************
//       Sigmoid Activation
// *************************************************

template <
    /*unsigned K,*/ int W1,
    int I1,
    bool S1,
    ac_q_mode Q1,
    ac_o_mode O1,
    int W2,
    int I2,
    bool S2,
    ac_q_mode Q2,
    ac_o_mode O2>
void ac_sigmoid_pwl_wrapper(
    const ac_fixed<W1, I1, S1, Q1, O1>(&input) /*[K]*/, ac_fixed<W2, I2, S2, Q2, O2>(&output) /*[K]*/
) {
    ac_fixed<W2, I2, false, Q2, O2> tmp; //[K];
    ac_math::ac_sigmoid_pwl<AC_TRN, W1, I1, true, Q1, O1, W2, I2, Q2, O2>(input, tmp);
    output = tmp;
}

inline float sigmoid_fcn_float(float input) { return 1.0 / (1 + std::exp(-input)); }

template <typename CONFIG_T, int N_TABLE> void init_sigmoid_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
#ifdef BUILD_TABLE_FILE
    char filename[1024];
    sprintf(filename, "sigmoid_table%d.tab", N_TABLE);
    FILE *f = fopen(filename, "w");
    fprintf(f, "// init_sigmoid_table()\n");
#endif
    // Default logistic sigmoid function:
    //   result = 1/(1+e^(-x))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = sigmoid_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
#ifdef BUILD_TABLE_FILE
        fprintf(f, "%32.31f", sigmoid_fcn_float(in_val));
        if (ii < N_TABLE - 1)
            fprintf(f, ",");
        fprintf(f, "   // sigmoid(%32.31f)", in_val);
        fprintf(f, "\n");
#endif
    }
#ifdef BUILD_TABLE_FILE
    fclose(f);
#endif
}

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T>
void sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
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

    //#pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii].to_double() * (int)CONFIG_T::table_size / 16;
        index = data_round + 8 * (int)CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = (int)CONFIG_T::table_size - 1;
        res[ii] = (res_T)sigmoid_table[index];
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T>
void sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        // res[ii] = ac_math::ac_sigmoid_pwl(data[ii]);
        ac_sigmoid_pwl_wrapper(data[ii], res[ii]);
    }
}

#endif

// *************************************************
//       Softmax Activation
// *************************************************

enum class softmax_implementation { latency = 0, legacy = 1, stable = 2 };

inline float exp_fcn_float(float input) { return std::exp(input); }

template <class data_T, typename CONFIG_T> inline float softmax_real_val_from_idx(unsigned i) {
    // Treat the index as the top N bits
    static constexpr int N = ceillog2(CONFIG_T::table_size); // number of address bits for table
    data_T x(0);
    // CATAPULT_PORT
    // x(x.width-1, x.width-N) = i;
    ac_int<N, false> tmp = i;
    x.template set_slc(x.width - N, tmp);
    return (float)x.to_double();
}

template <class data_T, typename CONFIG_T> inline unsigned softmax_idx_from_real_val(data_T x) {
    // Slice the top N bits to get an index into the table
    static constexpr int N = ceillog2(CONFIG_T::table_size); // number of address bits for table
    // CATAPULT_PORT
    // ac_int<N,false> y = x(x.width-1, x.width-N); // slice the top N bits of input
    // return (unsigned) y(N-1, 0);
    ac_int<N, false> y = x.template slc<N>(x.width - N); // slice the top N bits of input
    return (unsigned)y.template slc<N>(0);
}

template <class data_T, typename CONFIG_T>
void init_exp_table(typename CONFIG_T::exp_table_t table_out[CONFIG_T::table_size]) {
#ifdef BUILD_TABLE_FILE
    char filename[1024];
    sprintf(filename, "exp_table%d.tab", CONFIG_T::table_size);
    FILE *f = fopen(filename, "w");
    fprintf(f, "// init_exp_table()\n");
#endif
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < CONFIG_T::table_size; i++) {
        // Slicing bits for address is going to round towards 0, so take the central value
        float x = softmax_real_val_from_idx<data_T, CONFIG_T>(i);
        typename CONFIG_T::exp_table_t exp_x = exp_fcn_float(x);
        table_out[i] = exp_x;
#ifdef BUILD_TABLE_FILE
        fprintf(f, "%32.31f", exp_fcn_float(x));
        if (i < CONFIG_T::table_size - 1)
            fprintf(f, ",");
        fprintf(f, "   // exp(%32.31f)", x);
        fprintf(f, "\n");
#endif
    }
#ifdef BUILD_TABLE_FILE
    fclose(f);
#endif
}

template <class data_T, typename CONFIG_T>
void init_invert_table(typename CONFIG_T::inv_table_t table_out[CONFIG_T::table_size]) {
#ifdef BUILD_TABLE_FILE
    char filename[1024];
    sprintf(filename, "invert_table%d.tab", CONFIG_T::table_size);
    FILE *f = fopen(filename, "w");
    fprintf(f, "// init_invert_table()\n");
#endif
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < CONFIG_T::table_size; i++) {
        float x = softmax_real_val_from_idx<data_T, CONFIG_T>(i);
#ifdef __SYNTHESIS__
        // hack for now to get through the flow
        typename CONFIG_T::inv_table_t inv_x = 1 + x;
#else
        typename CONFIG_T::inv_table_t inv_x = 1 / x;
#endif
        table_out[i] = inv_x;
#ifdef BUILD_TABLE_FILE
        if (x > 0.0)
            fprintf(f, "%32.31f", (1.0 / x));
        else
            fprintf(f, "%32.31f", 0.0);
        if (i < CONFIG_T::table_size - 1)
            fprintf(f, ",");
        fprintf(f, "   // 1/(%32.31f)", x);
        fprintf(f, "\n");
#endif
    }
#ifdef BUILD_TABLE_FILE
    fclose(f);
#endif
}

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS pipeline
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
    //#pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        //#pragma HLS unroll
        unsigned x = softmax_idx_from_real_val<data_T, CONFIG_T>(data[i]);
        exp_res[i] = exp_table[x];
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    exp_sum =
        reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        //#pragma HLS unroll
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS pipeline
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
    ac_fixed<data_T::width, data_T::i_width, true, AC_RND, AC_SAT> d_xi_xmax[CONFIG_T::n_in];
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        //#pragma HLS unroll
        d_xi_xmax[i] = data[i] - x_max;
    }

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    //#pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        //#pragma HLS unroll
        unsigned x = softmax_idx_from_real_val<data_T, CONFIG_T>(d_xi_xmax[i]);
        exp_res[i] = exp_table[x];
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    exp_sum =
        reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        //#pragma HLS unroll
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

#endif

template <typename CONFIG_T, int N_TABLE> void init_exp_table_legacy(typename CONFIG_T::table_t table_out[N_TABLE]) {
#ifdef BUILD_TABLE_FILE
    char filename[1024];
    sprintf(filename, "exp_table_legacy%d.tab", N_TABLE);
    FILE *f = fopen(filename, "w");
    fprintf(f, "// init_exp_table_legacy()\n");
#endif
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = exp_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
#ifdef BUILD_TABLE_FILE
        fprintf(f, "%32.31f", exp_fcn_float(in_val));
        if (ii < N_TABLE - 1)
            fprintf(f, ",");
        fprintf(f, "   // exp(%32.31f)", in_val);
        fprintf(f, "\n");
#endif
    }
#ifdef BUILD_TABLE_FILE
    fclose(f);
#endif
}

template <typename CONFIG_T, int N_TABLE> void init_invert_table_legacy(typename CONFIG_T::table_t table_out[N_TABLE]) {
#ifdef BUILD_TABLE_FILE
    char filename[1024];
    sprintf(filename, "invert_table_legacy%d.tab", N_TABLE);
    FILE *f = fopen(filename, "w");
    fprintf(f, "// init_invert_table_legacy()\n");
#endif
    // Inversion function:
    //   result = 1/x
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +64)
        float in_val = 64.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        if (in_val > 0.0)
            table_out[ii] = 1.0 / in_val;
        else
            table_out[ii] = 0.0;
#ifdef BUILD_TABLE_FILE
        if (in_val > 0.0)
            fprintf(f, "%32.31f", (1.0 / in_val));
        else
            fprintf(f, "%32.31f", 0.0);
        if (ii < N_TABLE - 1)
            fprintf(f, ",");
        fprintf(f, "   // 1/%32.31f", in_val);
        fprintf(f, "\n");
#endif
    }
#ifdef BUILD_TABLE_FILE
    fclose(f);
#endif
}

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
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

    //#pragma HLS PIPELINE

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[CONFIG_T::n_in]; // different, independent, fixed point precision
    typename CONFIG_T::table_t exp_diff_res;            // different, independent, fixed point precision
    data_T data_cache[CONFIG_T::n_in];
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_cache[ii] = data[ii];
        exp_res[ii] = 0;
    }

    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        for (int jj = 0; jj < CONFIG_T::n_in; jj++) {
            if (ii == jj)
                exp_diff_res = 1;
            else {
                // CATAPULT_PORT
                // data_round = (data_cache[jj]-data_cache[ii])*CONFIG_T::table_size/16;
                auto tmp_data_round = (data_cache[jj] - data_cache[ii]) * CONFIG_T::table_size / 16;
                data_round = tmp_data_round.to_int();
                index = data_round + 8 * CONFIG_T::table_size / 16;
                if (index < 0)
                    index = 0;
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                exp_diff_res = exp_table[index];
            }
            exp_res[ii] += exp_diff_res;
        }
    }

    // Second loop to invert
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        // CATAPULT_PORT
        // int exp_res_index = exp_res[ii]*CONFIG_T::table_size/64;
        auto tmp_exp_res_index = exp_res[ii] * CONFIG_T::table_size / 64;
        int exp_res_index = tmp_exp_res_index.to_int();
        if (exp_res_index < 0)
            exp_res_index = 0;
        if (exp_res_index > CONFIG_T::table_size - 1)
            exp_res_index = CONFIG_T::table_size - 1;
        // typename CONFIG_T::table_t exp_res_invert = invert_table[exp_res_index];
        res[ii] = (res_T)invert_table[exp_res_index];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE
    switch (CONFIG_T::implementation) {
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

#else
// This is a workaround to help the template deduction to work correctly and fix the inconsistency that HLS4ML expects
// softmax output to be signed but AC Math softmax knows it is always unsigned
template <
    unsigned K,
    int W1,
    int I1,
    bool S1,
    ac_q_mode Q1,
    ac_o_mode O1,
    int W2,
    int I2,
    bool S2,
    ac_q_mode Q2,
    ac_o_mode O2>
void ac_softmax_pwl_wrapper(const ac_fixed<W1, I1, S1, Q1, O1> (&input)[K], ac_fixed<W2, I2, S2, Q2, O2> (&output)[K]) {
    ac_fixed<W2, I2, false, Q2, O2> tmp[K];
    ac_math::ac_softmax_pwl<
        AC_TRN,
        false,
        0,
        0,
        AC_TRN,
        AC_WRAP,
        false,
        0,
        0,
        AC_TRN,
        AC_WRAP,
        K,
        W1,
        I1,
        S1,
        Q1,
        O1,
        W2,
        I2,
        Q2,
        O2>(input, tmp);
    for (unsigned int x = 0; x < K; x++)
        output[x] = tmp[x];
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    data_T data_copy[CONFIG_T::n_in];
    res_T res_copy[CONFIG_T::n_in];
// workaround for the array passing - alternative is to change the signature of all of the functions to reference-of-array
COPY_IN_ARRAY:
    for (unsigned i = 0; i < CONFIG_T::n_in; i++)
        data_copy[i] = data[i];
    ac_softmax_pwl_wrapper(data_copy, res_copy);
COPY_OUT_ARRAY:
    for (unsigned i = 0; i < CONFIG_T::n_in; i++)
        res[i] = res_copy[i];
}

#endif

// *************************************************
//       TanH Activation
// *************************************************
template <typename CONFIG_T, int N_TABLE> void init_tanh_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
#ifdef BUILD_TABLE_FILE
    char filename[1024];
    sprintf(filename, "tanh_table%d.tab", N_TABLE);
    FILE *f = fopen(filename, "w");
    fprintf(f, "// init_tanh_table()\n");
#endif
    // Implement tanh lookup
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
        float in_val = 2 * 4.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = tanh(in_val);
        // std::cout << "Tanh:  Lookup table Index: " <<  ii<< " In Value: " << in_val << " Result: " << real_val <<
        // std::endl;
        table_out[ii] = real_val;
#ifdef BUILD_TABLE_FILE
        fprintf(f, "%32.31f", tanh(in_val));
        if (ii < N_TABLE - 1)
            fprintf(f, ",");
        fprintf(f, "   // tanh(%32.31f)", in_val);
        fprintf(f, "\n");
#endif
    }
#ifdef BUILD_TABLE_FILE
    fclose(f);
#endif
}

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
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

    //#pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii].to_double() * (int)CONFIG_T::table_size / 8;
        index = data_round + 4 * (int)CONFIG_T::table_size / 8;
        // std::cout << "Input: "  << data[ii] << " Round: " << data_round << " Index: " << index << std::endl;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = (int)CONFIG_T::table_size - 1;
        res[ii] = (res_T)tanh_table[index];
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        // res[ii] = ac_math::ac_tanh_pwl(data[ii]);
        ac_math::ac_tanh_pwl(data[ii], res[ii]);
    }
}

#endif

// *************************************************
//       Hard sigmoid Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void hard_sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    data_T slope = (data_T)0.2;
    data_T shift = (data_T)0.5;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = slope * data[ii] + shift;
        if (datareg > 1)
            datareg = 1;
        else if (datareg < 0)
            datareg = 0;
        res[ii] = datareg;
    }
}

// *************************************************
//       Hard TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void hard_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    data_T slope = (data_T)0.2;
    data_T shift = (data_T)0.5;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto sigmoid = CONFIG_T::slope * data[ii] + CONFIG_T::shift;
        if (sigmoid > 1)
            datareg = 1;
        else if (sigmoid < 0)
            datareg = 0;
        res[ii] = 2 * sigmoid - 1;
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void leaky_relu(data_T data[CONFIG_T::n_in], param_T alpha, res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = alpha * datareg;
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void thresholded_relu(data_T data[CONFIG_T::n_in], param_T theta, res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > theta)
            res[ii] = datareg;
        else
            res[ii] = 0;
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
inline float softplus_fcn_float(float input) { return std::log(std::exp(input) + 1.); }

template <typename CONFIG_T, int N_TABLE> void init_softplus_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
#ifdef BUILD_TABLE_FILE
    char filename[1024];
    sprintf(filename, "softplus_table%d.tab", N_TABLE);
    FILE *f = fopen(filename, "w");
    fprintf(f, "// init_softplus_table()\n");
#endif
    // Default softplus function:
    //   result = log(exp(x) + 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = softplus_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
#ifdef BUILD_TABLE_FILE
        fprintf(f, "%32.31f", softplus_fcn_float(in_val));
        if (ii < N_TABLE - 1)
            fprintf(f, ",");
        fprintf(f, "   // softplus(%32.31f)", in_val);
        fprintf(f, "\n");
#endif
    }
#ifdef BUILD_TABLE_FILE
    fclose(f);
#endif
}

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T>
void softplus(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
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

    //#pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii].to_double() * (int)CONFIG_T::table_size / 16;
        index = data_round + 8 * (int)CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = (int)CONFIG_T::table_size - 1;
        res[ii] = (res_T)softplus_table[index];
    }
}

#else
template <
    ac_q_mode pwl_Q = AC_TRN,
    int W,
    int I,
    bool S,
    ac_q_mode Q,
    ac_o_mode O,
    int outW,
    int outI,
    bool outS,
    ac_q_mode outQ,
    ac_o_mode outO>
void ac_softplus_pwl_wrapper(const ac_fixed<W, I, S, Q, O>(&input), ac_fixed<outW, outI, outS, outQ, outO>(&output)) {
    ac_fixed<outW, outI, false, outQ, outO> tmp;
    ac_math::ac_softplus_pwl<AC_TRN, W, I, S, Q, O, outW, outI, outQ, outO>(input, tmp);
    output = tmp;
}

template <class data_T, class res_T, typename CONFIG_T>
void softplus(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        ac_softplus_pwl_wrapper(data[ii], res[ii]);
    }
}

#endif

// *************************************************
//       Softsign Activation
// *************************************************
inline float softsign_fcn_float(float input) { return input / (std::abs(input) + 1.); }

template <typename CONFIG_T, int N_TABLE> void init_softsign_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
#ifdef BUILD_TABLE_FILE
    char filename[1024];
    sprintf(filename, "softsign_table%d.tab", N_TABLE);
    FILE *f = fopen(filename, "w");
    fprintf(f, "// init_softsign_table()\n");
#endif
    // Default softsign function:
    //   result = x / (abs(x) + 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = softsign_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
#ifdef BUILD_TABLE_FILE
        fprintf(f, "%32.31f", softsign_fcn_float(in_val));
        if (ii < N_TABLE - 1)
            fprintf(f, ",");
        fprintf(f, "   // softsign(%32.31f)", in_val);
        fprintf(f, "\n");
#endif
    }
#ifdef BUILD_TABLE_FILE
    fclose(f);
#endif
}

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T>
void softsign(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
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

    //#pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii].to_double() * (int)CONFIG_T::table_size / 16;
        index = data_round + 8 * (int)CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = (int)CONFIG_T::table_size - 1;
        res[ii] = (res_T)softsign_table[index];
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T>
void softsign(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        // res[ii] = ac_math::ac_softsign_pwl(data[ii]);
        ac_math::ac_softsign_pwl(data[ii], res[ii]);
    }
}

#endif

// *************************************************
//       ELU Activation
// *************************************************
inline float elu_fcn_float(float input) { return std::exp(input) - 1.; }

template <typename CONFIG_T, int N_TABLE> void init_elu_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
#ifdef BUILD_TABLE_FILE
    char filename[1024];
    sprintf(filename, "elu_table%d.tab", N_TABLE);
    FILE *f = fopen(filename, "w");
    fprintf(f, "// init_elu_table()\n");
#endif
    // Default ELU function:
    //   result = alpha * (e^(x) - 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to 0)
        float in_val = -8.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = elu_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
#ifdef BUILD_TABLE_FILE
        fprintf(f, "%32.31f", elu_fcn_float(in_val));
        if (ii < N_TABLE - 1)
            fprintf(f, ",");
        fprintf(f, "   // elu(%32.31f)", in_val);
        fprintf(f, "\n");
#endif
    }
#ifdef BUILD_TABLE_FILE
    fclose(f);
#endif
}

#ifndef USE_AC_MATH

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void elu(data_T data[CONFIG_T::n_in], const param_T alpha, res_T res[CONFIG_T::n_in]) {
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

    //#pragma HLS PIPELINE

    data_T datareg;
    // Index into the lookup table based on data
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = datareg;
        } else {
            index = datareg.to_double() * (int)CONFIG_T::table_size / -8;
            if (index > CONFIG_T::table_size - 1)
                index = (int)CONFIG_T::table_size - 1;
            res[ii] = alpha * elu_table[index];
        }
    }
}

#else

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void elu(data_T data[CONFIG_T::n_in], const param_T alpha, res_T res[CONFIG_T::n_in]) {
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        ac_math::ac_elu_pwl(data[ii], res[ii], alpha);
    }
}

#endif

// *************************************************
//       SELU Activation
// *************************************************
inline float selu_fcn_float(float input) {
    return 1.0507009873554804934193349852946 * (1.6732632423543772848170429916717 * (std::exp(input) - 1.));
}

template <typename CONFIG_T, int N_TABLE> void init_selu_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
#ifdef BUILD_TABLE_FILE
    char filename[1024];
    sprintf(filename, "selu_table%d.tab", N_TABLE);
    FILE *f = fopen(filename, "w");
    fprintf(f, "// init_selu_table()\n");
#endif
    // Default SELU function:
    //   result = 1.05 * (1.673 * (e^(x) - 1))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to 0)
        float in_val = -8.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = selu_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
#ifdef BUILD_TABLE_FILE
        fprintf(f, "%32.31f", selu_fcn_float(in_val));
        if (ii < N_TABLE - 1)
            fprintf(f, ",");
        fprintf(f, "   // selu(%32.31f)", in_val);
        fprintf(f, "\n");
#endif
    }
#ifdef BUILD_TABLE_FILE
    fclose(f);
#endif
}

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void selu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
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

    //#pragma HLS PIPELINE

    data_T datareg;
    // Index into the lookup table based on data
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = res_T(1.0507009873554804934193349852946) * datareg;
        } else {
            index = datareg.to_double() * (int)CONFIG_T::table_size / -8;
            if (index > CONFIG_T::table_size - 1)
                index = (int)CONFIG_T::table_size - 1;
            res[ii] = selu_table[index];
        }
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void selu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        res[ii] = ac_math::ac_selu_pwl<res_T>(data[ii]);
    }
}

#endif

// *************************************************
//       PReLU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void prelu(data_T data[CONFIG_T::n_in], param_T alpha[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = alpha[ii] * datareg;
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void binary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    //#pragma HLS PIPELINE

    data_T datareg;
    res_T cache;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > 0)
            cache = 1;
        else
            cache = -1;

        res[ii] = (res_T)cache;
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void ternary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {

    //#pragma HLS PIPELINE

    data_T datareg;
    res_T cache;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = 2 * data[ii];
        if (datareg > 1)
            cache = 1;
        else if (datareg > -1 && datareg <= 1)
            cache = 0;
        else
            cache = -1;

        res[ii] = (res_T)cache;
    }
}

} // namespace nnet

#endif
