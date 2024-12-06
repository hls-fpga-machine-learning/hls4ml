#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_stream.h"
#include "nnet_types.h"
#include <cmath>

namespace nnet {

// *************************************************
//       LINEAR Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void linear(hls::stream<data_T> &data, hls::stream<res_T> &res) {
LinearActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    LinearPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data[j];
        }

        res.write(out_data);
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void relu(hls::stream<data_T> &data, hls::stream<res_T> &res) {
ReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    ReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void sigmoid(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

SigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    SigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            int data_round = in_data[j] * CONFIG_T::table_size / 16;
            int index = data_round + 8 * CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            out_data[j] = sigmoid_table[index];
        }

        res.write(out_data);
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<typename data_T::value_type, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::inv_inp_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned ii = data_T::size / multiplier_limit;

    // Calculate all the e^x's
    typename CONFIG_T::accum_t exp_res[data_T::size];
    #pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::inv_inp_t exp_sum(0);
SoftmaxExpLoop:
    for (unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        #pragma HLS PIPELINE II=ii

        data_T in_pack = data.read();
    SoftmaxExpPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            unsigned x = softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T::exp_table_size>(in_pack[j]);
            exp_res[j] = exp_table[x];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::accum_t> op_add;
        exp_sum = reduce<typename CONFIG_T::accum_t, data_T::size, Op_add<typename CONFIG_T::accum_t>>(exp_res, op_add);

        typename CONFIG_T::inv_table_t inv_exp_sum =
            invert_table[softmax_idx_from_real_val<typename CONFIG_T::inv_inp_t, CONFIG_T::inv_table_size>(exp_sum)];

        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

    SoftmaxInvPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            #pragma HLS ALLOCATION operation instances=mul limit=multiplier_limit
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<typename CONFIG_T::inp_norm_t, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::inv_inp_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned ii = data_T::size / multiplier_limit;

    typename data_T::value_type data_array[data_T::size];
#pragma HLS ARRAY_PARTITION variable=data_array complete
SoftmaxArrayLoop:
    for (unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        #pragma HLS PIPELINE II=ii

        data_T in_pack = data.read();
    SoftmaxArrayPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            data_array[j] = in_pack[j];
        }

        // Find the max and compute all delta(x_i, x_max)
        Op_max<typename data_T::value_type> op_max;
        typename data_T::value_type x_max =
            reduce<typename data_T::value_type, data_T::size, Op_max<typename data_T::value_type>>(data_array, op_max);

        // For the diffs, use the same type as the input but force rounding and saturation
        typename CONFIG_T::inp_norm_t d_xi_xmax[data_T::size];
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            d_xi_xmax[j] = data_array[j] - x_max;
        }

        // Calculate all the e^x's
        typename CONFIG_T::accum_t exp_res[data_T::size];
        #pragma HLS ARRAY_PARTITION variable=exp_res complete
        typename CONFIG_T::inv_inp_t exp_sum(0);
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            unsigned x = softmax_idx_from_real_val<typename CONFIG_T::inp_norm_t, CONFIG_T::exp_table_size>(d_xi_xmax[j]);
            exp_res[j] = exp_table[x];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::accum_t> op_add;
        exp_sum = reduce<typename CONFIG_T::accum_t, data_T::size, Op_add<typename CONFIG_T::accum_t>>(exp_res, op_add);

        typename CONFIG_T::inv_table_t inv_exp_sum =
            invert_table[softmax_idx_from_real_val<typename CONFIG_T::accum_t, CONFIG_T::inv_table_size>(exp_sum)];

        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

    SoftmaxInvPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            #pragma HLS ALLOCATION operation instances=mul limit=multiplier_limit
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[data_T::size];
    typename CONFIG_T::table_t exp_diff_res;
    typename data_T::value_type data_cache[data_T::size];

SoftmaxInitLoop:
    for (unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        #pragma HLS PIPELINE
        data_T in_pack = data.read();
    SoftmaxInitPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            data_cache[j] = in_pack[j];
            exp_res[j] = 0;
        }

    SoftmaxExpLoop:
        for (int i = 0; i < data_T::size; i++) {
        #pragma HLS UNROLL
        SoftmaxExpInner:
            for (int j = 0; j < data_T::size; j++) {
                #pragma HLS UNROLL

                if (i == j) {
                    exp_diff_res = 1;
                } else {
                    int data_round = (data_cache[j] - data_cache[i]) * CONFIG_T::table_size / 16;
                    int index = data_round + 8 * CONFIG_T::table_size / 16;
                    if (index < 0)
                        index = 0;
                    if (index > CONFIG_T::table_size - 1)
                        index = CONFIG_T::table_size - 1;
                    exp_diff_res = exp_table[index];
                }

                exp_res[i] += exp_diff_res;
            }
        }

        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

    SoftmaxInvPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL

            int exp_res_index = exp_res[j] * CONFIG_T::table_size / 64;
            if (exp_res_index < 0)
                exp_res_index = 0;
            if (exp_res_index > CONFIG_T::table_size - 1)
                exp_res_index = CONFIG_T::table_size - 1;

            out_pack[j] = (typename res_T::value_type)invert_table[exp_res_index];
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_argmax(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE
        data_T in_data = data.read();
        res_T out_data;

        for (int i = 0; i < res_T::size; i++) {
            #pragma HLS UNROLL
            out_data[i] = (typename res_T::value_type)0;
        }

        typename data_T::value_type maximum = in_data[0];
        int idx = 0;

        for (int i = 1; i < res_T::size; i++) {
            #pragma HLS PIPELINE
            if (in_data[i] > maximum) {
                maximum = in_data[i];
                idx = i;
            }
        }

        out_data[idx] = (typename res_T::value_type)1;
        res.write(out_data);
    }
}

template <class data_T, class res_T, typename CONFIG_T> void softmax(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    assert(CONFIG_T::axis == -1);

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
    case softmax_implementation::argmax:
        softmax_argmax<data_T, res_T, CONFIG_T>(data, res);
        break;
    }
}

// *************************************************
//       TanH Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void tanh(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

TanHActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    TanHPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            int data_round = in_data[j] * CONFIG_T::table_size / 8;
            int index = data_round + 4 * CONFIG_T::table_size / 8;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            out_data[j] = tanh_table[index];
        }

        res.write(out_data);
    }
}

// *************************************************
//       UnaryLUT Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T>
void unary_lut(hls::stream<data_T> &data, hls::stream<res_T> &res, typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    #pragma HLS function_instantiate variable=table
    #pragma HLS ARRAY_PARTITION variable=table complete

UnaryLUTActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    UnaryLUTPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            unsigned index = get_index_unary_lut<CONFIG_T::table_size>(in_data[j].V);
            out_data[j] = table[index];
        }

        res.write(out_data);
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T>
void hard_sigmoid(hls::stream<data_T> &data, hls::stream<res_T> &res) {

HardSigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    HardSigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            auto datareg = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (datareg > 1)
                datareg = 1;
            else if (datareg < 0)
                datareg = 0;
            out_data[j] = datareg;
        }

        res.write(out_data);
    }
}

template <class data_T, class res_T, typename CONFIG_T> void hard_tanh(hls::stream<data_T> &data, hls::stream<res_T> &res) {

HardSigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    HardSigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            auto sigmoid = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (sigmoid > 1)
                sigmoid = 1;
            else if (sigmoid < 0)
                sigmoid = 0;
            out_data[j] = 2 * sigmoid - 1;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void leaky_relu(hls::stream<data_T> &data, param_T alpha, hls::stream<res_T> &res) {
LeakyReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    LeakyReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha * in_data[j];
        }
        res.write(out_data);
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void thresholded_relu(hls::stream<data_T> &data, param_T theta, hls::stream<res_T> &res) {
ThresholdedReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    ThresholdedReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            if (in_data[j] > theta)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Softplus Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void softplus(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

SoftplusActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    SoftplusPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            int data_round = in_data[j] * CONFIG_T::table_size / 16;
            int index = data_round + 8 * CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            out_data[j] = softplus_table[index];
        }
        res.write(out_data);
    }
}

// *************************************************
//       Softsign Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void softsign(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

SoftsignActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    SoftsignPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            int data_round = in_data[j] * CONFIG_T::table_size / 16;
            int index = data_round + 8 * CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            out_data[j] = softsign_table[index];
        }
        res.write(out_data);
    }
}

// *************************************************
//       ELU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void elu(hls::stream<data_T> &data, param_T alpha, hls::stream<res_T> &res) {
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

EluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    EluPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL

            typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = datareg;
            } else {
                int index = datareg * CONFIG_T::table_size / -8;
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data[j] = alpha * elu_table[index];
            }
        }
        res.write(out_data);
    }
}

template <class data_T, class res_T, typename CONFIG_T> void elu(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    elu<data_T, ap_uint<1>, res_T, CONFIG_T>(data, 1.0, res);
}

// *************************************************
//       SELU Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void selu(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

SeluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    SeluPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL

            typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = (typename data_T::value_type)1.0507009873554804934193349852946 * datareg;
            } else {
                int index = datareg * CONFIG_T::table_size / -8;
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data[j] = selu_table[index];
            }
        }
        res.write(out_data);
    }
}

// *************************************************
//       PReLU Activation
// *************************************************

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void prelu(hls::stream<data_T> &data, const param_T alpha[CONFIG_T::n_in], hls::stream<res_T> &res) {
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha[i * res_T::size + j] * in_data[j];
        }
        res.write(out_data);
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void binary_tanh(hls::stream<data_T> &data, hls::stream<res_T> &res) {
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            if (in_data[j] > 0)
                out_data[j] = (typename res_T::value_type)1;
            else
                out_data[j] = (typename res_T::value_type) - 1;
        }
        res.write(out_data);
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void ternary_tanh(hls::stream<data_T> &data, hls::stream<res_T> &res) {
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            if (in_data[j] > 1)
                out_data[j] = (typename res_T::value_type)1;
            else if (in_data[j] <= -1)
                out_data[j] = (typename res_T::value_type) - 1;
            else
                out_data[j] = (typename res_T::value_type)0;
        }
        res.write(out_data);
    }
}

} // namespace nnet

#endif
