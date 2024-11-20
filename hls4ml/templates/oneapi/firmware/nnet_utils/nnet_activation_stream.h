#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "nnet_common.h"
#include "nnet_types.h"

namespace nnet {

// *************************************************
//       Linear Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void linear_stream() {
LinearActLoop:
    [[intel::initiation_interval(1
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    LinearPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            out_data[j] = in_data[j];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       ReLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void relu_stream() {
ReLUActLoop:
    [[intel::initiation_interval(1
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    ReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void leaky_relu_stream(typename CONFIG_T::param_t alpha) {
    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{} / multiplier_limit;

LeakyReLUActLoop:
    [[intel::initiation_interval(pipeline
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    LeakyReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha * in_data[j];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T>
void thresholded_relu_stream(typename CONFIG_T::param_t theta) {
ThresholdedReLUActLoop:
    [[intel::initiation_interval(1
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    ThresholdedReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            if (in_data[j] > theta)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       ELU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void elu_stream(typename CONFIG_T::param_t alpha) {
#include "activation_tables/elu_table.tb"

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{} / multiplier_limit;

EluActLoop:
    [[intel::initiation_interval(pipeline
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    EluPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            [[intel::fpga_register]] typename ExtractPipeType<data_pipe>::value_type::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = datareg;
            } else {
                int index = (datareg * CONFIG_T::table_size / -8).to_int();
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data[j] = alpha * elu_table[index];
            }
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       SeLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void selu_stream() {
#include "activation_tables/selu_table.tb"

SeluActLoop:
    [[intel::initiation_interval(1
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    SeluPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            [[intel::fpga_register]] typename ExtractPipeType<data_pipe>::value_type::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] =
                    typename ExtractPipeType<data_pipe>::value_type::value_type(1.0507009873554804934193349852946) * datareg;
            } else {
                int index = (datareg * CONFIG_T::table_size / -8).to_int();
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data[j] = selu_table[index];
            }
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void prelu_stream(typename CONFIG_T::param_t alpha) {
    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{} / multiplier_limit;

PReLUActLoop:
    [[intel::initiation_interval(pipeline
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    PReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha[i * std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{} + j] * in_data[j];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void softplus_stream() {
#include "activation_tables/softplus_table.tb"

SoftplusActLoop:
    [[intel::initiation_interval(1
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    SoftplusPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            [[intel::fpga_register]] int data_round = (in_data[j] * CONFIG_T::table_size / 16).to_int();
            [[intel::fpga_register]] int index = data_round + 8 * CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            out_data[j] = softplus_table[index];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void softsign_stream() {
#include "activation_tables/softsign_table.tb"

    static const int MAX_VALUE = 8;

SoftsignActLoop:
    [[intel::initiation_interval(1
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    SoftsignPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            [[intel::fpga_register]] typename ExtractPipeType<data_pipe>::value_type::value_type absValue;
            ;
            if (in_data[j] < 0) {
                absValue = -in_data[j];
            } else {
                absValue = in_data[j];
            }
            ac_int<16> index = (absValue * CONFIG_T::table_size / MAX_VALUE).to_int();
            if (absValue > MAX_VALUE)
                index = CONFIG_T::table_size - 1;
            if (in_data[j] < 0) {
                out_data[j] =
                    static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(-softsign_table[index]);
            } else {
                out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(softsign_table[index]);
            }
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_stable_stream() {
#include "activation_tables/exp_table.tb"
#include "activation_tables/invert_table.tb"

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{} / multiplier_limit;

    [[intel::fpga_register]] typename ExtractPipeType<data_pipe>::value_type::value_type
        data_array[std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}];

SoftmaxArrayLoop:
    [[intel::initiation_interval(pipeline
    )]] for (unsigned i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}; i++) {
        auto in_pack = data_pipe::read();

    SoftmaxArrayPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}; j++) {
            data_array[j] = in_pack[j];
        }

        // Find the max and compute all delta(x_i, x_max)
        Op_max<typename ExtractPipeType<data_pipe>::value_type::value_type> op_max;
        [[intel::fpga_register]] typename ExtractPipeType<data_pipe>::value_type::value_type x_max = reduce<
            typename ExtractPipeType<data_pipe>::value_type::value_type,
            std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{},
            Op_max<typename ExtractPipeType<data_pipe>::value_type::value_type>>(data_array, op_max);

        // For the diffs, use the same type as the input but force rounding and saturation
        [[intel::fpga_register]] ac_fixed<
            ExtractPipeType<data_pipe>::value_type::value_type::width,
            ExtractPipeType<data_pipe>::value_type::value_type::i_width,
            true,
            AC_RND,
            AC_SAT>
            d_xi_xmax[std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}];
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}; j++) {
            d_xi_xmax[j] = data_array[j] - x_max;
        }

        // Calculate all the e^x's
        [[intel::fpga_register]]
        typename CONFIG_T::exp_table_t exp_res[std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}];
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}; j++) {
            exp_res[j] = exp_table[softmax_stable_idx_from_real_val<
                typename ExtractPipeType<data_pipe>::value_type::value_type,
                CONFIG_T>(d_xi_xmax[j])];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_sum = reduce<
            typename CONFIG_T::exp_table_t,
            std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{},
            Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        [[intel::fpga_register]] typename CONFIG_T::inv_table_t inv_exp_sum =
            invert_table[softmax_stable_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
        typename ExtractPipeType<res_pipe>::value_type out_pack;

    SoftmaxInvPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {

            // TODO - Find Quartus-equivalent pragma
            // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

            out_pack[j] = exp_res[j] * inv_exp_sum;
        }

        res_pipe::write(out_pack);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_latency_stream() {
#include "activation_tables/exp_table_latency.tb"
#include "activation_tables/invert_table_latency.tb"

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{} / multiplier_limit;

    // Calculate all the e^x's
    [[intel::fpga_register]]
    typename CONFIG_T::exp_table_t exp_res[std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}];

SoftmaxExpLoop:
    [[intel::initiation_interval(pipeline
    )]] for (unsigned i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}; i++) {
        auto in_pack = data_pipe::read();

    SoftmaxExpPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}; j++) {
            exp_res[j] = exp_table_latency[softmax_latency_idx_from_real_val<
                typename ExtractPipeType<data_pipe>::value_type::value_type,
                CONFIG_T>(in_pack[j])];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        [[intel::fpga_register]] typename CONFIG_T::exp_table_t exp_sum =
            reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        // Multiply previously calculated exponetials with the reciprocal of the sum
        [[intel::fpga_register]] typename CONFIG_T::inv_table_t inv_exp_sum =
            invert_table_latency[softmax_latency_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];

        typename ExtractPipeType<res_pipe>::value_type out_pack;
    SoftmaxInvPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }

        res_pipe::write(out_pack);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_legacy_stream() {
#include "activation_tables/exp_table_legacy.tb"
#include "activation_tables/invert_table_legacy.tb"

    // Index into the lookup table based on data for exponentials
    [[intel::fpga_register]]
    typename CONFIG_T::table_t exp_res[std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}];
    [[intel::fpga_register]] typename CONFIG_T::table_t exp_diff_res;
    [[intel::fpga_register]] typename ExtractPipeType<data_pipe>::value_type::value_type
        data_cache[std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}];

SoftmaxInitLoop:
    [[intel::initiation_interval(1
    )]] for (unsigned s = 0; s < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}; s++) {
        auto in_pack = data_pipe::read();

    SoftmaxInitPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}; j++) {
            data_cache[j] = in_pack[j];
            exp_res[j] = 0;
        }

    SoftmaxExpLoop:
        #pragma unroll
        for (int i = 0; i < std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}; i++) {
        SoftmaxExpInner:
            #pragma unroll
            for (int j = 0; j < std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}; j++) {
                if (i == j) {
                    exp_diff_res = 1;
                } else {
                    int data_round = ((data_cache[j] - data_cache[i]) * CONFIG_T::table_size / 16).to_int();
                    int index = data_round + 8 * CONFIG_T::table_size / 16;
                    if (index < 0)
                        index = 0;
                    if (index > CONFIG_T::table_size - 1)
                        index = CONFIG_T::table_size - 1;
                    exp_diff_res = exp_table_legacy[index];
                }
                exp_res[i] += exp_diff_res;
            }
        }

        typename ExtractPipeType<res_pipe>::value_type out_pack;
    SoftmaxInvPackLoop:
        #pragma unroll
        for (unsigned j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            int exp_res_index = (exp_res[j] * CONFIG_T::table_size / 64).to_int();
            if (exp_res_index < 0)
                exp_res_index = 0;
            if (exp_res_index > CONFIG_T::table_size - 1)
                exp_res_index = CONFIG_T::table_size - 1;
            out_pack[j] =
                static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(invert_table_legacy[exp_res_index]);
        }

        res_pipe::write(out_pack);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_argmax_stream() {
    [[intel::initiation_interval(1
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

        #pragma unroll
        for (int i = 0; i < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
            out_data[i] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(0);
        }

        [[intel::fpga_register]] typename ExtractPipeType<data_pipe>::value_type::value_type maximum = in_data[0];
        [[intel::fpga_register]] int idx = 0;

        [[intel::initiation_interval(1)]] for (int i = 1;
                                               i < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};
                                               i++) {
            if (in_data[i] > maximum) {
                maximum = in_data[i];
                idx = i;
            }
        }

        out_data[idx] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(1);
        res_pipe::write(out_data);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void softmax_stream() {
    switch (CONFIG_T::implementation) {
    case softmax_implementation::latency:
        softmax_latency_stream<data_pipe, res_pipe, CONFIG_T>();
        break;
    case softmax_implementation::stable:
        softmax_stable_stream<data_pipe, res_pipe, CONFIG_T>();
        break;
    case softmax_implementation::legacy:
        softmax_legacy_stream<data_pipe, res_pipe, CONFIG_T>();
        break;
    case softmax_implementation::argmax:
        softmax_argmax_stream<data_pipe, res_pipe, CONFIG_T>();
        break;
    default:
        softmax_stable_stream<data_pipe, res_pipe, CONFIG_T>();
        break;
    }
}

// *************************************************
//       TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void dense_tanh_stream() {
#include "activation_tables/tanh_table.tb"
    static const int MAX_VALUE = 4;

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{} / multiplier_limit;

TanHActLoop:
    [[intel::initiation_interval(pipeline
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {

        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    TanHPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            [[intel::fpga_register]] typename ExtractPipeType<data_pipe>::value_type::value_type absoluteValue;

            if (in_data[j] < 0)
                absoluteValue = (-1) * in_data[j];
            else
                absoluteValue = in_data[j];

            [[intel::fpga_register]] int index;
            if (absoluteValue <= MAX_VALUE)
                index = (absoluteValue * (CONFIG_T::table_size / MAX_VALUE)).to_int();
            else
                index = CONFIG_T::table_size - 1;

            if (in_data[j] > 0)
                out_data[j] = tanh_table[index];
            else
                out_data[j] = -tanh_table[index];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void sigmoid_stream() {
#include "activation_tables/sigmoid_table.tb"
    static const int MAX_VALUE = 8;

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{} / multiplier_limit;

SigmoidActLoop:
    [[intel::initiation_interval(pipeline
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {
        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    SigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            [[intel::fpga_register]] typename ExtractPipeType<data_pipe>::value_type::value_type absoluteValue;

            if (in_data[j] < 0)
                absoluteValue = (-1) * in_data[j];
            else
                absoluteValue = in_data[j];

            [[intel::fpga_register]] int index;
            if (absoluteValue <= MAX_VALUE)
                index = (absoluteValue * (CONFIG_T::table_size / MAX_VALUE)).to_int();
            else
                index = CONFIG_T::table_size - 1;

            if (in_data[j] > 0)
                out_data[j] = sigmoid_table[index];
            else
                out_data[j] = 1 - sigmoid_table[index];
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************
// Note - Theano and Tensorflow might have different definitions for hard sigmoid; could provide two implementations
template <class data_pipe, class res_pipe, typename CONFIG_T> void hard_sigmoid_stream() {

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{} / multiplier_limit;

HardSigmoidActLoop:
    [[intel::initiation_interval(pipeline
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {

        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    HardSigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            [[intel::fpga_register]] auto datareg = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (datareg > 1)
                datareg = 1;
            else if (datareg < 0)
                datareg = 0;
            out_data[j] = datareg;
        }

        res_pipe::write(out_data);
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void hard_tanh_stream() {

    constexpr unsigned multiplier_limit =
        DIV_ROUNDUP(std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{}, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{} / multiplier_limit;

HardSigmoidActLoop:
    [[intel::initiation_interval(pipeline
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {

        auto in_data = data_pipe::read();
        typename ExtractPipeType<res_pipe>::value_type out_data;

    HardSigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            auto sigmoid = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (sigmoid > 1)
                sigmoid = 1;
            else if (sigmoid < 0)
                sigmoid = 0;
            out_data[j] = 2 * sigmoid - 1;
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void binary_tanh_stream() {
BinaryTanHActLoop:
    [[intel::initiation_interval(1
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {

        [[intel::fpga_register]] auto in_data = data_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    BinaryTanHPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            if (in_data[j] > 0)
                out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(1);
            else
                out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(-1);
        }

        res_pipe::write(out_data);
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_pipe, class res_pipe, typename CONFIG_T> void ternary_tanh_stream() {
TernaryTanHActLoop:
    [[intel::initiation_interval(1
    )]] for (int i = 0; i < CONFIG_T::n_in / std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; i++) {

        [[intel::fpga_register]] auto in_data = data_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    TernaryTanHPackLoop:
        #pragma unroll
        for (int j = 0; j < std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{}; j++) {
            if (in_data[j] > 1)
                out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(1);
            else if (in_data[j] <= -1)
                out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(-1);
            else
                out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(0);
        }

        res_pipe::write(out_data);
    }
}

} // namespace nnet

#endif
