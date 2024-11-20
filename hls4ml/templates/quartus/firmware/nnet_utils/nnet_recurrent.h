#ifndef NNET_RECURRENT_H_
#define NNET_RECURRENT_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_recurrent_activation.h"

namespace nnet {

//----------------------
// Utils
//----------------------

template <class data_T, class res_T, class weight_t, int N_IN, int N_OUT>
void multiply_W(data_T input[N_IN], res_T out[N_OUT], const weight_t weight[N_IN * N_OUT]) {
MULTIPLY_W_LOOP_I:
    #pragma unroll
    for (int i = 0; i < N_OUT; i++) {
        out[i] = 0;

    MULTIPLY_W_LOOP_J:
        #pragma unroll
        for (int j = 0; j < N_IN; j++) {
            out[i] += input[j] * weight[i * N_IN + j];
        }
    }
}

template <class data_T, class res_T, class weight_t, int N_OUT>
void multiply_U(data_T input[N_OUT], res_T out[N_OUT], const weight_t weight[N_OUT * N_OUT]) {
MULTIPLY_U_LOOP_I:
    #pragma unroll
    for (int i = 0; i < N_OUT; i++) {
        out[i] = 0;

    MULTIPLY_U_LOOP_J:
        #pragma unroll
        for (int j = 0; j < N_OUT; j++) {
            out[i] += input[j] * weight[i * N_OUT + j];
        }
    }
}

template <class data_T, class res_T, class bias_t, int N>
void add_bias(data_T inputs[N], res_T out[N], const bias_t bias[N]) {
ADD_BIAS_LOOP:
    #pragma unroll
    for (int i = 0; i < N; i++) {
        out[i] = inputs[i] + bias[i];
    }
}

template <class data_T, class res_T, int N> void multiply_vectors(data_T in1[N], data_T in2[N], res_T out[N]) {
MULTIPLY_VECT_LOOP:
    #pragma unroll
    for (int i = 0; i < N; i++) {
        out[i] = in1[i] * in2[i];
    }
}

template <class data_T, class res_T, int N> void add_vectors(data_T in1[N], data_T in2[N], res_T out[N]) {
ADD_VECTOR_LOOP:
    #pragma unroll
    for (int i = 0; i < N; i++) {
        out[i] = in1[i] + in2[i];
    }
}

//----------------------
// GRU
//----------------------

struct gru_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
    static const unsigned n_units = 1;
    static const unsigned n_timesteps = 1;
    static const unsigned n_outputs = 1;
    static const bool return_sequences = false;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool pytorch_order = false;
    static const bool store_weights_in_bram = false;

    // Activation
    template <class x_T, class y_T, class config_T> using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;

    template <class x_T, class y_T, class config_T> using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void gru_cell(
    data_T x[CONFIG_T::n_in],
    res_T h[CONFIG_T::n_units],
    const typename CONFIG_T::weight_t weights[3 * CONFIG_T::n_units * CONFIG_T::n_in],
    const typename CONFIG_T::weight_t recurrent_weights[3 * CONFIG_T::n_units * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t bias[3 * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t recurrent_bias[3 * CONFIG_T::n_units]
) {
    static constexpr int recurrent_unroll_factor = CONFIG_T::n_units / CONFIG_T::reuse_factor;
    // A matrix containing the values of matrix product between input (x) and weights (weights), for update, reset and
    // candidate state gates, for each of the units
    hls_register typename CONFIG_T::accum_t mat_mul_x_w[3 * CONFIG_T::n_units];
    nnet::dense_resource<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config_x>(
        x, mat_mul_x_w, weights, bias
    );

    // A matrix containing the values of matrix product between previou state (h) and recurrent weights (recurrent_weights),
    // for update, reset and candidate state gates, for each of the units
    hls_register typename CONFIG_T::accum_t mat_mul_h_wr[3 * CONFIG_T::n_units];
    nnet::dense_resource<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config_h>(
        h, mat_mul_h_wr, recurrent_weights, recurrent_bias
    );

    // A vector containing both the values of z(t) and r(t) for every state
    hls_register typename CONFIG_T::accum_t z_r[2 * CONFIG_T::n_units];

    // Add the individual vectors from the multiplication of mat_mul_x_w = Wx*x(t) and mat_mul_h_wr = Wh*h(t-1)
    // Unrolled fully, no DSPs used
    #pragma unroll
    for (int i = 0; i < (2 * CONFIG_T::n_units); i++) {
        z_r[i] = mat_mul_x_w[i] + mat_mul_h_wr[i];
    }

    // Activation on z(t) and r(t)
    hls_register typename CONFIG_T::accum_t z_r_act[2 * CONFIG_T::n_units];
    CONFIG_T::template activation_recr<
        typename CONFIG_T::accum_t,
        typename CONFIG_T::accum_t,
        typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(z_r, z_r_act);

    // A matrix containing the values of Hadamard product between r(t) = z_r_act[n_units:2*n_units] and h(t-1) = h
    hls_register typename CONFIG_T::accum_t hadamard_r_h[CONFIG_T::n_units];
    #pragma unroll recurrent_unroll_factor
    for (int i = 0; i < (CONFIG_T::n_units); i++) {
        if (CONFIG_T::pytorch_order)
            hadamard_r_h[i] = z_r_act[i] * mat_mul_h_wr[i + 2 * CONFIG_T::n_units];
        else
            hadamard_r_h[i] = z_r_act[i + CONFIG_T::n_units] * mat_mul_h_wr[i + 2 * CONFIG_T::n_units];
    }

    // The candidate state; X * W_{hx} + hadmard(r(t), h_(t-1)) * W_{hh} + b_{h}
    typename CONFIG_T::accum_t h_cand[CONFIG_T::n_units];
    // Addition - can unroll fully; no DSPs used here
    #pragma unroll
    for (int i = 0; i < (CONFIG_T::n_units); i++) {
        h_cand[i] = mat_mul_x_w[i + 2 * CONFIG_T::n_units] + hadamard_r_h[i];
    }

    // Activation on candidate state
    hls_register typename CONFIG_T::accum_t h_cand_act[CONFIG_T::n_units];
    CONFIG_T::template activation<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, typename CONFIG_T::ACT_CONFIG_T>::
        activation(h_cand, h_cand_act);

    // Update state
    #pragma unroll recurrent_unroll_factor
    for (int i = 0; i < (CONFIG_T::n_units); i++) {
        if (CONFIG_T::pytorch_order)
            h[i] = static_cast<res_T>(
                h_cand_act[i] * (1 - z_r_act[i + CONFIG_T::n_units]) + h[i] * z_r_act[i + CONFIG_T::n_units]
            );
        else
            h[i] = static_cast<res_T>(h_cand_act[i] * (1 - z_r_act[i]) + h[i] * z_r_act[i]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void gru(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_outputs * CONFIG_T::n_units],
    const typename CONFIG_T::weight_t weights[3 * CONFIG_T::n_units * CONFIG_T::n_in],
    const typename CONFIG_T::weight_t recurrent_weights[3 * CONFIG_T::n_units * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t bias[3 * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t recurrent_bias[3 * CONFIG_T::n_units]
) {

    hls_register data_T x[CONFIG_T::n_in];
    hls_register res_T h[CONFIG_T::n_units];

    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_units; i++) {
        h[i] = 0;
    }

    // Loop depedency - cannot pipeline
    #pragma disable_loop_pipelining
    for (int t = 0; t < CONFIG_T::n_timesteps; t++) {
        // Get data at current time step
        #pragma unroll
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            x[j] = data[j + t * CONFIG_T::n_in];
        }

        nnet::gru_cell<data_T, res_T, CONFIG_T>(x, h, weights, recurrent_weights, bias, recurrent_bias);

        if (CONFIG_T::return_sequences) {
            #pragma unroll
            for (int i = 0; i < CONFIG_T::n_units; i++) {
                res[CONFIG_T::n_units * t + i] = h[i];
            }
        }
    }

    if (!CONFIG_T::return_sequences) {
        #pragma unroll
        for (int i = 0; i < (CONFIG_T::n_units); i++) {
            res[i] = h[i];
        }
    }
}

//----------------------
// SimpleRNN
//----------------------

struct simpleRNN_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
    static const unsigned n_outputs = 1;
    static const unsigned n_timesteps = 1;
    static const bool return_sequences = false;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;

    // Activation
    template <class x_T, class y_T, class config_T> using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;

    template <class x_T, class y_T, class config_T> using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void simple_rnn_cell(
    data_T inputs[CONFIG_T::n_in],
    res_T hidden_state[CONFIG_T::n_out],
    res_T hidden_state_o[CONFIG_T::n_out],
    const typename CONFIG_T::weight_t kernel[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t rec_kernel[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::bias_t bias[CONFIG_T::n_out]
) {
    // Weight multiplication
    typename CONFIG_T::accum_t afterW[CONFIG_T::n_out] hls_register;
    multiply_W<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_in, CONFIG_T::n_out>(
        inputs, afterW, kernel
    );

    // Bias addition
    typename CONFIG_T::accum_t afterBias[CONFIG_T::n_out] hls_register;
    add_bias<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, typename CONFIG_T::bias_t, CONFIG_T::n_out>(
        afterW, afterBias, bias
    );

    // Hidden state
    typename CONFIG_T::accum_t hiddenCand[CONFIG_T::n_out] hls_register;
    multiply_U<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_out>(
        hidden_state, hiddenCand, rec_kernel
    );

    // Vector addition
    typename CONFIG_T::accum_t afterAdd[CONFIG_T::n_out];
    add_vectors<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, CONFIG_T::n_out>(afterBias, hiddenCand, afterAdd);

    // Activation
    CONFIG_T::template activation<typename CONFIG_T::accum_t, data_T, typename CONFIG_T::ACT_CONFIG_T>::activation(
        afterAdd, hidden_state_o
    );
}

template <class data_T, class res_T, typename CONFIG_T>
void simple_rnn(
    data_T data[CONFIG_T::n_timesteps * CONFIG_T::n_in],
    res_T res[CONFIG_T::n_outputs * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t kernel[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t rec_kernel[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::bias_t bias[CONFIG_T::n_out]
) {
    res_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timesteps + 1] hls_register;
    res_T hidden_state_temp[CONFIG_T::n_out] hls_register;
    res_T h[CONFIG_T::n_out] hls_register;
    data_T in[CONFIG_T::n_in] hls_register;

// Set initially hidden state (output) to zero
INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][0] = 0;
    }

    #pragma disable_loop_pipelining
    for (int i = 0; i < CONFIG_T::n_timesteps; i++) {

        // Data at current time step
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_in; x++) {
            in[x] = data[x + i * CONFIG_T::n_in];
        }

        // Hidden state at current time step
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state_temp[x] = hidden_state[x][i];
        }

        // Do SimpleRNN
        simple_rnn_cell<data_T, res_T, CONFIG_T>(in, hidden_state_temp, h, kernel, rec_kernel, bias);

        // Write result
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state[x][i + 1] = h[x];
        }
    }

    if (CONFIG_T::return_sequences == 0) {
        // Output when return_sequences is false
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            res[x] = hidden_state[x][CONFIG_T::n_timesteps];
        }
    } else {
        // Output when return_sequences is true
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_timesteps; x++) {
            #pragma unroll
            for (int h = 0; h < CONFIG_T::n_out; h++) {
                res[x * CONFIG_T::n_out + h] = hidden_state[h][x + 1];
            }
        }
    }
}
//----------------------
// SimpleRNN with pytorch biases
//----------------------

struct simpleRNN_pytorch_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
    static const unsigned n_outputs = 1;
    static const unsigned n_timesteps = 1;
    static const bool return_sequences = false;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;

    // Activation
    template <class x_T, class y_T, class config_T> using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;

    template <class x_T, class y_T, class config_T> using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void simple_rnn_pytorch_cell(
    data_T inputs[CONFIG_T::n_in],
    res_T hidden_state[CONFIG_T::n_out],
    res_T hidden_state_o[CONFIG_T::n_out],
    const typename CONFIG_T::weight_t kernel[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t rec_kernel[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::bias_t bias[CONFIG_T::n_out],
    const typename CONFIG_T::bias_t rec_bias[CONFIG_T::n_out]
) {
    // Weight multiplication
    typename CONFIG_T::accum_t afterW[CONFIG_T::n_out] hls_register;
    multiply_W<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_in, CONFIG_T::n_out>(
        inputs, afterW, kernel
    );

    // Bias addition
    typename CONFIG_T::accum_t afterBias[CONFIG_T::n_out] hls_register;
    add_bias<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, typename CONFIG_T::bias_t, CONFIG_T::n_out>(
        afterW, afterBias, bias
    );

    // Hidden state
    typename CONFIG_T::accum_t hiddenCand[CONFIG_T::n_out] hls_register;
    multiply_U<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_out>(
        hidden_state, hiddenCand, rec_kernel
    );

    // Hidden state bias addition
    typename CONFIG_T::accum_t hiddenBias[CONFIG_T::n_out] hls_register;
    add_bias<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, typename CONFIG_T::bias_t, CONFIG_T::n_out>(
        hiddenCand, hiddenBias, rec_bias
    );

    // Vector addition
    typename CONFIG_T::accum_t afterAdd[CONFIG_T::n_out];
    add_vectors<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, CONFIG_T::n_out>(afterBias, hiddenBias, afterAdd);

    // Activation
    CONFIG_T::template activation<typename CONFIG_T::accum_t, data_T, typename CONFIG_T::ACT_CONFIG_T>::activation(
        afterAdd, hidden_state_o
    );
}

template <class data_T, class res_T, typename CONFIG_T>
void simple_rnn_pytorch(
    data_T data[CONFIG_T::n_timesteps * CONFIG_T::n_in],
    res_T res[CONFIG_T::n_outputs * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t kernel[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t rec_kernel[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::bias_t bias[CONFIG_T::n_out],
    const typename CONFIG_T::bias_t rec_bias[CONFIG_T::n_out]
) {
    res_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timesteps + 1] hls_register;
    res_T hidden_state_temp[CONFIG_T::n_out] hls_register;
    res_T h[CONFIG_T::n_out] hls_register;
    data_T in[CONFIG_T::n_in] hls_register;

// Set initially hidden state (output) to zero
INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][0] = 0;
    }

    #pragma disable_loop_pipelining
    for (int i = 0; i < CONFIG_T::n_timesteps; i++) {

        // Data at current time step
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_in; x++) {
            in[x] = data[x + i * CONFIG_T::n_in];
        }

        // Hidden state at current time step
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state_temp[x] = hidden_state[x][i];
        }

        // Do SimpleRNN
        simple_rnn_pytorch_cell<data_T, res_T, CONFIG_T>(in, hidden_state_temp, h, kernel, rec_kernel, bias, rec_bias);

        // Write result
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state[x][i + 1] = h[x];
        }
    }

    if (CONFIG_T::return_sequences == 0) {
        // Output when return_sequences is false
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            res[x] = hidden_state[x][CONFIG_T::n_timesteps];
        }
    } else {
        // Output when return_sequences is true
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_timesteps; x++) {
            #pragma unroll
            for (int h = 0; h < CONFIG_T::n_out; h++) {
                res[x * CONFIG_T::n_out + h] = hidden_state[h][x + 1];
            }
        }
    }
}

//----------------------
// LSTM
//----------------------

struct lstm_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 1;
    static const unsigned n_out = 1;
    static const unsigned n_outputs = 1;

    static const unsigned n_timesteps = 1;
    static const bool return_sequences = false;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;

    // Activation
    template <class x_T, class y_T, class config_T> using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;

    template <class x_T, class y_T, class config_T> using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void lstm_cell(
    data_T inputs[CONFIG_T::n_in],
    res_T hidden_state[CONFIG_T::n_out],
    res_T hidden_state_o[CONFIG_T::n_out],
    res_T cell_state[CONFIG_T::n_out],
    res_T cell_state_o[CONFIG_T::n_out],
    const typename CONFIG_T::weight_t WI[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t WF[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t WC[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t WO[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t RWI[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t RWF[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t RWC[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t RWO[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::bias_t BI[CONFIG_T::n_out],
    const typename CONFIG_T::bias_t BF[CONFIG_T::n_out],
    const typename CONFIG_T::bias_t BC[CONFIG_T::n_out],
    const typename CONFIG_T::bias_t BO[CONFIG_T::n_out]
) {

    // Internals definitions
    typename CONFIG_T::accum_t i_afterW[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t i_afterBias[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t c_afterW[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t c_afterBias[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t o_afterW[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t o_afterBias[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t f_afterW[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t f_afterBias[CONFIG_T::n_out] hls_register;

    // Hidden state Gate candidates, intermediate variables
    typename CONFIG_T::accum_t i_hiddenCand[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t f_hiddenCand[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t c_hiddenCand[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t o_hiddenCand[CONFIG_T::n_out] hls_register;

    // After addition, intermediate variables
    typename CONFIG_T::accum_t i_afterAdd[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t f_afterAdd[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t c_afterAdd[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t o_afterAdd[CONFIG_T::n_out] hls_register;

    // Gate outputs
    typename CONFIG_T::accum_t gate_i[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t gate_f[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t gate_c[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t gate_o[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t gate_ic[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t gate_forget[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t h[CONFIG_T::n_out] hls_register;

    // Intermediate variable cell calculation
    typename CONFIG_T::accum_t cell_act_multp[CONFIG_T::n_out] hls_register;
    typename CONFIG_T::accum_t cell_act_add[CONFIG_T::n_out] hls_register;

    //-----------Gate I Calculations
    // Weight multiplication
    multiply_W<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_in, CONFIG_T::n_out>(
        inputs, i_afterW, WI
    );

    // Bias addition
    add_bias<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, typename CONFIG_T::bias_t, CONFIG_T::n_out>(
        i_afterW, i_afterBias, BI
    );

    // Hidden Candidate
    multiply_U<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_out>(
        hidden_state, i_hiddenCand, RWI
    );

    // Vector addition
    add_vectors<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, CONFIG_T::n_out>(
        i_afterBias, i_hiddenCand, i_afterAdd
    );

    // Activation
    CONFIG_T::template activation_recr<
        typename CONFIG_T::accum_t,
        typename CONFIG_T::accum_t,
        typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(i_afterAdd, gate_i);

    //-----------Gate F Calculations
    // Weight multiplication
    multiply_W<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_in, CONFIG_T::n_out>(
        inputs, f_afterW, WF
    );

    // Bias addition
    add_bias<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, typename CONFIG_T::bias_t, CONFIG_T::n_out>(
        f_afterW, f_afterBias, BF
    );

    // Hidden Candidate
    multiply_U<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_out>(
        hidden_state, f_hiddenCand, RWF
    );

    // Vector addition
    add_vectors<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, CONFIG_T::n_out>(
        f_afterBias, f_hiddenCand, f_afterAdd
    );

    // Activation
    CONFIG_T::template activation_recr<
        typename CONFIG_T::accum_t,
        typename CONFIG_T::accum_t,
        typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(f_afterAdd, gate_f);

    //-----------Gate C Calculations
    // Weight multiplication
    multiply_W<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_in, CONFIG_T::n_out>(
        inputs, c_afterW, WC
    );

    // Bias addition
    add_bias<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, typename CONFIG_T::bias_t, CONFIG_T::n_out>(
        c_afterW, c_afterBias, BC
    );

    // Hidden Candidate
    multiply_U<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_out>(
        hidden_state, c_hiddenCand, RWC
    );

    // Vector addition
    add_vectors<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, CONFIG_T::n_out>(
        c_afterBias, c_hiddenCand, c_afterAdd
    );

    // Activation
    CONFIG_T::template activation<
        typename CONFIG_T::accum_t,
        typename CONFIG_T::accum_t,
        typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(c_afterAdd, gate_c);

    //-----------gate I and C multiply
    // Vector multiplication
    multiply_vectors<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, CONFIG_T::n_out>(gate_i, gate_c, gate_ic);

    //-----------Gate O Calculations
    // Weight multiplication
    multiply_W<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_in, CONFIG_T::n_out>(
        inputs, o_afterW, WO
    );

    // Bias addition
    add_bias<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, typename CONFIG_T::bias_t, CONFIG_T::n_out>(
        o_afterW, o_afterBias, BO
    );

    // Hidden Candidate
    multiply_U<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, CONFIG_T::n_out>(
        hidden_state, o_hiddenCand, RWO
    );

    // Vector addition
    add_vectors<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, CONFIG_T::n_out>(
        o_afterBias, o_hiddenCand, o_afterAdd
    );

    // Activation
    CONFIG_T::template activation_recr<
        typename CONFIG_T::accum_t,
        typename CONFIG_T::accum_t,
        typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(o_afterAdd, gate_o);

    //-----------Cell State Calculation
    // Vector multiplication
    multiply_vectors<typename CONFIG_T::accum_t, res_T, CONFIG_T::n_out>(gate_f, cell_state, cell_act_multp);

    // Vector addition
    add_vectors<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, CONFIG_T::n_out>(
        gate_ic, cell_act_multp, cell_act_add
    );

    //-----------Forget gate Calculation
    // Activation
    CONFIG_T::template activation<
        typename CONFIG_T::accum_t,
        typename CONFIG_T::accum_t,
        typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(cell_act_add, gate_forget);

    // Vector multiplication
    multiply_vectors<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t, CONFIG_T::n_out>(gate_o, gate_forget, h);

OUTPUT_WRITE_LOOP:
    #pragma unroll
    for (int x = (CONFIG_T::n_out - 1); x >= 0; x--) {
        hidden_state_o[x] = h[x];
        cell_state_o[x] = cell_act_add[x];
    }
}

template <class data_T, class res_T, class CONFIG_T>
void lstm(
    data_T data[CONFIG_T::n_timesteps * CONFIG_T::n_in],
    res_T res[CONFIG_T::n_outputs * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t WI[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t WF[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t WC[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t WO[CONFIG_T::n_in * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t RWI[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t RWF[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t RWC[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::weight_t RWO[CONFIG_T::n_out * CONFIG_T::n_out],
    const typename CONFIG_T::bias_t BI[CONFIG_T::n_out],
    const typename CONFIG_T::bias_t BF[CONFIG_T::n_out],
    const typename CONFIG_T::bias_t BC[CONFIG_T::n_out],
    const typename CONFIG_T::bias_t BO[CONFIG_T::n_out]
) {
    res_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timesteps + 1] hls_register;
    res_T hidden_state_temp[CONFIG_T::n_out] hls_register;
    res_T cell_state[CONFIG_T::n_out][CONFIG_T::n_timesteps + 1] hls_register;
    res_T cell_state_temp[CONFIG_T::n_out] hls_register;
    res_T h[CONFIG_T::n_out] hls_register;
    res_T c[CONFIG_T::n_out] hls_register;
    data_T in[CONFIG_T::n_in] hls_register;

// Set initially hidden state (output) to zero
INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][0] = 0;
        cell_state[x][0] = 0;
    }

    // Input dimension
    #pragma disable_loop_pipelining
    for (int i = 0; i < CONFIG_T::n_timesteps; i++) {
        // Data at current time step
        for (int x = 0; x < CONFIG_T::n_in; x++) {
            in[x] = data[x + i * CONFIG_T::n_in];
        }

        // Hidden state at current time step
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state_temp[x] = hidden_state[x][i];
            cell_state_temp[x] = cell_state[x][i];
        }

        // Do LSTM
        lstm_cell<data_T, res_T, CONFIG_T>(
            in, hidden_state_temp, h, cell_state_temp, c, WI, WF, WC, WO, RWI, RWF, RWC, RWO, BI, BF, BC, BO
        );

        // Write result
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state[x][i + 1] = h[x];
            cell_state[x][i + 1] = c[x];
        }
    }

    if (CONFIG_T::return_sequences == 0) {
        // Output when return_sequences is false
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            res[x] = hidden_state[x][CONFIG_T::n_timesteps];
        }
    } else {
        // Output when return_sequences is true
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_timesteps; x++) {
            for (int h = 0; h < CONFIG_T::n_out; h++) {
                res[x * CONFIG_T::n_out + h] = hidden_state[h][x + 1];
            }
        }
    }
}

} // namespace nnet

#endif
