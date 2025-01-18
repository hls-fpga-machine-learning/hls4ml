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
void multiply_W(const data_T &input, res_T &out, const weight_t &weight) {
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
void multiply_U(const data_T &input, res_T &out, const weight_t &weight) {
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
void add_bias(const data_T &inputs, res_T &out, const bias_t &bias) {
ADD_BIAS_LOOP:
    #pragma unroll
    for (int i = 0; i < N; i++) {
        out[i] = inputs[i] + bias[i];
    }
}

template <class data1_T, class data2_T, class res_T, int N>
void multiply_vectors(const data1_T &in1, const data2_T &in2, res_T &out) {
MULTIPLY_VECT_LOOP:
    #pragma unroll
    for (int i = 0; i < N; i++) {
        out[i] = in1[i] * in2[i];
    }
}

template <class data1_T, class data2_T, class res_T, int N>
void add_vectors(const data1_T &in1, const data2_T &in2, res_T &out) {
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
    static const bool store_weights_in_bram = false;

    // Activation
    template <class x_T, class y_T, class config_T> using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;

    template <class x_T, class y_T, class config_T> using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

template <class data_T, class h_T, typename CONFIG_T>
void gru_cell(const data_T &x, h_T &h, const typename CONFIG_T::weight_t &weights,
              const typename CONFIG_T::recurrent_weight_t &recurrent_weights, const typename CONFIG_T::bias_t &bias,
              const typename CONFIG_T::recurrent_bias_t &recurrent_bias) {
    static constexpr int recurrent_unroll_factor = CONFIG_T::n_units / CONFIG_T::reuse_factor;
    // A matrix containing the values of matrix product between input (x) and weights (weights), for update, reset and
    // candidate state gates, for each of the units

    using accum_array_T = array<typename CONFIG_T::accum_t, 3 * CONFIG_T::n_units>;

    [[intel::fpga_register]] accum_array_T mat_mul_x_w;
    nnet::dense_resource<data_T, accum_array_T, typename CONFIG_T::mult_config_x>(x, mat_mul_x_w, weights, bias);

    // A matrix containing the values of matrix product between previou state (h) and recurrent weights (recurrent_weights),
    // for update, reset and candidate state gates, for each of the units
    [[intel::fpga_register]] accum_array_T mat_mul_h_wr;
    nnet::dense_resource<h_T, accum_array_T, typename CONFIG_T::mult_config_h>(h, mat_mul_h_wr, recurrent_weights,
                                                                               recurrent_bias);

    // A vector containing both the values of z(t) and r(t) for every state
    using z_activ_array_T = array<typename CONFIG_T::accum_t, 2 * CONFIG_T::n_units>;
    [[intel::fpga_register]] z_activ_array_T z_r;

    // Add the individual vectors from the multiplication of mat_mul_x_w = Wx*x(t) and mat_mul_h_wr = Wh*h(t-1)
    // Unrolled fully, no DSPs used
    #pragma unroll
    for (int i = 0; i < (2 * CONFIG_T::n_units); i++) {
        z_r[i] = mat_mul_x_w[i] + mat_mul_h_wr[i];
    }

    // Activation on z(t) and r(t)
    [[intel::fpga_register]] z_activ_array_T z_r_act;
    CONFIG_T::template activation_recr<z_activ_array_T, z_activ_array_T,
                                       typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(z_r, z_r_act);

    // A matrix containing the values of Hadamard product between r(t) = z_r_act[n_units:2*n_units] and h(t-1) = h
    using h_activ_array_T = array<typename CONFIG_T::accum_t, CONFIG_T::n_units>;
    [[intel::fpga_register]] h_activ_array_T hadamard_r_h;
    #pragma unroll recurrent_unroll_factor
    for (int i = 0; i < (CONFIG_T::n_units); i++) {
        hadamard_r_h[i] = z_r_act[i + CONFIG_T::n_units] * mat_mul_h_wr[i + 2 * CONFIG_T::n_units];
    }

    // The candidate state; X * W_{hx} + hadmard(r(t), h_(t-1)) * W_{hh} + b_{h}
    [[intel::fpga_register]] h_activ_array_T h_cand;
    // Addition - can unroll fully; no DSPs used here
    #pragma unroll
    for (int i = 0; i < (CONFIG_T::n_units); i++) {
        h_cand[i] = mat_mul_x_w[i + 2 * CONFIG_T::n_units] + hadamard_r_h[i];
    }

    // Activation on candidate state
    [[intel::fpga_register]] h_activ_array_T h_cand_act;
    CONFIG_T::template activation<h_activ_array_T, h_activ_array_T, typename CONFIG_T::ACT_CONFIG_T>::activation(h_cand,
                                                                                                                 h_cand_act);

    // Update state
    #pragma unroll recurrent_unroll_factor
    for (int i = 0; i < (CONFIG_T::n_units); i++) {
        h[i] = static_cast<typename h_T::value_type>(h_cand_act[i] * (1 - z_r_act[i]) + h[i] * z_r_act[i]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void gru(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
         const typename CONFIG_T::recurrent_weight_t &recurrent_weights, const typename CONFIG_T::bias_t &bias,
         const typename CONFIG_T::recurrent_bias_t &recurrent_bias) {

    using h_T = array<typename res_T::value_type, CONFIG_T::n_units>;
    [[intel::fpga_register]] data_T x;
    [[intel::fpga_register]] h_T h;

    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_units; i++) {
        h[i] = 0;
    }

    // Loop depedency - cannot pipeline
    [[intel::disable_loop_pipelining]] for (int t = 0; t < CONFIG_T::n_timesteps; t++) {
        // Get data at current time step
        #pragma unroll
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            x[j] = data[j + t * CONFIG_T::n_in];
        }

        nnet::gru_cell<data_T, h_T, CONFIG_T>(x, h, weights, recurrent_weights, bias, recurrent_bias);

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

template <class data_T, class h_T, class res_T, typename CONFIG_T>
void gru_init_state(const data_T &data, const h_T &hin, res_T &res, const typename CONFIG_T::weight_t &weights,
                    const typename CONFIG_T::recurrent_weight_t &recurrent_weights, const typename CONFIG_T::bias_t &bias,
                    const typename CONFIG_T::recurrent_bias_t &recurrent_bias) {

    [[intel::fpga_register]] data_T x;

    [[intel::fpga_register]] h_T h = hin;

    // Loop depedency - cannot pipeline
    [[intel::disable_loop_pipelining]] for (int t = 0; t < CONFIG_T::n_timesteps; t++) {
        // Get data at current time step
        #pragma unroll
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            x[j] = data[j + t * CONFIG_T::n_in];
        }

        nnet::gru_cell<data_T, h_T, CONFIG_T>(x, h, weights, recurrent_weights, bias, recurrent_bias);

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

template <class in_T, class h_T, typename CONFIG_T>
void simple_rnn_cell(const in_T &inputs, h_T &hidden_state, h_T &hidden_state_o, const typename CONFIG_T::weight_t &kernel,
                     const typename CONFIG_T::recurrent_weight_t &rec_kernel, const typename CONFIG_T::bias_t &bias) {

    using accum_array_T = array<typename CONFIG_T::accum_t, CONFIG_T::n_out>;
    // Weight multiplication
    [[intel::fpga_register]] accum_array_T afterW;
    multiply_W<in_T, accum_array_T, typename CONFIG_T::weight_t, CONFIG_T::n_in, CONFIG_T::n_out>(inputs, afterW, kernel);

    // Bias addition
    [[intel::fpga_register]] accum_array_T afterBias;
    add_bias<accum_array_T, accum_array_T, typename CONFIG_T::bias_t, CONFIG_T::n_out>(afterW, afterBias, bias);

    // Hidden state
    [[intel::fpga_register]] accum_array_T hiddenCand;
    multiply_U<h_T, accum_array_T, typename CONFIG_T::recurrent_weight_t, CONFIG_T::n_out>(hidden_state, hiddenCand,
                                                                                           rec_kernel);

    // Vector addition
    [[intel::fpga_register]] accum_array_T afterAdd;
    add_vectors<accum_array_T, accum_array_T, accum_array_T, CONFIG_T::n_out>(afterBias, hiddenCand, afterAdd);

    // Activation
    CONFIG_T::template activation<accum_array_T, h_T, typename CONFIG_T::ACT_CONFIG_T>::activation(afterAdd, hidden_state_o);
}

template <class data_T, class res_T, typename CONFIG_T>
void simple_rnn(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &kernel,
                const typename CONFIG_T::recurrent_weight_t &rec_kernel, const typename CONFIG_T::bias_t &bias) {

    using in_T = array<typename data_T::value_type, CONFIG_T::n_in>;
    using h_T = array<typename res_T::value_type, CONFIG_T::n_out>;

    [[intel::fpga_register]] h_T hidden_state[CONFIG_T::n_timesteps + 1];
    [[intel::fpga_register]] h_T hidden_state_temp;
    [[intel::fpga_register]] h_T h;
    [[intel::fpga_register]] in_T in;

// Set initially hidden state (output) to zero
INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[0][x] = 0;
    }

    [[intel::disable_loop_pipelining]] for (int i = 0; i < CONFIG_T::n_timesteps; i++) {

        // Data at current time step
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_in; x++) {
            in[x] = data[x + i * CONFIG_T::n_in];
        }

        // Hidden state at current time step
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state_temp[x] = hidden_state[i][x];
        }

        // Do SimpleRNN
        simple_rnn_cell<in_T, h_T, CONFIG_T>(in, hidden_state_temp, h, kernel, rec_kernel, bias);

        // Write result
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state[i + 1][x] = h[x];
        }
    }

    if (CONFIG_T::return_sequences == 0) {
        // Output when return_sequences is false
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            res[x] = hidden_state[CONFIG_T::n_timesteps][x];
        }
    } else {
        // Output when return_sequences is true
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_timesteps; x++) {
            #pragma unroll
            for (int h = 0; h < CONFIG_T::n_out; h++) {
                res[x * CONFIG_T::n_out + h] = hidden_state[x + 1][h];
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

template <class in_T, class h_T, typename CONFIG_T>
void lstm_cell(const in_T &inputs, h_T &hidden_state, h_T &hidden_state_o, h_T &cell_state, h_T &cell_state_o,
               const typename CONFIG_T::weight_i_t &WI, const typename CONFIG_T::weight_f_t &WF,
               const typename CONFIG_T::weight_c_t &WC, const typename CONFIG_T::weight_o_t &WO,
               const typename CONFIG_T::recurrent_weight_i_t &RWI, const typename CONFIG_T::recurrent_weight_f_t &RWF,
               const typename CONFIG_T::recurrent_weight_c_t &RWC, const typename CONFIG_T::recurrent_weight_o_t &RWO,
               const typename CONFIG_T::bias_i_t &BI, const typename CONFIG_T::bias_f_t BF,
               const typename CONFIG_T::bias_c_t &BC, const typename CONFIG_T::bias_o_t BO) {

    using accum_array_T = array<typename CONFIG_T::accum_t, CONFIG_T::n_out>;

    // Internals definitions
    [[intel::fpga_register]] accum_array_T i_afterW;
    [[intel::fpga_register]] accum_array_T i_afterBias;
    [[intel::fpga_register]] accum_array_T c_afterW;
    [[intel::fpga_register]] accum_array_T c_afterBias;
    [[intel::fpga_register]] accum_array_T o_afterW;
    [[intel::fpga_register]] accum_array_T o_afterBias;
    [[intel::fpga_register]] accum_array_T f_afterW;
    [[intel::fpga_register]] accum_array_T f_afterBias;

    // Hidden state Gate candidates, intermediate variables
    [[intel::fpga_register]] accum_array_T i_hiddenCand;
    [[intel::fpga_register]] accum_array_T f_hiddenCand;
    [[intel::fpga_register]] accum_array_T c_hiddenCand;
    [[intel::fpga_register]] accum_array_T o_hiddenCand;

    // After addition, intermediate variables
    [[intel::fpga_register]] accum_array_T i_afterAdd;
    [[intel::fpga_register]] accum_array_T f_afterAdd;
    [[intel::fpga_register]] accum_array_T c_afterAdd;
    [[intel::fpga_register]] accum_array_T o_afterAdd;

    // Gate outputs
    [[intel::fpga_register]] accum_array_T gate_i;
    [[intel::fpga_register]] accum_array_T gate_f;
    [[intel::fpga_register]] accum_array_T gate_c;
    [[intel::fpga_register]] accum_array_T gate_o;
    [[intel::fpga_register]] accum_array_T gate_ic;
    [[intel::fpga_register]] accum_array_T gate_forget;
    [[intel::fpga_register]] accum_array_T h;

    // Intermediate variable cell calculation
    [[intel::fpga_register]] accum_array_T cell_act_multp;
    [[intel::fpga_register]] accum_array_T cell_act_add;

    //-----------Gate I Calculations
    // Weight multiplication
    multiply_W<in_T, accum_array_T, typename CONFIG_T::weight_i_t, CONFIG_T::n_in, CONFIG_T::n_out>(inputs, i_afterW, WI);

    // Bias addition
    add_bias<accum_array_T, accum_array_T, typename CONFIG_T::bias_i_t, CONFIG_T::n_out>(i_afterW, i_afterBias, BI);

    // Hidden Candidate
    multiply_U<h_T, accum_array_T, typename CONFIG_T::recurrent_weight_i_t, CONFIG_T::n_out>(hidden_state, i_hiddenCand,
                                                                                             RWI);

    // Vector addition
    add_vectors<accum_array_T, accum_array_T, accum_array_T, CONFIG_T::n_out>(i_afterBias, i_hiddenCand, i_afterAdd);

    // Activation
    CONFIG_T::template activation_recr<accum_array_T, accum_array_T, typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(
        i_afterAdd, gate_i);

    //-----------Gate F Calculations
    // Weight multiplication
    multiply_W<in_T, accum_array_T, typename CONFIG_T::weight_f_t, CONFIG_T::n_in, CONFIG_T::n_out>(inputs, f_afterW, WF);

    // Bias addition
    add_bias<accum_array_T, accum_array_T, typename CONFIG_T::bias_f_t, CONFIG_T::n_out>(f_afterW, f_afterBias, BF);

    // Hidden Candidate
    multiply_U<h_T, accum_array_T, typename CONFIG_T::recurrent_weight_f_t, CONFIG_T::n_out>(hidden_state, f_hiddenCand,
                                                                                             RWF);

    // Vector addition
    add_vectors<accum_array_T, accum_array_T, accum_array_T, CONFIG_T::n_out>(f_afterBias, f_hiddenCand, f_afterAdd);

    // Activation
    CONFIG_T::template activation_recr<accum_array_T, accum_array_T, typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(
        f_afterAdd, gate_f);

    //-----------Gate C Calculations
    // Weight multiplication
    multiply_W<in_T, accum_array_T, typename CONFIG_T::weight_c_t, CONFIG_T::n_in, CONFIG_T::n_out>(inputs, c_afterW, WC);

    // Bias addition
    add_bias<accum_array_T, accum_array_T, typename CONFIG_T::bias_c_t, CONFIG_T::n_out>(c_afterW, c_afterBias, BC);

    // Hidden Candidate
    multiply_U<h_T, accum_array_T, typename CONFIG_T::recurrent_weight_c_t, CONFIG_T::n_out>(hidden_state, c_hiddenCand,
                                                                                             RWC);

    // Vector addition
    add_vectors<accum_array_T, accum_array_T, accum_array_T, CONFIG_T::n_out>(c_afterBias, c_hiddenCand, c_afterAdd);

    // Activation
    CONFIG_T::template activation<accum_array_T, accum_array_T, typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(
        c_afterAdd, gate_c);

    //-----------gate I and C multiply
    // Vector multiplication
    multiply_vectors<accum_array_T, accum_array_T, accum_array_T, CONFIG_T::n_out>(gate_i, gate_c, gate_ic);

    //-----------Gate O Calculations
    // Weight multiplication
    multiply_W<in_T, accum_array_T, typename CONFIG_T::weight_o_t, CONFIG_T::n_in, CONFIG_T::n_out>(inputs, o_afterW, WO);

    // Bias addition
    add_bias<accum_array_T, accum_array_T, typename CONFIG_T::bias_o_t, CONFIG_T::n_out>(o_afterW, o_afterBias, BO);

    // Hidden Candidate
    multiply_U<h_T, accum_array_T, typename CONFIG_T::recurrent_weight_o_t, CONFIG_T::n_out>(hidden_state, o_hiddenCand,
                                                                                             RWO);

    // Vector addition
    add_vectors<accum_array_T, accum_array_T, accum_array_T, CONFIG_T::n_out>(o_afterBias, o_hiddenCand, o_afterAdd);

    // Activation
    CONFIG_T::template activation_recr<accum_array_T, accum_array_T, typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(
        o_afterAdd, gate_o);

    //-----------Cell State Calculation
    // Vector multiplication
    multiply_vectors<accum_array_T, h_T, accum_array_T, CONFIG_T::n_out>(gate_f, cell_state, cell_act_multp);

    // Vector addition
    add_vectors<accum_array_T, accum_array_T, accum_array_T, CONFIG_T::n_out>(gate_ic, cell_act_multp, cell_act_add);

    //-----------Forget gate Calculation
    // Activation
    CONFIG_T::template activation<accum_array_T, accum_array_T, typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(
        cell_act_add, gate_forget);

    // Vector multiplication
    multiply_vectors<accum_array_T, accum_array_T, accum_array_T, CONFIG_T::n_out>(gate_o, gate_forget, h);

OUTPUT_WRITE_LOOP:
    #pragma unroll
    for (int x = (CONFIG_T::n_out - 1); x >= 0; x--) {
        hidden_state_o[x] = h[x];
        cell_state_o[x] = cell_act_add[x];
    }
}

template <class data_T, class res_T, class CONFIG_T>
void lstm(const data_T &data, res_T &res, const typename CONFIG_T::weight_i_t &WI, const typename CONFIG_T::weight_f_t &WF,
          const typename CONFIG_T::weight_c_t &WC, const typename CONFIG_T::weight_o_t &WO,
          const typename CONFIG_T::recurrent_weight_i_t &RWI, const typename CONFIG_T::recurrent_weight_f_t &RWF,
          const typename CONFIG_T::recurrent_weight_c_t &RWC, const typename CONFIG_T::recurrent_weight_o_t &RWO,
          const typename CONFIG_T::bias_i_t &BI, const typename CONFIG_T::bias_f_t &BF,
          const typename CONFIG_T::bias_c_t &BC, const typename CONFIG_T::bias_o_t &BO) {

    // Note:  currently this does not support recurrent bias

    using in_T = array<typename data_T::value_type, CONFIG_T::n_in>;
    using h_T = array<typename res_T::value_type, CONFIG_T::n_out>;

    [[intel::fpga_register]] h_T hidden_state[CONFIG_T::n_timesteps + 1];
    [[intel::fpga_register]] h_T hidden_state_temp;
    [[intel::fpga_register]] h_T cell_state[CONFIG_T::n_timesteps + 1];
    [[intel::fpga_register]] h_T cell_state_temp;
    [[intel::fpga_register]] h_T h;
    [[intel::fpga_register]] h_T c;
    [[intel::fpga_register]] in_T in;

// Set initially hidden state (output) to zero
INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[0][x] = 0;
        cell_state[0][x] = 0;
    }

    // Input dimension
    [[intel::disable_loop_pipelining]] for (int i = 0; i < CONFIG_T::n_timesteps; i++) {
        // Data at current time step
        for (int x = 0; x < CONFIG_T::n_in; x++) {
            in[x] = data[x + i * CONFIG_T::n_in];
        }

        // Hidden state at current time step
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state_temp[x] = hidden_state[i][x];
            cell_state_temp[x] = cell_state[i][x];
        }

        // Do LSTM
        lstm_cell<in_T, h_T, CONFIG_T>(in, hidden_state_temp, h, cell_state_temp, c, WI, WF, WC, WO, RWI, RWF, RWC, RWO, BI,
                                       BF, BC, BO);

        // Write result
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state[i + 1][x] = h[x];
            cell_state[i + 1][x] = c[x];
        }
    }

    if (CONFIG_T::return_sequences == 0) {
        // Output when return_sequences is false
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            res[x] = hidden_state[CONFIG_T::n_timesteps][x];
        }
    } else {
        // Output when return_sequences is true
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_timesteps; x++) {
            for (int h = 0; h < CONFIG_T::n_out; h++) {
                res[x * CONFIG_T::n_out + h] = hidden_state[x + 1][h];
            }
        }
    }
}

template <class data_T, class h_T, class hc_T, class res_T, class CONFIG_T>
void lstm_init_state(const data_T &data, const h_T &hidden_state_initial, const hc_T &cell_state_initial, res_T &res,
                     const typename CONFIG_T::weight_i_t &WI, const typename CONFIG_T::weight_f_t &WF,
                     const typename CONFIG_T::weight_c_t &WC, const typename CONFIG_T::weight_o_t &WO,
                     const typename CONFIG_T::recurrent_weight_i_t &RWI, const typename CONFIG_T::recurrent_weight_f_t &RWF,
                     const typename CONFIG_T::recurrent_weight_c_t &RWC, const typename CONFIG_T::recurrent_weight_o_t &RWO,
                     const typename CONFIG_T::bias_i_t &BI, const typename CONFIG_T::bias_f_t &BF,
                     const typename CONFIG_T::bias_c_t &BC, const typename CONFIG_T::bias_o_t &BO) {

    // Note:  currently this does not support recurrent bias

    using in_T = array<typename data_T::value_type, CONFIG_T::n_in>;

    [[intel::fpga_register]] h_T hidden_state[CONFIG_T::n_timesteps + 1];
    [[intel::fpga_register]] h_T hidden_state_temp;
    [[intel::fpga_register]] h_T cell_state[CONFIG_T::n_timesteps + 1];
    [[intel::fpga_register]] h_T cell_state_temp; // should this be updated to a differnt type
    [[intel::fpga_register]] h_T h;
    [[intel::fpga_register]] h_T c;
    [[intel::fpga_register]] in_T in;

// Set initially hidden state (output) to zero
INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[0][x] = hidden_state_initial[x];
        cell_state[0][x] = cell_state_initial[x];
    }

    // Input dimension
    [[intel::disable_loop_pipelining]] for (int i = 0; i < CONFIG_T::n_timesteps; i++) {
        // Data at current time step
        for (int x = 0; x < CONFIG_T::n_in; x++) {
            in[x] = data[x + i * CONFIG_T::n_in];
        }

        // Hidden state at current time step
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state_temp[x] = hidden_state[i][x];
            cell_state_temp[x] = cell_state[i][x];
        }

        // Do LSTM
        lstm_cell<in_T, h_T, CONFIG_T>(in, hidden_state_temp, h, cell_state_temp, c, WI, WF, WC, WO, RWI, RWF, RWC, RWO, BI,
                                       BF, BC, BO);

        // Write result
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            hidden_state[i + 1][x] = h[x];
            cell_state[i + 1][x] = c[x];
        }
    }

    if (CONFIG_T::return_sequences == 0) {
        // Output when return_sequences is false
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
            res[x] = hidden_state[CONFIG_T::n_timesteps][x];
        }
    } else {
        // Output when return_sequences is true
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_timesteps; x++) {
            for (int h = 0; h < CONFIG_T::n_out; h++) {
                res[x * CONFIG_T::n_out + h] = hidden_state[x + 1][h];
            }
        }
    }
}

} // namespace nnet

#endif
