#ifndef NNET_RECURSIVE_H_
#define NNET_RECURSIVE_H_

#include "hls_stream.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_recr_activations.h"

namespace nnet {

struct lstm_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;

    // Layer Sizes
    static const unsigned n_in = 2;
    static const unsigned n_parts = 20;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned n_4state = 8;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;

    template <class x_T, class y_T, class config_T> using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;
    template <class x_T, class y_T, class config_T> using activation = nnet::activation::relu<x_T, y_T, config_T>;
};
// Long Short term Memory NN (LSTM)
// Resources:
// https://github.com/nicodjimenez/lstm/blob/master/lstm.py
// https://github.com/llSourcell/LSTM_Networks/blob/master/LSTM%20Demo.ipynb
// https://en.wikipedia.org/wiki/Long_short-term_memory
// Notes:
//  - LSTM naming conventions adopted from the above links
//      - s_newstate = activation(U*input + W*state)
//      - h_output   = activation(U*input + W*state)*activation(s_newstate)
//  - If softmax is needed on output, perform *outside* this operations
//  Originall had a version allows for the state in each layer to be saved, moved this to above (this requires are LARGE
//  dense network at the end)
template <class data_T, class res_T, typename CONFIG_T>
void lstm(
    bool reset_state,
    data_T data[CONFIG_T::n_in],
    res_T h_newstate[CONFIG_T::n_state],
    res_T s_newstate[CONFIG_T::n_state],
    typename CONFIG_T::weight_t param[CONFIG_T::n_state * 4 * CONFIG_T::n_in],
    typename CONFIG_T::weight_t param_r[CONFIG_T::n_state * 4 * CONFIG_T::n_state],
    typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
    typename CONFIG_T::bias_t param_br[CONFIG_T::n_state * 4]
) {
    // Initialize the state variable -- will maintain state between function calls

    typename CONFIG_T::accum_t tmpres[CONFIG_T::n_state * 4];
    typename CONFIG_T::accum_t tmpres_state[CONFIG_T::n_state * 4];
    typename CONFIG_T::accum_t tmpres_ifo[CONFIG_T::n_state * 3];   // activated i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t tmpres_c[CONFIG_T::n_state];         // activated c-matrix (keras notation)
    typename CONFIG_T::accum_t inputacc_ifo[CONFIG_T::n_state * 3]; // i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t inputacc_c[CONFIG_T::n_state];       // c-matrix (keras notation)
    typename CONFIG_T::accum_t s_actstate[CONFIG_T::n_state];

    #pragma HLS ARRAY_PARTITION variable=h_newstate   complete
    #pragma HLS ARRAY_PARTITION variable=s_newstate   complete
    #pragma HLS ARRAY_PARTITION variable=tmpres       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_ifo   complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_c     complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_ifo complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_c   complete
    #pragma HLS ARRAY_PARTITION variable=s_actstate   complete

    nnet::dense<data_T, res_T, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense<data_T, res_T, typename CONFIG_T::mult_config2>(h_newstate, tmpres_state, param_r, param_br);

    for (int iacc = 0; iacc < (3 * CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc;
        if (iacc > 2 * CONFIG_T::n_state - 1)
            index = iacc + CONFIG_T::n_state;
        inputacc_ifo[iacc] = tmpres[index] + tmpres_state[index];
    }
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc + CONFIG_T::n_state * 2;
        inputacc_c[iacc] = tmpres[index] + tmpres_state[index];
    }

    CONFIG_T::template activation_recr<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_LSTM>::activation(
        inputacc_ifo, tmpres_ifo
    );

    // Now for the confusion matrix
    CONFIG_T::template activation<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>::activation(
        inputacc_c, tmpres_c
    );

    // Operation: s=g*i+sold*f (update state with buffer to avoid timing issues)
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        s_newstate[iacc] = tmpres_c[iacc] * tmpres_ifo[iacc] + s_newstate[iacc] * tmpres_ifo[iacc + (CONFIG_T::n_state)];
    }
    // Operation: h=act(s)*o
    CONFIG_T::template activation<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>::activation(
        s_newstate, s_actstate
    );

    for (int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        #pragma HLS UNROLL
        h_newstate[iacc] = tmpres_ifo[iacc + 2 * (CONFIG_T::n_state)] * s_actstate[iacc];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void lstm_static(
    bool reset_state,
    data_T data[CONFIG_T::n_in],
    res_T h_newstate[CONFIG_T::n_state],
    res_T s_newstate[CONFIG_T::n_state],
    typename CONFIG_T::weight_t param[CONFIG_T::n_state * 4 * CONFIG_T::n_in],
    typename CONFIG_T::weight_t param_r[CONFIG_T::n_state * 4 * CONFIG_T::n_state],
    typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
    typename CONFIG_T::bias_t param_br[CONFIG_T::n_state * 4]
) {
    static res_T h_state[CONFIG_T::n_state];
    static res_T s_state[CONFIG_T::n_state];
    // Initialize the state variable -- will maintain state between function calls
    typename CONFIG_T::accum_t tmpres[CONFIG_T::n_state * 4];
    typename CONFIG_T::accum_t tmpres_state[CONFIG_T::n_state * 4];
    typename CONFIG_T::accum_t tmpres_ifo[CONFIG_T::n_state * 3];   // activated i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t tmpres_c[CONFIG_T::n_state];         // activated c-matrix (keras notation)
    typename CONFIG_T::accum_t inputacc_ifo[CONFIG_T::n_state * 3]; // i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t inputacc_c[CONFIG_T::n_state];       // c-matrix (keras notation)
    typename CONFIG_T::accum_t s_actstate[CONFIG_T::n_state];

    #pragma HLS ARRAY_PARTITION variable=h_newstate   complete
    #pragma HLS ARRAY_PARTITION variable=s_newstate   complete
    #pragma HLS ARRAY_PARTITION variable=h_state      complete
    #pragma HLS ARRAY_PARTITION variable=s_state      complete
    #pragma HLS ARRAY_PARTITION variable=tmpres       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_ifo   complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_c     complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_ifo complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_c   complete
    #pragma HLS ARRAY_PARTITION variable=s_actstate   complete

    if (reset_state) {
        for (int i_state = 0; i_state < (CONFIG_T::n_state); i_state++) {
            #pragma HLS UNROLL
            s_state[i_state] = 0;
            h_state[i_state] = 0;
        }
    }

    nnet::dense<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(
        h_state, tmpres_state, param_r, param_br
    );

    for (int iacc = 0; iacc < (3 * CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc;
        if (iacc > 2 * CONFIG_T::n_state - 1)
            index = iacc + CONFIG_T::n_state;
        inputacc_ifo[iacc] = tmpres[index] + tmpres_state[index];
    }
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc + CONFIG_T::n_state * 2;
        inputacc_c[iacc] = tmpres[index] + tmpres_state[index];
    }

    CONFIG_T::template activation_recr<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_LSTM>::activation(
        inputacc_ifo, tmpres_ifo
    );

    // Now for the confusion matrix
    CONFIG_T::template activation<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>::activation(
        inputacc_c, tmpres_c
    );

    // Operation: s=g*i+sold*f (update state with buffer to avoid timing issues)
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        s_state[iacc] = tmpres_c[iacc] * tmpres_ifo[iacc] + s_state[iacc] * tmpres_ifo[iacc + (CONFIG_T::n_state)];
        s_newstate[iacc] = s_state[iacc];
    }
    // Operation: h=act(s)*o
    CONFIG_T::template activation<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>::activation(
        s_state, s_actstate
    );

    for (int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        #pragma HLS UNROLL
        h_state[iacc] = tmpres_ifo[iacc + 2 * (CONFIG_T::n_state)] * s_actstate[iacc];
        h_newstate[iacc] = h_state[iacc];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void lstm_stack(
    data_T data[CONFIG_T::n_sequence * CONFIG_T::n_in],
    res_T res[CONFIG_T::n_sequence_out * CONFIG_T::n_state],
    typename CONFIG_T::weight_t param[CONFIG_T::n_state * 4 * CONFIG_T::n_in],
    typename CONFIG_T::weight_t param_r[CONFIG_T::n_state * 4 * CONFIG_T::n_state],
    typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
    typename CONFIG_T::bias_t param_br[CONFIG_T::n_state * 4]
) {

    res_T h_newstate[CONFIG_T::n_state];
    res_T s_newstate[CONFIG_T::n_state];
    data_T data_in[CONFIG_T::n_in];
    bool reset_state = true;

    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    #pragma HLS ARRAY_PARTITION variable=s_newstate complete

    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate[ii] = 0;
        s_newstate[ii] = 0;
    }
    for (int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            #pragma HLS UNROLL
            data_in[j] = data[j + iloop * CONFIG_T::n_in];
        }
        if (CONFIG_T::use_static)
            nnet::lstm_static<data_T, res_T, CONFIG_T>(
                reset_state, data_in, h_newstate, s_newstate, param, param_r, param_b, param_br
            );
        else
            nnet::lstm<data_T, res_T, CONFIG_T>(
                reset_state, data_in, h_newstate, s_newstate, param, param_r, param_b, param_br
            );
        if (CONFIG_T::n_sequence_out > 1)
            for (int i = CONFIG_T::n_state * iloop, j = 0; i < (CONFIG_T::n_state * (iloop + 1)); i++, j++) {
                #pragma HLS UNROLL
                res[i] = h_newstate[j];
            }
        reset_state = false;
    }
    if (CONFIG_T::n_sequence_out == 1)
        for (int i = 0; i < (CONFIG_T::n_state); i++) {
            #pragma HLS UNROLL
            res[i] = h_newstate[i];
        }
}

template <class data_T, class res_T, typename CONFIG_T>
void lstm_stack(
    hls::stream<data_T> &data_stream,
    hls::stream<res_T> &res_stream,
    typename CONFIG_T::weight_t param[CONFIG_T::n_state * 4 * CONFIG_T::n_in],
    typename CONFIG_T::weight_t param_r[CONFIG_T::n_state * 4 * CONFIG_T::n_state],
    typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
    typename CONFIG_T::bias_t param_br[CONFIG_T::n_state * 4]
) {

    typename res_T::value_type h_newstate[CONFIG_T::n_state];
    typename res_T::value_type s_newstate[CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    #pragma HLS ARRAY_PARTITION variable=s_newstate complete

    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate[ii] = 0;
        s_newstate[ii] = 0;
    }

    typename data_T::value_type data_in[CONFIG_T::n_in];
    bool reset_state = true;

DataPropagation:
    for (int i_in = 0; i_in < CONFIG_T::n_sequence * CONFIG_T::n_in / data_T::size; i_in++) {
        if (CONFIG_T::n_sequence * CONFIG_T::n_in / data_T::size > 1) {
            // #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
    DataPack:
        for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data_in[i_pack] = data_pack[i_pack];
        }
        if (CONFIG_T::use_static)
            nnet::lstm_static<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(
                reset_state, data_in, h_newstate, s_newstate, param, param_r, param_b, param_br
            );
        else
            nnet::lstm<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(
                reset_state, data_in, h_newstate, s_newstate, param, param_r, param_b, param_br
            );
        if (CONFIG_T::n_sequence_out > 1) {
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        ResPack_sequences:
            for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = h_newstate[i_pack];
            }
            res_stream.write(res_pack);
        }
        reset_state = false;
    }

    if (CONFIG_T::n_sequence_out == 1) {
        res_T res_pack;
        PRAGMA_DATA_PACK(res_pack)
    ResPack:
        for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = h_newstate[i_pack];
        }
        res_stream.write(res_pack);
    }
}

// Struct for the GRU template

struct gru_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 2;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned n_sequence = 2;
    static const unsigned n_4state = 8;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const bool pytorch_order = false;
    static const unsigned n_zeros = 0;

    template <class x_T, class y_T, class config_T> using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;
    template <class x_T, class y_T, class config_T> using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void gru(
    bool reset_state,
    data_T data[CONFIG_T::n_in],
    res_T h_newstate[CONFIG_T::n_state],
    typename CONFIG_T::weight_t param[CONFIG_T::n_state * 3 * CONFIG_T::n_in], // TODO - Check the layout of the param
                                                                               // weights - refer page in copy!!
    typename CONFIG_T::weight_t param_zr[CONFIG_T::n_state * 3 * CONFIG_T::n_state],
    typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
    typename CONFIG_T::bias_t param_br[CONFIG_T::n_state * 3]
) {
    // Initialize the state variable -- will maintain state between function calls
    typename CONFIG_T::accum_t tmpres[CONFIG_T::n_state * 3];
    typename CONFIG_T::accum_t tmpres_state_zr[CONFIG_T::n_state * 3];
    typename CONFIG_T::accum_t tmpres_state_h[CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres_zr[CONFIG_T::n_state * 2];   // activated i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t tmpres_h[CONFIG_T::n_state];        // activated c-matrix (keras notation)
    typename CONFIG_T::accum_t inputacc_zr[CONFIG_T::n_state * 2]; // i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t inputacc_h[CONFIG_T::n_state];      // c-matrix (keras notation)

    #pragma HLS ARRAY_PARTITION variable=h_newstate      complete
    #pragma HLS ARRAY_PARTITION variable=tmpres          complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_zr complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_h  complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_zr       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_h        complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_zr     complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_h      complete

    nnet::dense<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(
        h_newstate, tmpres_state_zr, param_zr, param_br
    );

    // Adding the individual vectors from the multiplication of tmpres = Wx*x(t); tmpres_state_zr = Wh*h(t-1); tmpres
    // initialized with biases -- DONE
    for (int iacc = 0; iacc < (2 * CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc;
        inputacc_zr[iacc] = tmpres[index] + tmpres_state_zr[index];
    }

    // Activation function Sub layer -- START
    CONFIG_T::template activation_recr<
        typename CONFIG_T::accum_t,
        typename CONFIG_T::weight_t,
        typename CONFIG_T::ACT_CONFIG_GRU>::activation(inputacc_zr, tmpres_zr);

    // Activation function Sub layer -- END

    // Hadamrd product of r(t) = inputacc_zr[2*n_state:n_state] and h(t-1) = h_newstate
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        if (CONFIG_T::pytorch_order)
            tmpres_state_h[iacc] = tmpres_zr[iacc] * tmpres_state_zr[iacc + (2 * CONFIG_T::n_state)];
        else
            tmpres_state_h[iacc] = tmpres_zr[iacc + (CONFIG_T::n_state)] * tmpres_state_zr[iacc + (2 * CONFIG_T::n_state)];
    }

    // Assuming reset_after is false
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc + CONFIG_T::n_state * 2;
        inputacc_h[iacc] = tmpres[index] + tmpres_state_h[iacc];
    }

    // Now run the activation on this guy
    CONFIG_T::template activation<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>::
        activation(inputacc_h, tmpres_h);

    // Mix the stat with the previous state
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        if (CONFIG_T::pytorch_order)
            h_newstate[iacc] = (res_T
            )(tmpres_h[iacc] * (1 - tmpres_zr[iacc + (CONFIG_T::n_state)]) +
              h_newstate[iacc] * tmpres_zr[iacc + (CONFIG_T::n_state)]);
        else
            h_newstate[iacc] = (res_T)(tmpres_h[iacc] * (1 - tmpres_zr[iacc]) + h_newstate[iacc] * tmpres_zr[iacc]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void gru_static(
    bool reset_state,
    data_T data[CONFIG_T::n_in],
    res_T h_newstate[CONFIG_T::n_state],
    typename CONFIG_T::weight_t param[CONFIG_T::n_state * 3 * CONFIG_T::n_in],
    typename CONFIG_T::weight_t param_zr[CONFIG_T::n_state * 3 * CONFIG_T::n_state],
    typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
    typename CONFIG_T::bias_t param_br[CONFIG_T::n_state * 3]
) {
    // Initialize the state variable -- will maintain state between function calls

    static res_T h_state[CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres[CONFIG_T::n_state * 3];
    typename CONFIG_T::accum_t tmpres_state_zr[CONFIG_T::n_state * 3];
    typename CONFIG_T::accum_t tmpres_state_h[CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres_zr[CONFIG_T::n_state * 2];   // activated i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t tmpres_h[CONFIG_T::n_state];        // activated c-matrix (keras notation)
    typename CONFIG_T::accum_t inputacc_zr[CONFIG_T::n_state * 2]; // i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t inputacc_h[CONFIG_T::n_state];      // c-matrix (keras notation)

    #pragma HLS ARRAY_PARTITION variable=h_state         complete
    #pragma HLS ARRAY_PARTITION variable=h_newstate      complete
    #pragma HLS ARRAY_PARTITION variable=tmpres          complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_zr complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_h  complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_zr       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_h        complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_zr     complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_h      complete

    if (reset_state) {
        for (int i_h_state = 0; i_h_state < (CONFIG_T::n_state); i_h_state++) {
            #pragma HLS UNROLL
            h_state[i_h_state] = 0;
        }
    }

    nnet::dense<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(
        h_state, tmpres_state_zr, param_zr, param_br
    );

    // Adding the individual vectors from the multiplication of tmpres = Wx*x(t); tmpres_state_zr = Wh*h(t-1); tmpres
    // initialized with biases -- DONE
    for (int iacc = 0; iacc < (2 * CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc;
        inputacc_zr[iacc] = tmpres[index] + tmpres_state_zr[index];
    }

    // Activation function Sub layer -- START
    CONFIG_T::template activation_recr<
        typename CONFIG_T::accum_t,
        typename CONFIG_T::weight_t,
        typename CONFIG_T::ACT_CONFIG_GRU>::activation(inputacc_zr, tmpres_zr);

    // Activation function Sub layer -- END

    // Hadamrd product of r(t) = inputacc_zr[2*n_state:n_state] and h(t-1) = h_newstate
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        if (CONFIG_T::pytorch_order)
            tmpres_state_h[iacc] = tmpres_zr[iacc] * tmpres_state_zr[iacc + (2 * CONFIG_T::n_state)];
        else
            tmpres_state_h[iacc] = tmpres_zr[iacc + (CONFIG_T::n_state)] * tmpres_state_zr[iacc + (2 * CONFIG_T::n_state)];
    }

    // Assuming reset_after is false
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc + CONFIG_T::n_state * 2;
        inputacc_h[iacc] = tmpres[index] + tmpres_state_h[iacc];
    }

    // Now run the activation on this guy
    CONFIG_T::template activation<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>::
        activation(inputacc_h, tmpres_h);

    // Mix the stat with the previous state
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        if (CONFIG_T::pytorch_order)
            h_state[iacc] = (res_T
            )(tmpres_h[iacc] * (1 - tmpres_zr[iacc + (CONFIG_T::n_state)]) +
              h_state[iacc] * tmpres_zr[iacc + (CONFIG_T::n_state)]);
        else
            h_state[iacc] = (res_T)(tmpres_h[iacc] * (1 - tmpres_zr[iacc]) + h_state[iacc] * tmpres_zr[iacc]);
        h_newstate[iacc] = h_state[iacc];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void gru_stack(
    data_T data[CONFIG_T::n_sequence * CONFIG_T::n_in],
    res_T res[CONFIG_T::n_sequence_out * CONFIG_T::n_state],
    typename CONFIG_T::weight_t param[CONFIG_T::n_state * 3 * CONFIG_T::n_in],
    typename CONFIG_T::weight_t param_zr[CONFIG_T::n_state * 3 * CONFIG_T::n_state],
    typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
    typename CONFIG_T::bias_t param_br[CONFIG_T::n_state * 3]
) {

    res_T h_state[CONFIG_T::n_state];
    data_T data_in[CONFIG_T::n_in];
    bool reset_state = true;

    #pragma HLS ARRAY_PARTITION variable=h_state complete
    #pragma HLS ARRAY_PARTITION variable=data_in complete

    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_state[ii] = 0;
    }
    for (int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            #pragma HLS UNROLL
            data_in[j] = data[j + iloop * CONFIG_T::n_in];
        }
        if (CONFIG_T::use_static)
            nnet::gru_static<data_T, res_T, CONFIG_T>(reset_state, data_in, h_state, param, param_zr, param_b, param_br);
        else
            nnet::gru<data_T, res_T, CONFIG_T>(reset_state, data_in, h_state, param, param_zr, param_b, param_br);
        if (CONFIG_T::n_sequence_out > 1)
            for (int i = CONFIG_T::n_state * iloop, j = 0; i < (CONFIG_T::n_state * (iloop + 1)); i++, j++) {
                #pragma HLS UNROLL
                res[i] = h_state[j];
            }
        reset_state = false;
    }
    if (CONFIG_T::n_sequence_out == 1)
        for (int i = 0; i < (CONFIG_T::n_state); i++) {
            #pragma HLS UNROLL
            res[i] = h_state[i];
        }
}

template <class data_T, class res_T, typename CONFIG_T>
void gru_stack(
    hls::stream<data_T> &data_stream,
    hls::stream<res_T> &res_stream,
    typename CONFIG_T::weight_t param[CONFIG_T::n_state * 3 * CONFIG_T::n_in],
    typename CONFIG_T::weight_t param_zr[CONFIG_T::n_state * 3 * CONFIG_T::n_state],
    typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
    typename CONFIG_T::bias_t param_br[CONFIG_T::n_state * 3]
) {

    typename res_T::value_type h_newstate[CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate[ii] = 0;
    }

    typename data_T::value_type data_in[CONFIG_T::n_in];
    bool reset_state = true;

DataPropagation:
    for (int i_in = 0; i_in < CONFIG_T::n_sequence * CONFIG_T::n_in / data_T::size; i_in++) {
        if (CONFIG_T::n_sequence * CONFIG_T::n_in / data_T::size > 1) {
            // #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
    DataPack:
        for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data_in[i_pack] = data_pack[i_pack];
        }
        if (CONFIG_T::use_static)
            nnet::gru_static<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(
                reset_state, data_in, h_newstate, param, param_zr, param_b, param_br
            );
        else
            nnet::gru<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(
                reset_state, data_in, h_newstate, param, param_zr, param_b, param_br
            );
        if (CONFIG_T::n_sequence_out > 1) {
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        ResPack_sequences:
            for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = h_newstate[i_pack];
            }
            res_stream.write(res_pack);
        }
        reset_state = false;
    }

    if (CONFIG_T::n_sequence_out == 1) {
        res_T res_pack;
        PRAGMA_DATA_PACK(res_pack)
    ResPack:
        for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = h_newstate[i_pack];
        }
        res_stream.write(res_pack);
    }
}

} // namespace nnet

#endif
