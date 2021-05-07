#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_activation.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<18,8> accum_default_t;
typedef ap_fixed<18,8> weight_default_t;
typedef ap_fixed<18,8> bias_default_t;
typedef ap_fixed<18,8> input_t;
typedef ap_fixed<18,8> result_t;
#define IN_HEIGHT 8
#define IN_WIDTH 8
#define N_CHAN 1
#define FILT_HEIGHT 3
#define FILT_WIDTH 3
#define N_FILT 2
#define STRIDE_HEIGHT 1
#define STRIDE_WIDTH 1
#define PAD_LEFT 1
#define PAD_RIGHT 1
#define PAD_TOP 1
#define PAD_BOTTOM 1
#define OUT_HEIGHT 8
#define OUT_WIDTH 8
#define N_OUTPUTS 10

//hls-fpga-machine-learning insert layer-precision
//typedef ap_fixed<18,8> conv1_t;

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::conv2d_config {

    // Internal data type definitions
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;

    // Convolutional parameters
    static const unsigned pad_top = PAD_TOP;
    static const unsigned pad_bottom = PAD_BOTTOM;
    static const unsigned pad_left = PAD_LEFT;
    static const unsigned pad_right = PAD_RIGHT;
    static const unsigned in_height = IN_HEIGHT;
    static const unsigned in_width = IN_WIDTH;
    static const unsigned n_chan = N_CHAN;
    static const unsigned filt_height = FILT_HEIGHT;
    static const unsigned filt_width = FILT_WIDTH;
    static const unsigned n_filt = N_FILT;
    static const unsigned stride_height = STRIDE_HEIGHT;
    static const unsigned stride_width = STRIDE_WIDTH;
    static const unsigned out_height = OUT_HEIGHT;
    static const unsigned out_width = OUT_WIDTH;

    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0; // not used yet                                                                                                                           
};

struct relu_config1 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT*OUT_WIDTH*N_FILT;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct config2 : nnet::layer_config {
    static const unsigned n_in = OUT_HEIGHT*OUT_WIDTH*N_FILT;
    static const unsigned n_out = N_OUTPUTS;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
};
struct softmax_config2 : nnet::activ_config {
    static const unsigned n_in = N_OUTPUTS;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};


#endif 
