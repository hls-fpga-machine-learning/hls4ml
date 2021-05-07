#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_dense.h"
#include "nnet_conv.h"
#include "nnet_activation.h"
#include "nnet_graph.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<16,6> accum_default_t;
typedef ap_fixed<16,6> weight_default_t;
typedef ap_fixed<16,6> bias_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> result_t;
#define REUSE 7
#define N_FEATURES 3
#define N_HIDDEN_FEATURES 4
//2x2 example:
//#define N_NODES 4
//#define N_EDGES 4
//3x3 example:
#define N_NODES 9
#define N_EDGES 18
//4x4 example:
//#define N_NODES 16
//#define N_EDGES 48
//5x5 example:
//#define N_NODES 25
//#define N_EDGES 100

//hls-fpga-machine-learning insert layer-config
struct graph_config1 : nnet::graph_config {
  static const unsigned n_node = N_NODES;
  static const unsigned n_edge = N_EDGES;
  static const unsigned n_input_dim = N_FEATURES+N_HIDDEN_FEATURES;
  static const unsigned n_hidden_dim = N_HIDDEN_FEATURES;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
  static const unsigned n_zeros = 0;
};
                                                                                              
struct dense_config1 : nnet::dense_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_FEATURES;
  static const unsigned n_out = N_HIDDEN_FEATURES;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};

struct tanh_config1 : nnet::activ_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
};

struct dense_config2 : nnet::dense_config {
  static const unsigned n_batch = N_EDGES;
  static const unsigned n_in = 2*(N_FEATURES+N_HIDDEN_FEATURES);
  static const unsigned n_out = N_HIDDEN_FEATURES;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};

struct tanh_config2 : nnet::activ_config {
  static const unsigned n_batch = N_EDGES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
};

struct dense_config3 : nnet::dense_config {
  static const unsigned n_batch = N_EDGES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned n_out = 1;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};

struct sigmoid_config1 : nnet::activ_config {
  static const unsigned n_batch = N_EDGES;
  static const unsigned n_in = 1;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
};

struct dense_config4 : nnet::dense_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = 3*(N_FEATURES+N_HIDDEN_FEATURES);
  static const unsigned n_out = N_HIDDEN_FEATURES;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};

struct tanh_config3 : nnet::activ_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
};

struct tanh_config4 : nnet::activ_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
};

struct dense_config5 : nnet::dense_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned n_out = N_HIDDEN_FEATURES;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};

struct tanh_config5 : nnet::activ_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned table_size = 1024;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = REUSE;
};

#endif 
