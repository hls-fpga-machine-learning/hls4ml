#ifndef KL_LAYER_H_
#define KL_LAYER_H_

#include "nnet_activation.h"
#include "nnet_common.h"
#include <cmath>
#include <cstdlib>

namespace nnet {

struct distance_config {
    // IO size
    static const unsigned n_in = 10;
    static const unsigned n_out = 1;

    // Internal data type definitions
    typedef float accum_t;
    typedef float sum_t;
    typedef ap_fixed<18, 8> exp_table_t;

    // Internal info
    static const unsigned table_size = 1024;
    static constexpr unsigned exp_range = 8;
};

template <typename CONFIG_T, int N_TABLE> void init_klloss_exp_table(typename CONFIG_T::exp_table_t table_out[N_TABLE]) {
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (range -1 to +1)
        float in_val = 2 * CONFIG_T::exp_range * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::exp_table_t real_val = exp_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << " Index: " << ii << std::endl;
        table_out[ii] = real_val;
    }
}
template <class data1_T, class data2_T, class res_T, typename CONFIG_T>
void klloss(data1_T mean[CONFIG_T::n_in], data2_T log_var[CONFIG_T::n_in], res_T res[CONFIG_T::n_out]) {
    #pragma HLS PIPELINE
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_klloss_exp_table<CONFIG_T, CONFIG_T::table_size>(exp_table);
        initialized = true;
    }
    typename CONFIG_T::accum_t kl[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=kl complete
    typename CONFIG_T::accum_t mean_sq[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=mean_sq complete
    typename CONFIG_T::accum_t kl_sum(0);
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
        mean_sq[i] = mean[i] * mean[i];
        kl[i] = data2_T(1.) + log_var[i];
        // std::cout << "Log var: " << log_var[i] << " Result: " << kl[i] << std::endl;
    }
    constexpr unsigned table_scale = (unsigned)(CONFIG_T::table_size / (2 * CONFIG_T::exp_range));
    constexpr unsigned index_scale = (unsigned)(CONFIG_T::exp_range * table_scale);
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
        auto data_round = log_var[i] * table_scale;
        auto index = data_round + index_scale;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        kl[i] -= exp_table[index];
        // std::cout << "Exp var: " << exp_table[index] << " Result: " << kl[i] << " Index: " << index << std::endl;
    }
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
        kl[i] -= mean_sq[i];
    }
    Op_add<typename CONFIG_T::accum_t> op_add;
    kl_sum = reduce<typename CONFIG_T::accum_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::accum_t>>(kl, op_add);
    // std::cout << "KL sum: " << kl_sum << std::endl;
    kl_sum *= typename CONFIG_T::accum_t(1. / CONFIG_T::n_in);
    res[0] = res_T(-0.5) * kl_sum;
}
} // namespace nnet

#endif
