#ifndef NNET_MHT_H_
#define NNET_MHT_H_

#include "hls_stream.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_mult.h"
#include <math.h>

namespace nnet {

struct multiheadattention_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;
    typedef ap_fixed<16, 8> multi_t;

    // Layer Sizes
    static const unsigned num_heads = 10;
    static const unsigned head_dim_key = 10;
    static const unsigned head_dim_value = 10;
    static const unsigned feature_dim = 20;
    static const unsigned seq_len = 500;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned strategy = latency;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <int PackSize, class data_T> struct datapack { data_T data[PackSize]; };

template <class data_T, int size> void read_stream_array(hls::stream<data_T> data_in[size], data_T out[size]) {
    for (int k = 0; k < size; ++k) {
        #pragma HLS UNROLL
        out[k] = data_in[k].read();
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void matrixmul_transpose(hls::stream<datapack<CONFIG_T::head_dim_key, data_T>> &Q,
                         hls::stream<datapack<CONFIG_T::head_dim_key, data_T>> &K,
                         res_T QK[CONFIG_T::seq_len][CONFIG_T::seq_len]) // seq_Q, seq_K
{
    const data_T dk = 1.0 / sqrt(CONFIG_T::head_dim_key);
    data_T QK_1;
    typename CONFIG_T::accum_t QKij;
    data_T Qi[CONFIG_T::head_dim_key];
    data_T Product[CONFIG_T::seq_len]; // seq_Q, seq_K
    res_T qk_smout[CONFIG_T::seq_len];
    data_T krow[CONFIG_T::seq_len * CONFIG_T::head_dim_key];
    #pragma HLS ARRAY_PARTITION variable=Qi complete
    #pragma HLS ARRAY_PARTITION variable=Product complete
    #pragma HLS ARRAY_PARTITION variable=qk_smout complete
    #pragma HLS ARRAY_PARTITION variable=QK complete dim=2
    #pragma HLS ARRAY_PARTITION variable=krow complete

    datapack<CONFIG_T::head_dim_key, data_T> datak_pack, dataq_pack;
    #pragma HLS DATA_PACK variable=Q
    #pragma HLS DATA_PACK variable=K
    #pragma HLS DATA_PACK variable=datak_pack
    #pragma HLS DATA_PACK variable=dataq_pack

    // int multiplier_limit = ceil(float(CONFIG_T::seq_len * CONFIG_T::head_dim_key) / float(CONFIG_T::reuse_factor));
    // CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::limit(multiplier_limit);

prep_k:
    for (int i = 0; i < CONFIG_T::seq_len; ++i) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        datak_pack = K.read();
        for (int j = 0; j < CONFIG_T::head_dim_key; ++j) {
            #pragma HLS UNROLL
            krow[i * CONFIG_T::head_dim_key + j] = datak_pack.data[j];
        }
    }

row:
    for (int i = 0; i < CONFIG_T::seq_len; ++i) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        dataq_pack = Q.read();

    q:
        for (int q_i = 0; q_i < CONFIG_T::head_dim_key; ++q_i) {
            #pragma HLS UNROLL
            Qi[q_i] = dataq_pack.data[q_i];
        }
    col:
        for (int j = 0; j < CONFIG_T::seq_len; ++j) {
            QKij = 0;
        product:
            for (int k = 0; k < CONFIG_T::head_dim_key; ++k) {
                QK_1 = CONFIG_T::template product<data_T, data_T>::product(Qi[k], krow[j * CONFIG_T::head_dim_key + k]);
                QKij += QK_1;
            }
            Product[j] = QKij * dk;
        }
        softmax<data_T, res_T, typename CONFIG_T::softmax_config1>(Product, qk_smout);
        for (int n = 0; n < CONFIG_T::seq_len; ++n) {
            #pragma HLS UNROLL
            QK[i][n] = qk_smout[n];
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void matrixmul(data_T QK[CONFIG_T::seq_len][CONFIG_T::seq_len], hls::stream<datapack<CONFIG_T::head_dim_key, data_T>> &V,
               hls::stream<res_T> S[CONFIG_T::head_dim_value]) // S: attention score
{
    #pragma HLS DATA_PACK variable=V
    #pragma HLS ARRAY_PARTITION variable=QK complete dim=2
    #pragma HLS ARRAY_PARTITION variable=S complete dim=1

    datapack<CONFIG_T::head_dim_key, data_T> datav_pack;
    #pragma HLS DATA_PACK variable=datav_pack

    // int multiplier_limit = ceil(float(CONFIG_T::seq_len * CONFIG_T::head_dim_value) / float(CONFIG_T::reuse_factor));
    // CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::limit(multiplier_limit);

    data_T dataV[CONFIG_T::seq_len * CONFIG_T::head_dim_value];
    #pragma HLS ARRAY_PARTITION variable = dataV complete dim = 1

    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        datav_pack = V.read();
        for (int i = 0; i < CONFIG_T::head_dim_value; ++i) {
            #pragma HLS UNROLL
            dataV[CONFIG_T::seq_len * i + j] = datav_pack.data[i];
        }
    }

    data_T Sij, S_1;
    data_T QKi[CONFIG_T::seq_len];
#pragma HLS ARRAY_Partition variable=QKi complete
row:
    for (int i = 0; i < CONFIG_T::seq_len; ++i) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    qk:
        for (int q_i = 0; q_i < CONFIG_T::seq_len; ++q_i) {
            #pragma HLS UNROLL
            QKi[q_i] = QK[i][q_i];
        }
    col:
        for (int j = 0; j < CONFIG_T::head_dim_value; ++j) {
            Sij = 0;
        product:
            for (int k = 0; k < CONFIG_T::seq_len; ++k) {
                S_1 = CONFIG_T::template product<data_T, data_T>::product(QKi[k], dataV[j * CONFIG_T::seq_len + k]);
                Sij += S_1;
            }
            S[j].write(Sij);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void lin_projection(hls::stream<data_T> data_q[CONFIG_T::feature_dim], hls::stream<data_T> data_vk[CONFIG_T::feature_dim],
                    hls::stream<datapack<CONFIG_T::head_dim_key, res_T>> &k_proj,
                    hls::stream<datapack<CONFIG_T::head_dim_key, res_T>> &q_proj,
                    hls::stream<datapack<CONFIG_T::head_dim_value, res_T>> &v_proj,
                    typename CONFIG_T::weight_t key_weight[CONFIG_T::feature_dim * CONFIG_T::head_dim_key],
                    typename CONFIG_T::bias_t key_bias[CONFIG_T::head_dim_key],
                    typename CONFIG_T::weight_t query_weight[CONFIG_T::feature_dim * CONFIG_T::head_dim_key],
                    typename CONFIG_T::bias_t query_bias[CONFIG_T::head_dim_key],
                    typename CONFIG_T::weight_t value_weight[CONFIG_T::feature_dim * CONFIG_T::head_dim_value],
                    typename CONFIG_T::bias_t value_bias[CONFIG_T::head_dim_value]) {
    #pragma HLS DATA_PACK variable=k_proj
    #pragma HLS DATA_PACK variable=q_proj
    #pragma HLS DATA_PACK variable=v_proj

    #pragma HLS ARRAY_PARTITION variable=data_q complete dim=1
    #pragma HLS ARRAY_PARTITION variable=data_vk complete dim=1

k_h:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        #pragma HLS PIPELINE

        data_T proj_k[CONFIG_T::head_dim_key];
        data_T proj_q[CONFIG_T::head_dim_key];
        data_T proj_v[CONFIG_T::head_dim_value];
        data_T in_q[CONFIG_T::feature_dim];
        data_T in_v[CONFIG_T::feature_dim];
        #pragma HLS ARRAY_PARTITION variable=proj_k complete dim=1
        #pragma HLS ARRAY_PARTITION variable=proj_q complete dim=1
        #pragma HLS ARRAY_PARTITION variable=proj_v complete dim=1
        #pragma HLS ARRAY_PARTITION variable=in_q complete dim=1
        #pragma HLS ARRAY_PARTITION variable=in_v complete dim=1

        datapack<CONFIG_T::head_dim_key, res_T> proj_k_pack;
        datapack<CONFIG_T::head_dim_key, res_T> proj_q_pack;
        datapack<CONFIG_T::head_dim_value, res_T> proj_v_pack;
        #pragma HLS DATA_PACK variable=proj_k_pack
        #pragma HLS DATA_PACK variable=proj_q_pack
        #pragma HLS DATA_PACK variable=proj_v_pack

        read_stream_array<data_T, CONFIG_T::feature_dim>(data_q, in_q);
        read_stream_array<data_T, CONFIG_T::feature_dim>(data_vk, in_v);

        dense<data_T, res_T, typename CONFIG_T::config_mult1>(in_v, proj_k_pack.data, key_weight, key_bias);
        dense<data_T, res_T, typename CONFIG_T::config_mult1>(in_q, proj_q_pack.data, query_weight, query_bias);
        dense<data_T, res_T, typename CONFIG_T::config_mult1>(in_v, proj_v_pack.data, value_weight, value_bias);

        k_proj.write(proj_k_pack);
        q_proj.write(proj_q_pack);
        v_proj.write(proj_v_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_out(hls::stream<data_T> data_in[CONFIG_T::num_heads][CONFIG_T::head_dim_value],
               res_T res[CONFIG_T::seq_len * CONFIG_T::feature_dim],
               typename CONFIG_T::weight_t
                   attention_output_weight[CONFIG_T::num_heads * CONFIG_T::head_dim_value * CONFIG_T::feature_dim],
               typename CONFIG_T::bias_t attention_output_bias[CONFIG_T::feature_dim]) {
    data_T mat_res_con[CONFIG_T::num_heads * CONFIG_T::head_dim_value];
    res_T dense_out[CONFIG_T::feature_dim];
#pragma HLS ARRAY_PARTITION variable=mat_res_con complete dim=1
#pragma HLS ARRAY_PARTITION variable=dense_out complete dim=1
output_dense:
    for (int k = 0; k < CONFIG_T::seq_len; ++k) {

        #pragma HLS PIPELINE
        for (int i = 0; i < CONFIG_T::num_heads; ++i) {
            #pragma HLS UNROLL
            for (int j = 0; j < CONFIG_T::head_dim_value; ++j) {
                #pragma HLS UNROLL
                mat_res_con[CONFIG_T::head_dim_value * i + j] = data_in[i][j].read();
            }
        }
        dense<data_T, res_T, typename CONFIG_T::config_mult2>(mat_res_con, dense_out, attention_output_weight,
                                                              attention_output_bias);
        for (int i = 0; i < CONFIG_T::feature_dim; ++i) {
            #pragma HLS UNROLL
            res[CONFIG_T::feature_dim * k + i] = dense_out[i];
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void data_prep(data_T data[CONFIG_T::seq_len * CONFIG_T::feature_dim], hls::stream<data_T> d[CONFIG_T::feature_dim]) {
    #pragma HLS ARRAY_PARTITION variable=d complete dim=1
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        for (int k = 0; k < CONFIG_T::feature_dim; ++k) {
            #pragma HLS UNROLL
            d[k].write(data[j * CONFIG_T::feature_dim + k]);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void multiheadattention(
    data_T data_q[CONFIG_T::seq_len * CONFIG_T::feature_dim], data_T data_vk[CONFIG_T::seq_len * CONFIG_T::feature_dim],
    res_T res[CONFIG_T::seq_len * CONFIG_T::feature_dim],
    typename CONFIG_T::weight_t attention_output_weight[CONFIG_T::num_heads * CONFIG_T::head_dim_value *
                                                        CONFIG_T::feature_dim], // num_heads,head_size_v,dim
    typename CONFIG_T::bias_t attention_output_bias[CONFIG_T::feature_dim],
    typename CONFIG_T::weight_t
        key_weight[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_key], // n_head,dim,head_dim
    typename CONFIG_T::bias_t key_bias[CONFIG_T::num_heads * CONFIG_T::head_dim_key],
    typename CONFIG_T::weight_t
        query_weight[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_key], // same shape as key
    typename CONFIG_T::bias_t query_bias[CONFIG_T::num_heads * CONFIG_T::head_dim_key],
    typename CONFIG_T::weight_t value_weight[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_value],
    typename CONFIG_T::bias_t value_bias[CONFIG_T::num_heads * CONFIG_T::head_dim_value]) {
    hls::stream<data_T> d_value[CONFIG_T::num_heads][CONFIG_T::feature_dim];
    hls::stream<data_T> d_query[CONFIG_T::num_heads][CONFIG_T::feature_dim];
    hls::stream<datapack<CONFIG_T::head_dim_key, res_T>> q_proj[CONFIG_T::num_heads];
    hls::stream<datapack<CONFIG_T::head_dim_key, res_T>> k_proj[CONFIG_T::num_heads];
    hls::stream<datapack<CONFIG_T::head_dim_value, res_T>> v_proj[CONFIG_T::num_heads];
    res_T qk_mul[CONFIG_T::num_heads][CONFIG_T::seq_len][CONFIG_T::seq_len];
    hls::stream<res_T> matr_out[CONFIG_T::num_heads][CONFIG_T::head_dim_value];
    #pragma HLS stream variable=d_value type=fifo depth=CONFIG_T::feature_dim
    #pragma HLS stream variable=d_query type=fifo depth=CONFIG_T::feature_dim
    #pragma HLS stream variable=q_proj type=fifo depth=CONFIG_T::seq_len
    #pragma HLS stream variable=k_proj type=fifo depth=CONFIG_T::seq_len
    #pragma HLS stream variable=v_proj type=fifo depth=CONFIG_T::seq_len
    #pragma HLS stream variable=matr_out type=fifo depth=CONFIG_T::head_dim_value

    #pragma HLS DATAFLOW
    #pragma HLS ARRAY_PARTITION variable=d_query complete dim=1
    #pragma HLS ARRAY_PARTITION variable=v_proj complete dim=1
    #pragma HLS ARRAY_PARTITION variable=q_proj complete dim=1
    #pragma HLS ARRAY_PARTITION variable=k_proj complete dim=1
    #pragma HLS ARRAY_PARTITION variable=qk_mul complete dim=1
    #pragma HLS ARRAY_PARTITION variable=matr_out complete dim=1
prepq:
    for (int i = 0; i < CONFIG_T::num_heads; ++i) {
        #pragma HLS UNROLL
        nnet::data_prep<data_T, res_T, CONFIG_T>(data_q, d_query[i]);
    }
prepvk:
    for (int i = 0; i < CONFIG_T::num_heads; ++i) {
        #pragma HLS UNROLL
        nnet::data_prep<data_T, res_T, CONFIG_T>(data_vk, d_value[i]);
    }

lin_proj:
    for (int i = 0; i < CONFIG_T::num_heads; ++i) {
        #pragma HLS UNROLL
        nnet::lin_projection<data_T, res_T, CONFIG_T>(
            d_query[i], d_value[i], k_proj[i], q_proj[i], v_proj[i],
            key_weight + (CONFIG_T::head_dim_key * CONFIG_T::feature_dim * i), key_bias + (CONFIG_T::head_dim_key * i),
            query_weight + (CONFIG_T::head_dim_key * CONFIG_T::feature_dim * i), query_bias + (CONFIG_T::head_dim_key * i),
            value_weight + (CONFIG_T::head_dim_value * CONFIG_T::feature_dim * i),
            value_bias + (CONFIG_T::head_dim_value * i));
    }

maxtrixmul1:
    for (int i = 0; i < CONFIG_T::num_heads; ++i) {
        #pragma HLS UNROLL
        nnet::matrixmul_transpose<res_T, res_T, CONFIG_T>(q_proj[i], k_proj[i], qk_mul[i]);
    }

maxtrixmul2:
    for (int i = 0; i < CONFIG_T::num_heads; ++i) {
        #pragma HLS UNROLL
        nnet::matrixmul<res_T, res_T, CONFIG_T>(qk_mul[i], v_proj[i], matr_out[i]); // stream
    }

    nnet::dense_out<res_T, res_T, CONFIG_T>(matr_out, res, attention_output_weight, attention_output_bias);
}
} // namespace nnet

#endif
