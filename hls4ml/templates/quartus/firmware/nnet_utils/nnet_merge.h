#ifndef NNET_MERGE_H_
#define NNET_MERGE_H_

#include "nnet_mult.h"

namespace nnet {

struct merge_config {
    static const unsigned n_elem = 10;
    static const unsigned reuse_factor = 1;
};

struct dot_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 1;

    static const unsigned reuse_factor = 1;

    typedef float accum_t;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

struct concat_config {
    static const unsigned n_elem1_0 = 10;
    static const unsigned n_elem1_1 = 10;
    static const unsigned n_elem1_2 = 10;
    static const unsigned n_elem2_0 = 10;
    static const unsigned n_elem2_1 = 10;
    static const unsigned n_elem2_2 = 10;

    static const unsigned axis = -1;
};

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem; i++) {
        res[i] = static_cast<res_T>(data1[i] + data2[i]);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void subtract(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem; i++) {
        res[i] = static_cast<res_T>(data1[i] - data2[i]);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void multiply(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem; i++) {
        res[i] = static_cast<res_T>(data1[i] * data2[i]);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void average(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem; i++) {
        res[i] = static_cast<res_T>((data1[i] + data2[i]) / (res_T)2);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void maximum(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem; i++) {
        res[i] = (data1[i] > data2[i]) ? static_cast<res_T>(data1[i]) : static_cast<res_T>(data2[i]);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void minimum(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem; i++) {
        res[i] = (data1[i] < data2[i]) ? static_cast<res_T>(data1[i]) : static_cast<res_T>(data2[i]);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void dot1d(input1_T data1[CONFIG_T::n_in], input2_T data2[CONFIG_T::n_in], res_T res[CONFIG_T::n_out]) {
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in, CONFIG_T::reuse_factor);

    hls_register typename CONFIG_T::accum_t mult[CONFIG_T::n_in];
Product:
    #pragma unroll multiplier_limit
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        mult[i] = CONFIG_T::template product<input1_T, input2_T>::product(data1[i], data2[i]);
    }

    hls_register typename CONFIG_T::accum_t acc = 0;
Accum:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        acc += mult[i];
    }

    res[0] = static_cast<res_T>(acc);
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate1d(
    input1_T data1[CONFIG_T::n_elem1_0],
    input2_T data2[CONFIG_T::n_elem2_0],
    res_T res[CONFIG_T::n_elem1_0 + CONFIG_T::n_elem2_0]
) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        res[i] = static_cast<res_T>(data1[i]);
    }

    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
        res[CONFIG_T::n_elem1_0 + i] = static_cast<res_T>(data2[i]);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_0(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1],
    input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1]
) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1; i++) {
        res[i] = static_cast<res_T>(data1[i]);
    }

    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1; i++) {
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + i] = static_cast<res_T>(data2[i]);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_1(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1],
    input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1]
) {
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        #pragma unroll
        for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            res[i * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) + j] =
                static_cast<res_T>(data1[i * CONFIG_T::n_elem1_1 + j]);
        }

        #pragma unroll
        for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {
            res[i * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) + CONFIG_T::n_elem1_1 + j] =
                static_cast<res_T>(data2[i * CONFIG_T::n_elem2_1 + j]);
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1],
    input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1]
) {
    if (CONFIG_T::axis == 2 || CONFIG_T::axis == -1) {
        concatenate2d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate2d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_0(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2],
    input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 +
            CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2]
) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2; i++) {
        res[i] = static_cast<res_T>(data1[i]);
    }

    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2; i++) {
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + i] = static_cast<res_T>(data2[i]);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_1(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2],
    input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 +
            CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2]
) {
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_elem1_2; k++) {
                int res_idx =
                    i * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) * CONFIG_T::n_elem1_2 + j * CONFIG_T::n_elem1_2 + k;
                int data_idx = i * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + j * CONFIG_T::n_elem1_2 + k;
                res[res_idx] = static_cast<res_T>(data1[data_idx]);
            }
        }

        for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_elem2_2; k++) {
                int res_idx = i * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) * CONFIG_T::n_elem1_2 +
                              (j + CONFIG_T::n_elem1_1) * CONFIG_T::n_elem1_2 + k;
                int data_idx = i * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2 + j * CONFIG_T::n_elem2_2 + k;
                res[res_idx] = static_cast<res_T>(data2[data_idx]);
            }
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_2(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2],
    input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 +
            CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2]
) {
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {

            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_elem1_2; k++) {
                int res_idx = i * CONFIG_T::n_elem1_1 * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2) +
                              j * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2) + k;
                int data_idx = i * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + j * CONFIG_T::n_elem1_2 + k;
                res[res_idx] = static_cast<res_T>(data1[data_idx]);
            }

            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_elem1_2; k++) {
                int res_idx = i * CONFIG_T::n_elem1_1 * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2) +
                              j * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2) + k + CONFIG_T::n_elem1_2;
                int data_idx = i * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2 + j * CONFIG_T::n_elem2_2 + k;
                res[res_idx] = static_cast<res_T>(data2[data_idx]);
            }
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2],
    input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 +
            CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2]
) {
    if (CONFIG_T::axis == 3 || CONFIG_T::axis == -1) {
        concatenate3d_2<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else if (CONFIG_T::axis == 2 || CONFIG_T::axis == -2) {
        concatenate3d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate3d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

} // namespace nnet

#endif
