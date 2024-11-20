
#ifndef NNET_MERGE_H_
#define NNET_MERGE_H_

#include "ac_channel.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include <math.h>

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
    // Product function to use
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

struct concat_config {
    static const unsigned n_elem1_0 = 10;
    static const unsigned n_elem1_1 = 10;
    static const unsigned n_elem1_2 = 10;
    static const unsigned n_elem2_0 = 10;
    static const unsigned n_elem2_1 = 10;
    static const unsigned n_elem2_2 = 10;

    static const int axis = -1;
};

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    for (int ii = 0; ii < CONFIG_T::n_elem; ii++) {
        res[ii] = data1[ii] + data2[ii];
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void subtract(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    for (int ii = 0; ii < CONFIG_T::n_elem; ii++) {
        res[ii] = data1[ii] - data2[ii];
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void multiply(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    for (int ii = 0; ii < CONFIG_T::n_elem; ii++) {
        res[ii] = data1[ii] * data2[ii];
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void average(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    for (int ii = 0; ii < CONFIG_T::n_elem; ii++) {
        res[ii] = (data1[ii] + data2[ii]) / (res_T)2;
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void maximum(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    for (int ii = 0; ii < CONFIG_T::n_elem; ii++) {
        res[ii] = (data1[ii] > data2[ii]) ? data1[ii] : data2[ii];
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void minimum(input1_T data1[CONFIG_T::n_elem], input2_T data2[CONFIG_T::n_elem], res_T res[CONFIG_T::n_elem]) {
    for (int ii = 0; ii < CONFIG_T::n_elem; ii++) {
        res[ii] = (data1[ii] < data2[ii]) ? data1[ii] : data2[ii];
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void dot1d(input1_T data1[CONFIG_T::n_in], input2_T data2[CONFIG_T::n_in], res_T res[CONFIG_T::n_out]) {
    //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    CONFIG_T::template product<input1_T, input2_T>::limit(multiplier_limit);

    typename CONFIG_T::accum_t mult[CONFIG_T::n_in];
    //#pragma HLS ARRAY_PARTITION variable=mult complete
    typename CONFIG_T::accum_t acc = 0;

Product:
    for (int i_mult = 0; i_mult < CONFIG_T::n_in; i_mult++) {
        // #pragma HLS UNROLL
        mult[i_mult] = CONFIG_T::template product<input1_T, input2_T>::product(data1[i_mult], data2[i_mult]);
    }

Accum:
    for (int i_acc = 0; i_acc < CONFIG_T::n_in; i_acc++) {
        // #pragma HLS UNROLL
        acc += mult[i_acc];
    }

    res[0] = cast<input1_T, res_T, CONFIG_T>(acc);
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate1d(
    input1_T data1[CONFIG_T::n_elem1_0],
    input2_T data2[CONFIG_T::n_elem2_0],
    res_T res[CONFIG_T::n_elem1_0 + CONFIG_T::n_elem2_0]
) {
    for (int ii = 0; ii < CONFIG_T::n_elem1_0; ii++) {
        res[ii] = data1[ii];
    }
    for (int ii = 0; ii < CONFIG_T::n_elem2_0; ii++) {
        res[CONFIG_T::n_elem1_0 + ii] = data2[ii];
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_0(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1],
    input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1]
) {
    for (int ii = 0; ii < CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1; ii++) {
        res[ii] = data1[ii];
    }
    for (int ii = 0; ii < CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1; ii++) {
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + ii] = data2[ii];
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_1(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1],
    input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1]
) {
    for (int ii = 0; ii < CONFIG_T::n_elem1_0; ii++) {
        for (int jj = 0; jj < CONFIG_T::n_elem1_1; jj++) {
            res[ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) + jj] = data1[ii * CONFIG_T::n_elem1_1 + jj];
        }
        for (int jj = 0; jj < CONFIG_T::n_elem2_1; jj++) {
            res[ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) + CONFIG_T::n_elem1_1 + jj] =
                data2[ii * CONFIG_T::n_elem2_1 + jj];
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
    for (int ii = 0; ii < CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2; ii++) {
        res[ii] = data1[ii];
    }
    for (int ii = 0; ii < CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2; ii++) {
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + ii] = data2[ii];
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
    for (int ii = 0; ii < CONFIG_T::n_elem1_0; ii++) {
        for (int jj = 0; jj < CONFIG_T::n_elem1_1; jj++) {
            for (int kk = 0; kk < CONFIG_T::n_elem1_2; kk++) {
                int res_idx =
                    ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) * CONFIG_T::n_elem1_2 + jj * CONFIG_T::n_elem1_2 + kk;
                int data_idx = ii * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + jj * CONFIG_T::n_elem1_2 + kk;
                res[res_idx] = data1[data_idx];
            }
        }
        for (int jj = 0; jj < CONFIG_T::n_elem2_1; jj++) {
            for (int kk = 0; kk < CONFIG_T::n_elem2_2; kk++) {
                int res_idx = ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) * CONFIG_T::n_elem1_2 +
                              (jj + CONFIG_T::n_elem1_1) * CONFIG_T::n_elem1_2 + kk;
                int data_idx = ii * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2 + jj * CONFIG_T::n_elem2_2 + kk;
                res[res_idx] = data2[data_idx];
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
    for (int ii = 0; ii < CONFIG_T::n_elem1_0; ii++) {
        for (int jj = 0; jj < CONFIG_T::n_elem1_1; jj++) {
            for (int kk = 0; kk < CONFIG_T::n_elem1_2; kk++) {
                int res_idx = ii * CONFIG_T::n_elem1_1 * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2) +
                              jj * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2) + kk;
                int data_idx = ii * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + jj * CONFIG_T::n_elem1_2 + kk;
                res[res_idx] = data1[data_idx];
            }
            for (int kk = 0; kk < CONFIG_T::n_elem1_2; kk++) {
                int res_idx = ii * CONFIG_T::n_elem1_1 * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2) +
                              jj * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2) + kk + CONFIG_T::n_elem1_2;
                int data_idx = ii * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2 + jj * CONFIG_T::n_elem2_2 + kk;
                res[res_idx] = data2[data_idx];
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
