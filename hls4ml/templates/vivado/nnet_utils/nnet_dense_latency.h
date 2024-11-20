#ifndef NNET_DENSE_LATENCY_H_
#define NNET_DENSE_LATENCY_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_latency(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in * CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    // For parallel inputs:
    //   - completely partition arrays -- target fabric
    //   - if we have an unroll factor, limit number of multipliers
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=mult complete
    #pragma HLS ARRAY_PARTITION variable=acc complete

    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

// Do the matrix-multiply
Product1:
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        cache = data[ii];
    Product2:
        for (int jj = 0; jj < CONFIG_T::n_out; jj++) {
            int index = ii * CONFIG_T::n_out + jj;
            mult[index] = CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(cache, weights[index]);
        }
    }

// Initialize accumulator with input biases
ResetAccum:
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

// Accumulate multiplication result
Accum1:
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
    Accum2:
        for (int jj = 0; jj < CONFIG_T::n_out; jj++) {
            int index = ii * CONFIG_T::n_out + jj;
            acc[jj] += mult[index];
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        // res[ires] = (res_T) (acc[ires]);
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

} // namespace nnet

#endif
