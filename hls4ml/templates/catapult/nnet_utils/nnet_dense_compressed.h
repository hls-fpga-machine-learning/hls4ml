//
//    hls4ml: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2018 Giuseppe Di Guglielmo
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_COMPRESSED_LAYER_H_
#define NNET_COMPRESSED_LAYER_H_

#include "ac_channel.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include <math.h>

namespace nnet {

template <typename CONFIG_T>
void fill_mult(
    typename CONFIG_T::index_t index, typename CONFIG_T::accum_t mult[CONFIG_T::n_out], typename CONFIG_T::accum_t weight
) {
    for (unsigned k = 0; k < CONFIG_T::n_out; k++) {
        // #pragma HLS UNROLL
        if (k == index)
            mult[k] += weight;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_compressed(
    data_T data[CONFIG_T::n_in],
    res_T res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_nonzeros],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {

    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_nonzeros, CONFIG_T::reuse_factor);

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    //#pragma HLS ARRAY_PARTITION variable=acc    complete
    //#pragma HLS ARRAY_PARTITION variable=biases complete
    //#pragma HLS ARRAY_RESHAPE   variable=weights block factor=multiplier_limit
    // if (CONFIG_T::store_weights_in_bram){
    ////#pragma HLS RESOURCE variable=weights core=ROM_1P_BRAM
    //#pragma HLS data_pack variable=weights struct_level
    //}

InitAccum:
    for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
        acc[i] = (typename CONFIG_T::accum_t)(biases[i]);
    }

    // Do the compressed matrix-multiply
    const int rufactor = CONFIG_T::reuse_factor;
ReuseLoop:
    for (unsigned ir = 0; ir < rufactor; ir++) {
        //#pragma HLS PIPELINE  II=1 rewind

        typename CONFIG_T::accum_t mult[CONFIG_T::n_out];
        //#pragma HLS ARRAY_PARTITION variable=mult complete

    ResetMult:
        for (int imult = 0; imult < CONFIG_T::n_out; imult++) {
            // #pragma HLS UNROLL
            mult[imult] = 0;
        }

    CompressedMultLoop:
        for (unsigned im = 0; im < multiplier_limit; im++) {
            // #pragma HLS UNROLL
            unsigned w = im * rufactor + ir;
            auto row = weights[w].row_index;
            auto col = weights[w].col_index;
            auto weight_cache = weights[w].weight;
            data_T data_cache = data[row];
            // mult[col] += weight_cache * data_cache;
            typename CONFIG_T::accum_t prod =
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data_cache, weight_cache);
            fill_mult<CONFIG_T>(col, mult, prod);
        }

        for (int im = 0; im < CONFIG_T::n_out; im++) {
            acc[im] += mult[im];
        }
    }

// Cast to "res_t" type
ResultLoop:
    for (unsigned i = 0; i < CONFIG_T::n_out; i++) {
        // #pragma HLS UNROLL
        // res[i] = (res_T) (acc[i]);
        res[i] = cast<data_T, res_T, CONFIG_T>(acc[i]);
    }
}

} // namespace nnet

#endif
