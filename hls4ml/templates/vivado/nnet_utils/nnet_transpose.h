#ifndef NNET_PERMUTE_H_
#define NNET_PERMUTE_H_

namespace nnet {

struct transpose_config {
    static const unsigned dims;
    static const unsigned N;
    // vivado/vitis hls can't index constexpr array for some reason
    // and vivado hls don't like template recursion either (vitis is fine)
    // thus this appears to be the only workaround (or overkill it with codegen)
    static const unsigned *const from_shape;
    static const unsigned *const to_shape;
    static const unsigned *const perm;
    static const unsigned *const perm_strides;
};

template <typename CONFIG_T> unsigned transfer_idx(int index) {
    // Given output idx in c-order flat array, return input idx
    int idx = 0;
    for (int i = CONFIG_T::dims - 1; i >= 0; i--) {
        idx += (index % CONFIG_T::to_shape[i]) * CONFIG_T::perm_strides[i];
        index /= CONFIG_T::to_shape[i];
    }
    return idx;
}

template <typename data_T, typename res_T, typename CONFIG_T>
void transpose(const data_T data[CONFIG_T::N], res_T res[CONFIG_T::N]) {
    for (int i = 0; i < CONFIG_T::N; i++) {
        #pragma HLS UNROLL
        int idx = transfer_idx<CONFIG_T>(i);
        res[i] = data[idx];
    }
}

} // namespace nnet

#endif
