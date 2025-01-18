#ifndef NNET_TRANSPOSE_H_
#define NNET_TRANSPOSE_H_

namespace nnet {

struct transpose_config {
    static constexpr unsigned dims = 0;
    static constexpr unsigned N = 0;

    // Inherited struct should define these
    // static constexpr std::array<unsigned, dims> from_shape;
    // static constexpr std::array<unsigned, dims> to_shape;
    // static constexpr std::array<unsigned, dims> perm;
    // static constexpr std::array<unsigned, dims> perm_strides;
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

template <class data_T, class res_T, typename CONFIG_T> void transpose(const data_T &data, res_T &res) {
    #pragma unroll
    for (int i = 0; i < CONFIG_T::N; i++) {
        int idx = transfer_idx<CONFIG_T>(i);
        res[i] = data[idx];
    }
}

} // namespace nnet

#endif
