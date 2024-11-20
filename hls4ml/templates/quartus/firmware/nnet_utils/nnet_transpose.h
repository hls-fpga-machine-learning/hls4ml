#ifndef NNET_TRANSPOSE_H_
#define NNET_TRANSPOSE_H_

namespace nnet {

struct transpose_config {
    static const unsigned height = 10;
    static const unsigned width = 10;
    static const unsigned depth = 10;
    static constexpr unsigned perm[3] = {2, 0, 1};
};

template <class data_T, class res_T, typename CONFIG_T>
void transpose_2d(data_T data[CONFIG_T::height * CONFIG_T::width], res_T res[CONFIG_T::height * CONFIG_T::width]) {
    for (int i = 0; i < CONFIG_T::height; i++) {
        #pragma unroll
        for (int j = 0; j < CONFIG_T::width; j++) {
            res[j * CONFIG_T::height + i] = static_cast<res_T>(data[i * CONFIG_T::width + j]);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void transpose_3d(
    data_T data[CONFIG_T::depth * CONFIG_T::height * CONFIG_T::width],
    res_T res[CONFIG_T::depth * CONFIG_T::height * CONFIG_T::width]
) {
    static constexpr unsigned dim_data[3] = {CONFIG_T::depth, CONFIG_T::height, CONFIG_T::width};
    static constexpr unsigned dim_res[3] = {
        dim_data[CONFIG_T::perm[0]], dim_data[CONFIG_T::perm[1]], dim_data[CONFIG_T::perm[2]]};

    int index_data[3] = {0}, index_res[3] = {0};

    for (index_data[0] = 0; index_data[0] < dim_data[0]; index_data[0]++) {
        #pragma unroll
        for (index_data[1] = 0; index_data[1] < dim_data[1]; index_data[1]++) {
            #pragma unroll
            for (index_data[2] = 0; index_data[2] < dim_data[2]; index_data[2]++) {
                index_res[0] = index_data[CONFIG_T::perm[0]];
                index_res[1] = index_data[CONFIG_T::perm[1]];
                index_res[2] = index_data[CONFIG_T::perm[2]];

                res[index_res[0] * dim_res[1] * dim_res[2] + index_res[1] * dim_res[2] + index_res[2]] = static_cast<res_T>(
                    data[index_data[0] * dim_data[1] * dim_data[2] + index_data[1] * dim_data[2] + index_data[2]]
                );
            }
        }
    }
}

} // namespace nnet

#endif
