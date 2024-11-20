#ifndef NNET_PADDING_H_
#define NNET_PADDING_H_

namespace nnet {

struct padding1d_config {
    static const unsigned in_width = 10;
    static const unsigned out_width = 10;
    static const unsigned n_chan = 10;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
};

template <class data_T, class res_T, typename CONFIG_T>
void zeropad1d_cl(data_T data[CONFIG_T::n_chan * CONFIG_T::in_width], res_T res[CONFIG_T::n_chan * CONFIG_T::out_width]) {
    for (int i = 0; i < CONFIG_T::pad_left; i++) {
        #pragma unroll
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            *(res++) = 0;
        }
    }

    for (int i = 0; i < CONFIG_T::in_width; i++) {
        #pragma unroll
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            *(res++) = (res_T) * (data++);
        }
    }

    for (int i = 0; i < CONFIG_T::pad_right; i++) {
        #pragma unroll
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            *(res++) = 0;
        }
    }
}

struct padding2d_config {
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;

    static const unsigned out_height = 10;
    static const unsigned out_width = 10;

    static const unsigned n_chan = 10;

    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
};

template <class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl(
    data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
    res_T res[CONFIG_T::n_chan * CONFIG_T::out_height * CONFIG_T::out_width]
) {
    for (int i = 0; i < CONFIG_T::pad_top; i++) {
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                *(res++) = 0;
            }
        }
    }

    for (int i = 0; i < CONFIG_T::in_height; i++) {
        for (int j = 0; j < CONFIG_T::pad_left; j++) {
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                *(res++) = 0;
            }
        }
        for (int j = 0; j < CONFIG_T::in_width; j++) {
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                *(res++) = (res_T) * (data++);
            }
        }
        for (int j = 0; j < CONFIG_T::pad_right; j++) {
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                *(res++) = 0;
            }
        }
    }

    for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                *(res++) = 0;
            }
        }
    }
}

} // namespace nnet

#endif
