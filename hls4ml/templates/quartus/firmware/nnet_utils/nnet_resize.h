#ifndef NNET_IMAGE_H_
#define NNET_IMAGE_H_

namespace nnet {

struct resize_config {
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;

    static const unsigned out_height = 10;
    static const unsigned out_width = 10;

    static const unsigned n_chan = 10;
};

template <class data_T, typename CONFIG_T>
void resize_nearest(
    data_T image[CONFIG_T::height * CONFIG_T::width * CONFIG_T::n_chan],
    data_T resized[CONFIG_T::new_height * CONFIG_T::new_width * CONFIG_T::n_chan]
) {
    int y_ratio = (int)((CONFIG_T::height << 16) / CONFIG_T::new_height) + 1;
    int x_ratio = (int)((CONFIG_T::width << 16) / CONFIG_T::new_width) + 1;

    for (int i = 0; i < CONFIG_T::new_height; i++) {
        for (int j = 0; j < CONFIG_T::new_width; j++) {
            int x = ((j * x_ratio) >> 16);
            int y = ((i * y_ratio) >> 16);

            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                resized[(i * CONFIG_T::new_width * CONFIG_T::n_chan) + j * CONFIG_T::n_chan + k] =
                    image[(y * CONFIG_T::width * CONFIG_T::n_chan) + x * CONFIG_T::n_chan + k];
            }
        }
    }
}

} // namespace nnet

#endif
