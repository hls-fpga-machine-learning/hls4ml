#ifndef NNET_IMAGE_H_
#define NNET_IMAGE_H_

#include "ac_channel.h"
#include "nnet_common.h"
#include <math.h>

namespace nnet {

struct resize_config {
    static const unsigned height = 10;
    static const unsigned width = 10;
    static const unsigned n_chan = 10;
    static const unsigned new_height = 10;
    static const unsigned new_width = 10;
};

template <class data_T, typename CONFIG_T>
void resize_nearest(
    data_T image[CONFIG_T::height * CONFIG_T::width * CONFIG_T::n_chan],
    data_T resized[CONFIG_T::new_height * CONFIG_T::new_width * CONFIG_T::n_chan]
) {
    int y_ratio = (int)((CONFIG_T::height << 16) / CONFIG_T::new_height) + 1;
    int x_ratio = (int)((CONFIG_T::width << 16) / CONFIG_T::new_width) + 1;
    int x2, y2;

    //#pragma HLS PIPELINE

    for (int i = 0; i < CONFIG_T::new_height; i++) {
        for (int j = 0; j < CONFIG_T::new_width; j++) {
            x2 = ((j * x_ratio) >> 16);
            y2 = ((i * y_ratio) >> 16);
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                resized[(i * CONFIG_T::new_width * CONFIG_T::n_chan) + j * CONFIG_T::n_chan + k] =
                    image[(y2 * CONFIG_T::width * CONFIG_T::n_chan) + x2 * CONFIG_T::n_chan + k];
            }
        }
    }
}

} // namespace nnet

#endif
