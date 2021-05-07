//
//    Vivado HLS code for conv2d
//
//    Copyright (C) 2018 Giuseppe Di Guglielmo, Columbia University
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

#ifndef NNET_CONV2D_TF_H_
#define NNET_CONV2D_TF_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet
{

    template<class data_T, class res_T, typename CONFIG_T>
        void extract_feature_patches(
                data_T patches[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan * CONFIG_T::filt_height * CONFIG_T::filt_width],
                data_T in_features[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan])
        {
            // Input features
            const unsigned in_height = CONFIG_T::in_height;
            const unsigned in_width = CONFIG_T::in_width;
            const unsigned in_n_chan = CONFIG_T::n_chan;

            // Weights
            const unsigned filt_height = CONFIG_T::filt_height;
            const unsigned filt_width = CONFIG_T::filt_width;
            const unsigned n_filt = CONFIG_T::n_filt;

            // Padding
            const int pad_h = CONFIG_T::pad_top;
            const int pad_w = CONFIG_T::pad_left;

            // Output features
            const int out_height = in_height;
            const int out_width = in_width;
            const int out_n_chan = in_n_chan * filt_height * filt_width;

            int p_index = 0;
            for (int fhi = 0; fhi < in_height; fhi++)
            {
                for (int fwi = 0; fwi < in_width; fwi++)
                {
                    int offset = (fhi - pad_h) * in_width * in_n_chan + (fwi - pad_w) * in_n_chan;
                    for (int khi = 0; khi < filt_height; khi++)
                    {
                        for (int kwi = 0; kwi < filt_width; kwi++)
                        {
                            for (int ci = 0; ci < in_n_chan; ci++)
                            {
                                int f_index = offset + khi * in_width * in_n_chan + kwi * in_n_chan + ci;
                                bool boundary_condition =
                                    (fhi < pad_h && khi < pad_h)
                                    || (fwi < pad_w && kwi < pad_w)
                                    || (fhi >= (in_height - pad_h) && khi >= (filt_height - pad_h))
                                    || (fwi >= (in_width - pad_w) && kwi >= (filt_width - pad_w));
                                if (boundary_condition)
                                    patches[p_index] = 0;
                                else
                                    patches[p_index] = in_features[f_index];
                                p_index++;
                            }
                        }
                    }
                }
            }
        }

    template<class data_T, class res_T, typename CONFIG_T>
        void multiply_accumulator(
                res_T out_features[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
                data_T feature_patches[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan * CONFIG_T::filt_height*CONFIG_T::filt_width],
                typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt])
        {
            data_T buffer[CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::in_width];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=0

            // Input features
            const unsigned in_height = CONFIG_T::in_height;
            const unsigned in_width = CONFIG_T::in_width;
            const unsigned in_n_chan = CONFIG_T::n_chan;

            // Weights
            const unsigned filt_height = CONFIG_T::filt_height;
            const unsigned filt_width = CONFIG_T::filt_width;
            const unsigned n_filt = CONFIG_T::n_filt;

            // Padding
            const int pad_h = CONFIG_T::pad_top;
            const int pad_w = CONFIG_T::pad_left;

            // Output features
            const int out_height = in_height;
            const int out_width = in_width;
            const int out_n_chan = in_n_chan * filt_height * filt_width;


            // Element-wise multiplication and accumulation between each
            // weights row and and each feature-map patch

            for (int fhi = 0; fhi < out_height; fhi++)
            {
                for (int fwi = 0; fwi < out_width; fwi++)
                {
                    for (int ki = 0; ki < n_filt; ki++)
                    {
                        for (int ci = 0; ci < out_n_chan; ci++)
                        {
                            int w_index = ki * out_n_chan + ci;
                            int p_index = fhi * out_width * out_n_chan + fwi * out_n_chan + ci;
                            buffer[ci] = weights[w_index] * feature_patches[p_index];
                        }
                        /* sum-reduce */
                        for (int ci = 1; ci < out_n_chan; ci++)
                        {
                            buffer[0] += buffer[ci];
                        }

                        int f_index = (fwi + fhi * out_width) * n_filt + ki;

                        out_features[f_index] = buffer[0];
                    }
                }
            }
        }

        template<class data_T, class res_T, typename CONFIG_T>
            void conv_2d_tf(
                    data_T in_features[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                    res_T out_features[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
                    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
                    typename CONFIG_T::bias_t    biases[CONFIG_T::n_filt])
            {
#pragma HLS DATAFLOW

                // Input features
                const unsigned in_height = CONFIG_T::in_height;
                const unsigned in_width = CONFIG_T::in_width;
                const unsigned in_n_chan = CONFIG_T::n_chan;

                // (1) Flatten the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels]

                // (2) Extract image patches from the input matrix to form a virtual matrix of shape [batch, out_height, out_width, filter_height * filter_width * in_channels]
                data_T feature_patches[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan * CONFIG_T::filt_height * CONFIG_T::filt_width];
#pragma HLS ARRAY_PARTITION variable=feature_patches complete dim=0

                extract_feature_patches<data_T, res_T, CONFIG_T>(feature_patches, in_features);

                // (3) For each patch, right-multiply the filter matrix and the image patch vector
                multiply_accumulator<data_T, res_T, CONFIG_T>(out_features, feature_patches, weights);
            }


        }//end namespace

#endif
