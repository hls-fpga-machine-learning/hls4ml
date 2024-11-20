#ifndef NNET_MERGE_STREAM_H_
#define NNET_MERGE_STREAM_H_

namespace nnet {

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

AddLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        hls_register input1_T in_data1 = data1.read();
        hls_register input2_T in_data2 = data2.read();

        hls_register res_T out_data;

    AddPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = static_cast<typename res_T::value_type>(in_data1[j] + in_data2[j]);
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void subtract(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

SubtractLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        hls_register input1_T in_data1 = data1.read();
        hls_register input2_T in_data2 = data2.read();

        hls_register res_T out_data;

    SubtractPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = static_cast<typename res_T::value_type>(in_data1[j] - in_data2[j]);
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void multiply(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

MultLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        hls_register input1_T in_data1 = data1.read();
        hls_register input2_T in_data2 = data2.read();

        hls_register res_T out_data;

    MultPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = static_cast<typename res_T::value_type>(in_data1[j] * in_data2[j]);
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void average(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

AvgLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        hls_register input1_T in_data1 = data1.read();
        hls_register input2_T in_data2 = data2.read();

        hls_register res_T out_data;

    AvgPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] =
                static_cast<typename res_T::value_type>((in_data1[j] + in_data2[j]) / (typename res_T::value_type)2);
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void maximum(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

MaxLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        hls_register input1_T in_data1 = data1.read();
        hls_register input2_T in_data2 = data2.read();

        hls_register res_T out_data;

    MaxPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = static_cast<typename res_T::value_type>(
                out_data[j] = (in_data1[j] > in_data2[j]) ? in_data1[j] : in_data2[j]
            );
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void minimum(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

MinLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        hls_register input1_T in_data1 = data1.read();
        hls_register input2_T in_data2 = data2.read();

        hls_register res_T out_data;

    MinPack:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = static_cast<typename res_T::value_type>(
                out_data[j] = (in_data1[j] < in_data2[j]) ? in_data1[j] : in_data2[j]
            );
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate1d(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
    hls_register res_T out_data;

ConcatLoop1:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem1_0 / input1_T::size; i++) {
        hls_register input1_T in_data1 = data1.read();
    ConcatPack1:
        #pragma unroll
        for (int j = 0; j < input1_T::size; j++) {
            out_data[j + (i * input1_T::size)] = static_cast<typename res_T::value_type>(in_data1[j]);
        }
    }

ConcatLoop2:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem2_0 / input2_T::size; i++) {
        hls_register input2_T in_data2 = data2.read();
    ConcatPack2:
        #pragma unroll
        for (int j = 0; j < input2_T::size; j++) {
            out_data[j + (i * input2_T::size) + (CONFIG_T::n_elem1_0)] =
                static_cast<typename res_T::value_type>(in_data2[j]);
        }
    }
    res.write(out_data);
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_0(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
ConcatLoopHeight1:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {

        hls_register input1_T in_data1 = data1.read();
        hls_register res_T out_data;

    ConcatPackInput1:
        #pragma unroll
        for (int k = 0; k < input1_T::size; k++) {
            out_data[k] = static_cast<typename res_T::value_type>(in_data1[k]);
        }

        res.write(out_data);
    }

ConcatLoopHeight2:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
        hls_register input2_T in_data2 = data2.read();
        hls_register res_T out_data;

    ConcatPackInput2:
        #pragma unroll
        for (int k = 0; k < input2_T::size; k++) {
            out_data[k] = static_cast<typename res_T::value_type>(in_data2[k]);
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_1(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
ConcatLoopHeight:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        hls_register input1_T in_data1 = data1.read();
        hls_register input2_T in_data2 = data2.read();
        hls_register res_T out_data;

    ConcatPackInput1:
        #pragma unroll
        for (int k = 0; k < input1_T::size; k++) {
            out_data[k] = static_cast<typename res_T::value_type>(in_data1[k]);
        }

    ConcatPackInput2:
        #pragma unroll
        for (int k = 0; k < input2_T::size; k++) {
            out_data[input1_T::size + k] = static_cast<typename res_T::value_type>(in_data2[k]);
        }

        res.write(out_data);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
    if (CONFIG_T::axis == 2 || CONFIG_T::axis == -1) {
        concatenate2d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate2d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_0(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
ConcatLoopHeight1:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth1:
        #pragma ii 1
        for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {

            hls_register input1_T in_data1 = data1.read();
            hls_register res_T out_data;
        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1_T::size; k++) {
                out_data[k] = static_cast<typename res_T::value_type>(in_data1[k]);
            }

            res.write(out_data);
        }
    }

ConcatLoopHeight2:
    for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
    ConcatLoopWidth2:
        #pragma ii 1
        for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {

            hls_register input2_T in_data2 = data2.read();
            hls_register res_T out_data;

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2_T::size; k++) {
                out_data[k] = static_cast<typename res_T::value_type>(in_data2[k]);
            }

            res.write(out_data);
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_1(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
ConcatLoopHeight:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth1:
        #pragma ii 1
        for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {

            hls_register input1_T in_data1 = data1.read();
            hls_register res_T out_data;

        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1_T::size; k++) {
                out_data[k] = static_cast<typename res_T::value_type>(in_data1[k]);
            }

            res.write(out_data);
        }
    ConcatLoopWidth2:
        #pragma ii 1
        for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {

            hls_register input2_T in_data2 = data2.read();
            hls_register res_T out_data;

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2_T::size; k++) {
                out_data[k] = static_cast<typename res_T::value_type>(in_data2[k]);
            }

            res.write(out_data);
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_2(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
ConcatLoopHeight:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth:
        #pragma ii 1
        for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {

            hls_register input1_T in_data1 = data1.read();
            hls_register input2_T in_data2 = data2.read();
            hls_register res_T out_data;

        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1_T::size; k++) {
                out_data[k] = static_cast<typename res_T::value_type>(in_data1[k]);
            }

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2_T::size; k++) {
                out_data[input1_T::size + k] = static_cast<typename res_T::value_type>(in_data2[k]);
            }

            res.write(out_data);
        }
    }
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d(stream<input1_T> &data1, stream<input2_T> &data2, stream<res_T> &res) {
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
