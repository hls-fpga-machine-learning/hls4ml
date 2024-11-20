#ifndef NNET_MERGE_STREAM_H_
#define NNET_MERGE_STREAM_H_

namespace nnet {

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void add_stream() {
    // both inputs are the same size
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};

AddLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {
        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();

        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    AddPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data1[j] + in_data2[j]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void subtract_stream() {
    // both inputs are the same size
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};

SubtractLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {
        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();

        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    SubtractPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data1[j] - in_data2[j]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void multiply_stream() {
    // both inputs are the same size
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};

MultLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {
        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();

        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    MultPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data1[j] * in_data2[j]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void average_stream() {
    // both inputs are the same size
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};

AvgLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {
        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();

        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    AvgPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(
                (in_data1[j] + in_data2[j]) / (typename ExtractPipeType<res_pipe>::value_type::value_type)2
            );
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void maximum_stream() {
    // both inputs are the same size
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};

MaxLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {
        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();

        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    MaxPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(
                (in_data1[j] > in_data2[j]) ? in_data1[j] : in_data2[j]
            );
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void minimum_stream() {
    // both inputs are the same size
    constexpr auto inputSize = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto outputSize = std::tuple_size<typename ExtractPipeType<res_pipe>::value_type>{};

MinLoop:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem / inputSize; i++) {
        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();

        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    MinPack:
        #pragma unroll
        for (int j = 0; j < outputSize; j++) {
            out_data[j] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(
                (in_data1[j] < in_data2[j]) ? in_data1[j] : in_data2[j]
            );
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate1d_stream() {
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};

    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

ConcatLoop1:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem1_0 / input2Size; i++) {
        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
    ConcatPack1:
        #pragma unroll
        for (int j = 0; j < input1Size; j++) {
            out_data[j + (i * input1Size)] =
                static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data1[j]);
        }
    }

ConcatLoop2:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem2_0 / input2Size; i++) {
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
    ConcatPack2:
        #pragma unroll
        for (int j = 0; j < input2Size; j++) {
            out_data[j + (i * input2Size) + (CONFIG_T::n_elem1_0)] =
                static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data2[j]);
        }
    }
    res_pipe::write(out_data);
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate2d_0_stream() {
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};

ConcatLoopHeight1:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {

        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    ConcatPackInput1:
        #pragma unroll
        for (int k = 0; k < input1Size; k++) {
            out_data[k] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data1[k]);
        }

        res_pipe::write(out_data);
    }

ConcatLoopHeight2:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    ConcatPackInput2:
        #pragma unroll
        for (int k = 0; k < input2Size; k++) {
            out_data[k] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data2[k]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate2d_1_stream() {
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};

ConcatLoopHeight:
    [[intel::initiation_interval(1)]] for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
        [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
        [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

    ConcatPackInput1:
        #pragma unroll
        for (int k = 0; k < input1Size; k++) {
            out_data[k] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data1[k]);
        }

    ConcatPackInput2:
        #pragma unroll
        for (int k = 0; k < input2Size; k++) {
            out_data[input1Size + k] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data2[k]);
        }

        res_pipe::write(out_data);
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate2d_stream() {
    if (CONFIG_T::axis == 2 || CONFIG_T::axis == -1) {
        concatenate2d_1_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    } else {
        concatenate2d_0_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_0_stream() {
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};

ConcatLoopHeight1:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth1:
        [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {

            [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;
        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1Size; k++) {
                out_data[k] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data1[k]);
            }

            res_pipe::write(out_data);
        }
    }

ConcatLoopHeight2:
    for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
    ConcatLoopWidth2:
        [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {

            [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2Size; k++) {
                out_data[k] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data2[k]);
            }

            res_pipe::write(out_data);
        }
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_1_stream() {
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};

ConcatLoopHeight:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth1:
        [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {

            [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1Size; k++) {
                out_data[k] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data1[k]);
            }

            res_pipe::write(out_data);
        }
    ConcatLoopWidth2:
        [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {

            [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2Size; k++) {
                out_data[k] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data2[k]);
            }

            res_pipe::write(out_data);
        }
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_2_stream() {
    constexpr auto input1Size = std::tuple_size<typename ExtractPipeType<input1_pipe>::value_type>{};
    constexpr auto input2Size = std::tuple_size<typename ExtractPipeType<input2_pipe>::value_type>{};

ConcatLoopHeight:
    for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
    ConcatLoopWidth:
        [[intel::initiation_interval(1)]] for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {

            [[intel::fpga_register]] auto in_data1 = input1_pipe::read();
            [[intel::fpga_register]] auto in_data2 = input2_pipe::read();
            [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type out_data;

        ConcatPackInput1:
            #pragma unroll
            for (int k = 0; k < input1Size; k++) {
                out_data[k] = static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data1[k]);
            }

        ConcatPackInput2:
            #pragma unroll
            for (int k = 0; k < input2Size; k++) {
                out_data[input1Size + k] =
                    static_cast<typename ExtractPipeType<res_pipe>::value_type::value_type>(in_data2[k]);
            }

            res_pipe::write(out_data);
        }
    }
}

template <class input1_pipe, class input2_pipe, class res_pipe, typename CONFIG_T> void concatenate3d_stream() {
    if (CONFIG_T::axis == 3 || CONFIG_T::axis == -1) {
        concatenate3d_2_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    } else if (CONFIG_T::axis == 2 || CONFIG_T::axis == -2) {
        concatenate3d_1_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    } else {
        concatenate3d_0_stream<input1_pipe, input2_pipe, res_pipe, CONFIG_T>();
    }
}

} // namespace nnet

#endif
