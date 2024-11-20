#ifndef NNET_MATH_H_
#define NNET_MATH_H_

#include "hls_math.h"

namespace nnet {

// This header defines the functions that return type different from the input
// For example, hls::sin(x) returns ap_fixed<W-I+2,2>
// By ensuring we return the same type we can avoid casting issues in expressions

template <typename T> T sin(T x) { return (T)hls::sin(x); };

template <typename T> T cos(T x) { return (T)hls::cos(x); };

template <typename T> T asin(T x) { return (T)hls::asin(x); };

template <typename T> T acos(T x) { return (T)hls::acos(x); };

template <typename T> T atan(T x) { return (T)hls::atan(x); };

template <typename T> T atan2(T x, T y) { return (T)hls::atan2(x, y); };

template <class T, int W, int I> void init_sincos_table(T table[1 << (W - I - 3)][2]) {
    unsigned int NTE = 1 << (W - I - 3); // No of table entries
    double step = M_PI / (4 * NTE);      // Interval between angles
    double y = 0;
    // double scaled_angle = 0;

    for (unsigned int i = 0; i < NTE; i++) {
        table[i][0] = std::cos(y);
        table[i][1] = std::sin(y);
        y += step;
        // scaled_angle = y/(2*M_PI);
        // printf("cos(%f) = %23.22f, sin(%f) = %23.22f index = %d, scaled angle = %13.12f \n", y, cos(y), y, sin(y), i,
        // scaled_angle);
    }
}

template <class T> void sincos_lut(const T &input, T output[2]) {

    #pragma HLS INLINE

    // This implementation is based on ac_sincos_lut.h from AC math library

    static bool flag = true;
    if (flag && T::width - T::iwidth > 12) {
#if !defined(__SYNTHESIS__) && defined(SINCOS_LUT_DEBUG)
        std::cout << "FILE : " << __FILE__ << ", LINE : " << __LINE__ << std::endl;
        std::cout << "Warning: The output of sincos_lut will not be accurate" << std::endl;
#endif
        flag = false;
    }
    // Datatype for lookup table entries
    typedef ap_ufixed<T::width, T::iwidth, AP_RND> luttype;
    // Datatype for posinput which is used to handle negative inputs
    typedef ap_ufixed<T::width - T::iwidth, 0> posinputtype;

    typedef ap_uint<9> lutindextype; // 9 bits required for indexing into 512 entry table
    typedef ap_uint<3> octanttype;   // 3 bits required for octant value range of 0 thru 7
    T outputtemp[2];
    lutindextype luTdex = 0;
    posinputtype posinput = input;

    // Initialize the lookup table
#ifdef __SYNTHESIS__
    bool initialized = false;
    luttype sincos[512][2];
#else
    static bool initialized = false;
    static luttype sincos[512][2];
#endif
    if (!initialized) {
        init_sincos_table<luttype, 12, 0>(sincos);
        initialized = true;
    }

    // Leaving this commented out makes the table to to BRAM
    //#pragma HLS ARRAY_PARTITION variable=sincos complete dim=0

    typedef ap_uint<AP_MAX(T::width - T::iwidth - 3, 1)> lutindextype1;
    // Extracting (MSB-3:LSB) bits of scaled input to determine the lookup table index
    lutindextype1 luTdex1 = posinput.range(AP_MAX(T::width - T::iwidth - 3, 1), 0); // Extracting the lookup table index

    if (T::width - T::iwidth >= 4 && T::width - T::iwidth <= 12) {
        luTdex(8, 12 - (T::width - T::iwidth)) = luTdex1; // stride
    }
    // Approximation for the scaled inputs whose number of bits are greater than 12
    else if (T::width - T::iwidth > 12) {
        // Lookup table index for the scaled inputs whose number of bits are greater than 12
        luTdex = luTdex1 / (1 << (AP_MAX(T::width - T::iwidth - 12, 0)));
        if ((luTdex1 % (1 << (AP_MAX(T::width - T::iwidth - 12, 0)))) > (1 << (AP_MAX(T::width - T::iwidth - 13, 0)))) {
            luTdex = luTdex + 1;
        }
        typedef ap_ufixed<
            AP_MAX((AP_MAX(T::width - T::iwidth - 3, 1) + T::width - T::iwidth - 12), 1),
            AP_MAX(T::width - T::iwidth - 3, 1)>
            datatype;
        datatype x = (datatype)luTdex1;
        x = x >> AP_MAX(T::width - T::iwidth - 12, 0);
        if (x > 511.5) {
            luTdex = 511;
        }
        if (luTdex1 <= 1 << (AP_MAX(T::width - T::iwidth - 13, 0)) && luTdex1 != 0) {
            luTdex = 1;
        }
    }

    if (T::width - T::iwidth >= 3) {
        // Getting the octant 0-7 by extracting the first 3 bits from MSB side of scaled input where
        //   octant 0 corresponds to [0-PI/4),
        //   octant 1 corresponds to [PI/4-2PI/4),
        //   octant 2 corresponds to [2PI/4-3PI/4) and so on
        // octanttype octant = posinput.template slc<3>(T::width-T::iwidth-3);
        octanttype octant = posinput(T::width - T::iwidth - 1, T::width - T::iwidth - 3);
        luTdex = (octant[0] == 1) ? (lutindextype)(512 - luTdex) : (lutindextype)(luTdex);
        // imaginary part is sine
        outputtemp[1] = ((octant == 0) | (octant == 3))   ? (T)sincos[luTdex][1]
                        : ((octant == 2) | (octant == 1)) ? (T)sincos[luTdex][0]
                        : ((octant == 7) | (octant == 4)) ? (T)-sincos[luTdex][1]
                                                          : (T)-sincos[luTdex][0];
        // real part is cosine
        outputtemp[0] = ((octant == 6) | (octant == 1))   ? (T)sincos[luTdex][1]
                        : ((octant == 3) | (octant == 4)) ? (T)-sincos[luTdex][0]
                        : ((octant == 2) | (octant == 5)) ? (T)-sincos[luTdex][1]
                                                          : (T)sincos[luTdex][0];
        // Below two are the cases when the output corresponds to + or - (0 or 1) for which there is no entry in the lookup
        // table
        output[1] = ((posinput == 0.125) | (posinput == 0.375))   ? T(0.7071067811865475244008)
                    : ((posinput == 0.625) | (posinput == 0.875)) ? T(-0.7071067811865475244008)
                                                                  : outputtemp[1];
        output[0] = ((posinput == 0.125) | (posinput == 0.875))   ? T(0.7071067811865475244008)
                    : ((posinput == 0.375) | (posinput == 0.625)) ? T(-0.7071067811865475244008)
                                                                  : outputtemp[0];
    }

    if (T::width - T::iwidth <= 2) {
        output[1] = (posinput == 0)      ? (T)0
                    : (posinput == 0.25) ? (T)1
                    : (posinput == 0.5)  ? (T)0
                    : (posinput == 0.75) ? (T)-1
                                         : outputtemp[1];
        output[0] = (posinput == 0)      ? (T)1
                    : (posinput == 0.25) ? (T)0
                    : (posinput == 0.5)  ? (T)-1
                    : (posinput == 0.75) ? (T)0
                                         : outputtemp[0];
    }

#if !defined(__SYNTHESIS__) && defined(SINCOS_LUT_DEBUG)
    std::cout << "FILE : " << __FILE__ << ", LINE : " << __LINE__ << std::endl;
    std::cout << "============AP_FIXED SINCOS======================" << std::endl;
    std::cout << "positive input is   = " << posinput << std::endl;
    std::cout << "lut index is   = " << luTdex << std::endl;
    std::cout << "sin value is    = " << output[1] << std::endl;
    std::cout << "cos value is    = " << output[0] << std::endl;
    std::cout << "=================================================" << std::endl;
#endif
}

template <class T> T sin_lut(const T input) {
    #pragma HLS INLINE
    T sincos_res[2];
    T scaled_input = input * ap_ufixed<16, 0>(0.15915494309); // 1/(2*pi)
    sincos_lut(scaled_input, sincos_res);
    return sincos_res[1];
}

template <class T> T cos_lut(const T input) {
    #pragma HLS INLINE
    T sincos_res[2];
    T scaled_input = input * ap_ufixed<16, 0>(0.15915494309); // 1/(2*pi)
    sincos_lut(scaled_input, sincos_res);
    return sincos_res[0];
}

} // namespace nnet

#endif
