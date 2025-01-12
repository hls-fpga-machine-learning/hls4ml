#pragma once

#include <cstdint>

template <class T, class U> struct Batch {
    const T *dataIn;
    U *dataOut;
};
