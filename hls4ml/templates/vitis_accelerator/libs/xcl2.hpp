/**
 * Copyright (C) 2019-2021 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// OCL_CHECK doesn't work if call has templatized function call
#define OCL_CHECK(error, call)                                                                                              \
    call;                                                                                                                   \
    if (error != CL_SUCCESS) {                                                                                              \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error);                            \
        exit(EXIT_FAILURE);                                                                                                 \
    }

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>
#include <fstream>
#include <iostream>
// When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
// hood
// User ptr is used if and only if it is properly aligned (page aligned). When
// not
// aligned, runtime has no choice but to create its own host side buffer that
// backs
// user ptr. This in turn implies that all operations that move data to and from
// device incur an extra memcpy to move data to/from runtime's own host buffer
// from/to user pointer. So it is recommended to use this allocator if user wish
// to
// Create Buffer/Memory Object with CL_MEM_USE_HOST_PTR to align user buffer to
// the
// page boundary. It will ensure that user buffer will be used when user create
// Buffer/Mem Object with CL_MEM_USE_HOST_PTR.
template <typename T> struct aligned_allocator {
    using value_type = T;

    aligned_allocator() {}

    aligned_allocator(const aligned_allocator &) {}

    template <typename U> aligned_allocator(const aligned_allocator<U> &) {}

    T *allocate(std::size_t num) {
        void *ptr = nullptr;

#if defined(_WINDOWS)
        {
            ptr = _aligned_malloc(num * sizeof(T), 4096);
            if (ptr == nullptr) {
                std::cout << "Failed to allocate memory" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
#else
        {
            if (posix_memalign(&ptr, 4096, num * sizeof(T)))
                throw std::bad_alloc();
        }
#endif
        return reinterpret_cast<T *>(ptr);
    }
    void deallocate(T *p, std::size_t num) {
#if defined(_WINDOWS)
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

namespace xcl {
std::vector<cl::Device> get_xil_devices();
std::vector<cl::Device> get_devices(const std::string &vendor_name);
cl::Device find_device_bdf(const std::vector<cl::Device> &devices, const std::string &bdf);
cl_device_id find_device_bdf_c(cl_device_id *devices, const std::string &bdf, cl_uint dev_count);
std::string convert_size(size_t size);
std::vector<unsigned char> read_binary_file(const std::string &xclbin_file_name);
bool is_emulation();
bool is_hw_emulation();
bool is_xpr_device(const char *device_name);
class P2P {
  public:
    static decltype(&xclGetMemObjectFd) getMemObjectFd;
    static decltype(&xclGetMemObjectFromFd) getMemObjectFromFd;
    static void init(const cl_platform_id &platform) {
        void *bar = clGetExtensionFunctionAddressForPlatform(platform, "xclGetMemObjectFd");
        getMemObjectFd = (decltype(&xclGetMemObjectFd))bar;
        bar = clGetExtensionFunctionAddressForPlatform(platform, "xclGetMemObjectFromFd");
        getMemObjectFromFd = (decltype(&xclGetMemObjectFromFd))bar;
    }
};
class Ext {
  public:
    static decltype(&xclGetComputeUnitInfo) getComputeUnitInfo;
    static void init(const cl_platform_id &platform) {
        void *bar = clGetExtensionFunctionAddressForPlatform(platform, "xclGetComputeUnitInfo");
        getComputeUnitInfo = (decltype(&xclGetComputeUnitInfo))bar;
    }
};
} // namespace xcl
