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

#include "xcl2.hpp"
#include <climits>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>
#if defined(_WINDOWS)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace xcl {
std::vector<cl::Device> get_devices(const std::string &vendor_name) {
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    OCL_CHECK(err, err = cl::Platform::get(&platforms));
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++) {
        platform = platforms[i];
        OCL_CHECK(err, std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
        if (!(platformName.compare(vendor_name))) {
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "Error: Failed to find Xilinx platform" << std::endl;
        std::cout << "Found the following platforms : " << std::endl;
        for (size_t j = 0; j < platforms.size(); j++) {
            platform = platforms[j];
            OCL_CHECK(err, std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
            std::cout << "Platform Name: " << platformName.c_str() << std::endl;
        }
        exit(EXIT_FAILURE);
    }
    // Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
    return devices;
}

std::vector<cl::Device> get_xil_devices() { return get_devices("Xilinx"); }

cl::Device find_device_bdf(const std::vector<cl::Device> &devices, const std::string &bdf) {
    char device_bdf[20];
    cl_int err;
    cl::Device device;
    int cnt = 0;
    for (uint32_t i = 0; i < devices.size(); i++) {
        OCL_CHECK(err, err = devices[i].getInfo(CL_DEVICE_PCIE_BDF, &device_bdf));
        if (bdf == device_bdf) {
            device = devices[i];
            cnt++;
            break;
        }
    }
    if (cnt == 0) {
        std::cout << "Invalid device bdf: " << bdf << ". Please check and provide valid bdf\n";
        exit(EXIT_FAILURE);
    }
    return device;
}
cl_device_id find_device_bdf_c(cl_device_id *devices, const std::string &bdf, cl_uint device_count) {
    char device_bdf[20];
    cl_int err;
    cl_device_id device;
    int cnt = 0;
    for (uint32_t i = 0; i < device_count; i++) {
        err = clGetDeviceInfo(devices[i], CL_DEVICE_PCIE_BDF, sizeof(device_bdf), device_bdf, 0);
        if (err != CL_SUCCESS) {
            std::cout << "Unable to extract the device BDF details\n";
            exit(EXIT_FAILURE);
        }
        if (bdf == device_bdf) {
            device = devices[i];
            cnt++;
            break;
        }
    }
    if (cnt == 0) {
        std::cout << "Invalid device bdf. Please check and provide valid bdf\n";
        exit(EXIT_FAILURE);
    }
    return device;
}
std::vector<unsigned char> read_binary_file(const std::string &xclbin_file_name) {
    FILE *fp;
    if ((fp = fopen(xclbin_file_name.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    // Loading XCL Bin into char buffer
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    auto nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    std::vector<unsigned char> buf;
    buf.resize(nb);
    bin_file.read(reinterpret_cast<char *>(buf.data()), nb);
    return buf;
}

bool is_emulation() {
    bool ret = false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if (xcl_mode != nullptr) {
        ret = true;
    }
    return ret;
}

bool is_hw_emulation() {
    bool ret = false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if ((xcl_mode != nullptr) && !strcmp(xcl_mode, "hw_emu")) {
        ret = true;
    }
    return ret;
}
double round_off(double n) {
    double d = n * 100.0;
    int i = d + 0.5;
    d = i / 100.0;
    return d;
}

std::string convert_size(size_t size) {
    static const char *SIZES[] = {"B", "KB", "MB", "GB"};
    uint32_t div = 0;
    size_t rem = 0;

    while (size >= 1024 && div < (sizeof SIZES / sizeof *SIZES)) {
        rem = (size % 1024);
        div++;
        size /= 1024;
    }

    double size_d = (float)size + (float)rem / 1024.0;
    double size_val = round_off(size_d);

    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << size_val;
    std::string size_str = stream.str();
    std::string result = size_str + " " + SIZES[div];
    return result;
}

bool is_xpr_device(const char *device_name) {
    const char *output = strstr(device_name, "xpr");

    if (output == nullptr) {
        return false;
    } else {
        return true;
    }
}
}; // namespace xcl
