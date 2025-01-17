#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "DataBatcher.hpp"
#include "Params.hpp"
#include "Types.hpp"
#include "Worker.hpp"
#include "xcl2.hpp"

template <class T, class U> class FpgaObj {
  public:
    /**
     * \brief Constructor
     * \param batchsize Number of samples
     * \param sampleInputSize Flattened length of a single input to the model
     * \param sampleOutputSize Flattened length of a single output from the model
     * \param numCU Number of compute units synthesized on the FPGA
     * \param xclbinFilename String containing path of synthesized xclbin
     */
    FpgaObj(const Params &params)
        : _batchsize(params.batchSize), _sampleInputSize(params.sampleInputSize), _sampleOutputSize(params.sampleOutputSize),
          _numCU(params.numCU), _xclbinFilename(params.xclbinFilename) {

        if (params.deviceBDFs.size() == 0) {
            // Finds all AMD/Xilinx devices present in system
            devices = xcl::get_xil_devices();
            if (devices.size() == 0) {
                throw std::runtime_error("No AMD/Xilinx FPGA devices found");
            }
            for (auto &device : devices) {
                std::string device_bdf;
                OCL_CHECK(err, err = device.getInfo(CL_DEVICE_PCIE_BDF, &device_bdf));
                std::cout << "Found device: " << device.getInfo<CL_DEVICE_NAME>() << " (" << device_bdf << ")" << std::endl;
            }

        } else {
            // Find devices by BDF
            devices.reserve(params.deviceBDFs.size());
            for (auto &bdf : params.deviceBDFs) {
                devices.push_back(xcl::find_device_bdf(xcl::get_xil_devices(), bdf));
                std::cout << "Found device: " << devices.back().getInfo<CL_DEVICE_NAME>() << " (" << bdf << ")" << std::endl;
            }
        }

        // Ensure that all devices are of the same type
        for (auto &device : devices) {
            std::string device_name = device.getInfo<CL_DEVICE_NAME>();
            if (_deviceName.empty()) {
                _deviceName = device_name;
            } else if (_deviceName != device_name) {
                throw std::runtime_error(
                    "All devices must be of the same type, use -d to specify the BDFs of the devices you want to use");
            }
        }

        _numDevice = devices.size();

        // Load xclbin
        std::cout << "Loading: " << _xclbinFilename << std::endl;
        std::vector<unsigned char> fileBuf = xcl::read_binary_file(_xclbinFilename);
        cl::Program::Binaries bins;
        for (int i = 0; i < _numDevice; i++) {
            bins.push_back({fileBuf.data(), fileBuf.size()});
        }

        // Create OpenCL context
        OCL_CHECK(err, context = cl::Context(devices, nullptr, nullptr, nullptr, &err));

        // Create OpenCL program from binary file
        OCL_CHECK(err, program = cl::Program(context, devices, bins, nullptr, &err));

        std::cout << "Device programmed successfully" << std::endl;

        // Create OpenCL program, and command queues for each device
        comQueues.resize(_numDevice);
        for (int i = 0; i < _numDevice; i++) {
            comQueues[i].resize(_numCU);
            // Create OpenCL out-of-order command queues (One per compute unit)
            for (int j = 0; j < _numCU; j++) {
                comQueues[i][j] = cl::CommandQueue(context, devices[i],
                                                   CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
            }
        }
    }

    /**
     * \brief Creates worker objects for each compute unit on each device
     * \param workersPerCU Number of worker objects that will drive each compute unit
     */
    void createWorkers(int workersPerCU) {
        _workersPerCU = workersPerCU;

        // Construct workers
        workers.reserve(_numCU * _workersPerCU);
        for (int d = 0; d < _numDevice; d++) {
            for (int cu = 0; cu < _numCU; cu++) {
                for (int w = 0; w < _workersPerCU; w++) {
                    workers.emplace_back(d, d * (_numCU * _workersPerCU) + cu * _workersPerCU + w, _batchsize,
                                         _sampleInputSize, _sampleOutputSize, comQueues[d][cu]);
                }
            }
        }

        // Initialize workers
        for (int d = 0; d < _numDevice; d++) {
            for (int cu = 0; cu < _numCU; cu++) {
                for (int w = 0; w < _workersPerCU; w++) {
                    workers[d * (_numCU * _workersPerCU) + cu * _workersPerCU + w].initialize(context, program, cu + 1);
                }
            }
        }
    }

    /**
     * \brief Loads data from a file into batches and distribute evenly amongst Workers.
     * \param fin Filename
     * \param s Input type. VitisAccelerator Backend currently uses text input. However,
     * the code also supports binary input in the format produced by NumPy's toFile().
     * \param profilingDataRepeat Additional number of times the given data is iterated
     * over. Profiling is enabled if this is greater than 0.
     */
    void loadData(const std::string &fin, int profilingDataRepeat = 0) {
        // Set-up containers for each Worker's batches/workload
        batchedData.reserve(_numCU * _workersPerCU * _numDevice);
        for (int i = 0; i < _numCU * _workersPerCU * _numDevice; i++) {
            batchedData.emplace_back();
        }

        // Batch and distribute data
        db = new DataBatcher<T, U>(_batchsize, _sampleInputSize, _sampleOutputSize, _numCU * _workersPerCU * _numDevice,
                                   profilingDataRepeat);
        db->read(fin);
        db->createResultBuffers();
        db->batch(batchedData);
    }

    /**
     * \brief Workers evaluate all loaded data. Each worker uses a separate thread.
     */
    void evaluateAll() {
        // Check that data has been loaded and batched
        if (batchedData.size() == 0 || db == nullptr) {
            throw std::runtime_error("No data loaded");
        }

        std::cout << "Starting FPGA run" << std::endl;

        auto ts_start = std::chrono::system_clock::now();

        std::vector<std::thread> accelThreads;
        accelThreads.reserve(_numCU * _workersPerCU * _numDevice);
        for (int i = 0; i < _numCU * _workersPerCU * _numDevice; i++) {
            accelThreads.emplace_back([this, i]() { this->workers[i].evalLoop(this->batchedData[i]); });
        }
        for (int i = 0; i < _numCU * _workersPerCU * _numDevice; i++) {
            accelThreads[i].join();
        }

        for (auto deviceQueue : comQueues) {
            for (auto queue : deviceQueue) {
                OCL_CHECK(err, err = queue.finish());
            }
        }

        auto ts_end = std::chrono::system_clock::now();

        uint64_t ns_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(ts_end - ts_start).count();
        if (db->isProfilingMode()) {
            double profilingThroughput = 1.0e9 * static_cast<double>(db->getProfilingSampleCount()) / ns_elapsed;
            std::cout << "Processed " << db->getProfilingSampleCount() << " samples in " << ns_elapsed / 1000000 << " ms"
                      << std::endl;
            std::cout << "Profiling throughput: " << profilingThroughput << " predictions/second" << std::endl;
        } else {
            double throughput = 1.0e9 * static_cast<double>(db->getSampleCount()) / ns_elapsed;
            double maxThroughput = 1.0e9 * static_cast<double>(db->getPaddedSampleCount()) / ns_elapsed;
            std::cout << "Utilized throughput: " << throughput << " predictions/second" << std::endl;
            std::cout << "Max possible throughput: " << maxThroughput << " predictions/second" << std::endl;
        }
    }

    void checkResults(const std::string &ref) {
        if (db == nullptr) {
            throw std::runtime_error("No data loaded");
        }
        if (db->readReference(ref)) {
            db->checkResults();
        } else {
            std::cout << "No reference file provided, skipping results check" << std::endl;
        }
    }

    /**
     * \brief Writes results, in text format, to provided file. Releases resources
     * \param fout Filename. If file already exists, it will be overwritten with current results.
     */
    void saveResults(const std::string &fout) {
        if (db == nullptr) {
            throw std::runtime_error("No data loaded");
        }
        db->write(fout);
        db->closeFile();
    }

  private:
    int _batchsize;
    int _sampleInputSize;
    int _sampleOutputSize;
    int _numCU;
    int _numDevice;
    std::string _xclbinFilename;
    std::string _deviceName;

    /// @brief A list of connected AMD/Xilinx devices
    std::vector<cl::Device> devices;
    /// @brief OpenCL Program that each compute unit executes
    cl::Program program;
    /// @brief OpenCL Device Context
    cl::Context context;
    /// @brief OpenCL Command Queues for each compute unit
    std::vector<std::vector<cl::CommandQueue>> comQueues;
    /// @brief Error code storage
    cl_int err;

    int _workersPerCU = 0;
    /// @brief Workers, indexed by (i_CU * _workersPerCU + i_worker)
    std::vector<Worker<T, U>> workers;
    /// @brief Data Batcher
    DataBatcher<T, U> *db = nullptr;
    /// @brief A vector containing each Worker's batches/workload
    std::vector<std::list<Batch<T, U>>> batchedData;
};
