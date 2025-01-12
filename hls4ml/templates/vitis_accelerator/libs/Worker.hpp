#pragma once

#include <cstdint>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <vector>

#include "Types.hpp"
#include "xcl2.hpp"

template <class T, class U> class Worker {
  public:
    /**
     * \brief Constructor
     * \param batchsize Number of samples
     * \param sampleInputSize Flattened length of a single input to the model
     * \param sampleOutputSize Flattened length of a single output from the model
     * \param commandQueue cl::CommandQueue that the worker will enqueue operations to
     * \param queueMutex Mutex protecting the CommandQueue (potentially shared with other workers)
     */
    Worker(int deviceId, int workerId, int batchsize, int sampleInputSize, int sampleOutputSize, cl::CommandQueue &queue)
        : _deviceId(deviceId), _workerId(workerId), _batchsize(batchsize), _sampleInputSize(sampleInputSize),
          _sampleOutputSize(sampleOutputSize), _queue(queue), writeEvents(1), executionEvents(1) {
        memmap_in.resize(_batchsize * _sampleInputSize, T(0.0f));
        memmap_out.resize(_batchsize * _sampleOutputSize, U(0.0f));
    }

    /**
     * \brief Initializes all resources the Worker needs to drive a compute unit.
     * \param context cl::Context of the FPGA.
     * \param program cl:Program of the FPGA.
     * \param computeUnit The number of the physical compute unit this worker will use.
     */
    void initialize(cl::Context &context, cl::Program &program, int computeUnit) {
        cl_int err;

        // This is AMD's format for specifying the Compute Unit a kernel object uses
        std::string krnl_name = "kernel_wrapper:{kernel_wrapper_" + std::to_string(computeUnit) + "}";

        // Creating Kernel object
        OCL_CHECK(err, krnl = cl::Kernel(program, krnl_name.c_str(), &err));

        // Per AMD documentation we can leave XRT infer the bank location for the buffer:
        // " The XRT can obtain the bank location for the buffer if the buffer
        //   is used for setting the kernel arguments right after the buffer
        //   creation, i.e. before any enqueue operation on the buffer."

        const size_t vector_in_size_bytes = sizeof(T) * _batchsize * _sampleInputSize;
        const size_t vector_out_size_bytes = sizeof(U) * _batchsize * _sampleOutputSize;

        OCL_CHECK(err, input_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_in_size_bytes,
                                                 memmap_in.data(), &err));

        OCL_CHECK(err, output_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_out_size_bytes,
                                                  memmap_out.data(), &err));

        // Set kernel arguments will effectively affect the memory bank location
        OCL_CHECK(err, err = krnl.setArg(0, input_buffer));
        OCL_CHECK(err, err = krnl.setArg(1, output_buffer));

        // Perform a dummy transfer input batch to FPGA to ensure that allocation time is not counted
        // in the evaluation time. Also allows us to query the memory bank location.
        int mem_bank_index = -1;
        OCL_CHECK(err, err = _queue.enqueueMigrateMemObjects({input_buffer},
                                                             0,    // 0 means from host
                                                             NULL, // No dependencies
                                                             &writeEvents[0]));
        OCL_CHECK(err, err = writeEvents[0].wait());
        OCL_CHECK(err, err = clGetMemObjectInfo(input_buffer.get(), CL_MEM_BANK, sizeof(int), &mem_bank_index, nullptr));

        std::cout << "Initialized Worker " << _workerId << ", using CU " << computeUnit << " and memory bank "
                  << mem_bank_index << " on device " << _deviceId << std::endl;
    }

    /**
     * \brief Evaluates the single batch currently in input_buffer and writes to output_buffer.
     */
    void evaluate() {

        cl_int err;

        // Transfer input batch to FPGA
        OCL_CHECK(err, err = _queue.enqueueMigrateMemObjects({input_buffer},
                                                             0,    // 0 means from host
                                                             NULL, // No dependencies
                                                             &writeEvents[0]));

        // Run kernel on the batch
        OCL_CHECK(err, err = _queue.enqueueNDRangeKernel(krnl, 0, 1, 1, &writeEvents, &executionEvents[0]));

        // Transfer output batch from FPGA
        OCL_CHECK(err, err = _queue.enqueueMigrateMemObjects({output_buffer}, CL_MIGRATE_MEM_OBJECT_HOST, &executionEvents,
                                                             &readEvent));

        // Wait for all operations to complete
        OCL_CHECK(err, err = readEvent.wait());
    }

    /**
     * \brief Evaluates each batch of data provided via dataTracker. Uses float datatype
     * \param dataTracker Vector of input locations to read from and output locations to write to
     */
    void evalLoop(std::list<Batch<T, U>> &dataTracker) {

        while (!dataTracker.empty()) {
            // Copy inputs into memory-mapped buffer
            // FIXME: It there a way to avoid this copy? Could the orignal batch be used directly if aligned?
            const T *dataLoc = dataTracker.front().dataIn;
            memcpy(&memmap_in[0], dataLoc, _batchsize * _sampleInputSize * sizeof(T));

            // Evaluate
            evaluate();

            // Copy outputs into persistent results vector
            U *resLoc = dataTracker.front().dataOut;
            memcpy(resLoc, &memmap_out[0], _batchsize * _sampleOutputSize * sizeof(U));
            dataTracker.pop_front();
        }
    }

  private:
    int _deviceId;
    int _workerId;
    int _batchsize;
    int _sampleInputSize;
    int _sampleOutputSize;

    /// @brief Reference to the OpenCL command queue
    const cl::CommandQueue &_queue;

    /// @brief Vector mapped to FPGA input buffer
    std::vector<T, aligned_allocator<T>> memmap_in;
    /// @brief Vector mapped to FPGA output buffer
    std::vector<U, aligned_allocator<U>> memmap_out;

    /// @brief OpenCL buffer object for input
    cl::Buffer input_buffer;
    /// @brief OpenCL buffer object for output
    cl::Buffer output_buffer;
    /// @brief OpenCL kernel object
    cl::Kernel krnl;

    /// @brief Vector tracking write events. Required by OpenCL queue functions.
    std::vector<cl::Event> writeEvents;
    /// @brief Vector tracking kernel execution events. Required by OpenCL queue functions.
    std::vector<cl::Event> executionEvents;
    /// @brief Event for signaling output transfer completion
    cl::Event readEvent;
};
