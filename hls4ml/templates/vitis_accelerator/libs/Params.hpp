#pragma once

#include <ctype.h>
#include <getopt.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../kernel_wrapper.h"
#include "FpgaObj.hpp"

class Params {

  public:
    Params(int argc, char **argv) {
        int opt, temp;
        while ((opt = getopt(argc, argv, "x:vhr:n:i:o:d:c:")) != EOF)
            switch (opt) {
            case 'd':
                deviceBDFs.push_back(optarg);
                break;
            case 'x':
                xclbinFilename = optarg;
                break;
            case 'i':
                inputFilename = optarg;
                break;
            case 'o':
                outputFilename = optarg;
                break;
            case 'c':
                temp = atoi(optarg);
                if (temp > 0 && temp < NUM_CU)
                    ;
                numCU = temp;
                break;
            case 'n':
                temp = atoi(optarg);
                if (temp > 0)
                    numWorker = temp;
                break;
            case 'r':
                dataRepeatCount = atoi(optarg);
                break;
            case 'v':
                verbose++;
                break;
            case 'h':
                help();
                exit(0);
            default:
                std::cout << std::endl;
                abort();
            }

        if (verbose > 0)
            print();
    }

    void help(void) {
        std::cout << "Options:" << std::endl;
        std::cout << "  -d: device BDF (can be specified multiple times)" << std::endl;
        std::cout << "  -x: XCLBIN path" << std::endl;
        std::cout << "  -i: input file" << std::endl;
        std::cout << "  -o: output file" << std::endl;
        std::cout << "  -c: maximum computing units count" << std::endl;
        std::cout << "  -n: maximum workers count" << std::endl;
        std::cout << "  -r: input data repeat count" << std::endl;
        std::cout << "  -v: enable verbose output" << std::endl;
        std::cout << "  -h: this helps message" << std::endl;
    }

    void print(void) {
        std::cout << "Run parameters:" << std::endl;
        std::cout << "   xclbinFilename: " << xclbinFilename << std::endl;
        std::cout << "        batchSize: " << batchSize << std::endl;
        std::cout << "  sampleInputSize: " << sampleInputSize << std::endl;
        std::cout << " sampleOutputSize: " << sampleOutputSize << std::endl;
        std::cout << "            numCU: " << numCU << std::endl;
        std::cout << "    inputFilename: " << inputFilename << std::endl;
        std::cout << "   outputFilename: " << outputFilename << std::endl;
        std::cout << "        numWorker: " << numWorker << std::endl;
        std::cout << "  dataRepeatCount: " << dataRepeatCount << std::endl;
    }

    // Device
    std::vector<std::string> deviceBDFs;

    // Bitstream
    std::string xclbinFilename = "./build_hw_rel/kernel_wrapper.xclbin";
    size_t batchSize = BATCHSIZE;
    const size_t sampleInputSize = INSTREAMSIZE;
    const size_t sampleOutputSize = OUTSTREAMSIZE;
    size_t numCU = NUM_CU;

    // Data paths
    std::string inputFilename = "./tb_data/tb_input_features.dat";
    std::string referenceFilename = "tb_data/tb_output_predictions.dat";
    std::string outputFilename = "./tb_data/hw_results.dat";

    // Workers
    int numWorker = NUM_WORKER;

    // Benchmark
    int dataRepeatCount = -1;
    int verbose = 0;
};
