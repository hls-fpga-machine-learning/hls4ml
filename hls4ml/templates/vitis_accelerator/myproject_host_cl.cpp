#include <string>

#include "FpgaObj.hpp"
#include "Params.hpp"
#include "Types.hpp"
#include "kernel_wrapper.h"
#include "xcl2.hpp"

int main(int argc, char **argv) {

    Params params(argc, argv);

    FpgaObj</*INTERFACE_TYPES*/> fpga(params);

    fpga.createWorkers(params.numWorker);

    fpga.loadData(params.inputFilename, params.dataRepeatCount);

    fpga.evaluateAll();

    fpga.checkResults(params.referenceFilename);

    fpga.saveResults(params.outputFilename);

    return EXIT_SUCCESS;
}
