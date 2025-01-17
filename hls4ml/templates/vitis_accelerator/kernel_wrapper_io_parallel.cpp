#include "firmware/myproject.h"
#include "kernel_wrapper.h"

static void read_input(const /*IN_INTERFACE_TYPE*/ *in, in_buffer_t (&in_buf)[BATCHSIZE][DATA_SIZE_IN]) {
    for (int i = 0; i < BATCHSIZE; i++) {
        #pragma HLS PIPELINE
        for (int j = 0; j < DATA_SIZE_IN; j++) {
            #pragma HLS UNROLL
            in_buf[i][j] = /*IN_HW_QUANT*/ in[i * DATA_SIZE_IN + j];
        }
    }
}
static void run_inference(in_buffer_t (&in_buf)[BATCHSIZE][DATA_SIZE_IN],
                          out_buffer_t (&out_buf)[BATCHSIZE][DATA_SIZE_OUT]) {
    for (int i = 0; i < BATCHSIZE; i++) {
        #pragma HLS DATAFLOW
        myproject(in_buf[i], out_buf[i]);
    }
}
static void write_result(/*OUT_INTERFACE_TYPE*/ *out, out_buffer_t (&out_buf)[BATCHSIZE][DATA_SIZE_OUT]) {
    for (int i = 0; i < BATCHSIZE; i++) {
        #pragma HLS PIPELINE
        for (int j = 0; j < DATA_SIZE_OUT; j++) {
            #pragma HLS UNROLL
            out[i * DATA_SIZE_OUT + j] = /*OUT_HW_QUANT*/ out_buf[i][j];
        }
    }
}

extern "C" {
/**
  \brief HLS4ML Kernel Implementation
  \param in Input Vector
  \param out Output Vector
*/
void kernel_wrapper(const /*IN_INTERFACE_TYPE*/ *in, /*OUT_INTERFACE_TYPE*/ *out) {
    in_buffer_t in_buf[BATCHSIZE][DATA_SIZE_IN];
    out_buffer_t out_buf[BATCHSIZE][DATA_SIZE_OUT];
    #pragma HLS ARRAY_RESHAPE   variable=in_buf  complete dim=2
    #pragma HLS ARRAY_RESHAPE   variable=out_buf complete dim=2

    #pragma HLS DATAFLOW
    read_input(in, in_buf);
    run_inference(in_buf, out_buf);
    write_result(out, out_buf);
}
}
