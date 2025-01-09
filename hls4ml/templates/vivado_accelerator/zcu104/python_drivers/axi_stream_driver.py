from datetime import datetime

import numpy as np
from pynq import Overlay, allocate


class NeuralNetworkOverlay(Overlay):
    def __init__(
        self, bitfile_name, x_shape, y_shape, dtype=np.float32, dtbo=None, download=True, ignore_version=False, device=None
    ):
        super().__init__(bitfile_name, dtbo=None, download=True, ignore_version=False, device=None)
        self.sendchannel = self.hier_0.axi_dma_0.sendchannel
        self.recvchannel = self.hier_0.axi_dma_0.recvchannel
        self.input_buffer = allocate(shape=x_shape, dtype=dtype)
        self.output_buffer = allocate(shape=y_shape, dtype=dtype)

    def _print_dt(self, timea, timeb, N):
        dt = timeb - timea
        dts = dt.seconds + dt.microseconds * 10**-6
        rate = N / dts
        print(f"Classified {N} samples in {dts} seconds ({rate} inferences / s)")
        return dts, rate

    def predict(self, X, debug=False, profile=False, encode=None, decode=None):
        """
        Obtain the predictions of the NN implemented in the FPGA.
        Parameters:
        - X : the input vector. Should be numpy ndarray.
        - dtype : the data type of the elements of the input/output vectors.
                  Note: it should be set depending on the interface of the accelerator; if it uses 'float'
                  types for the 'data' AXI-Stream field, 'np.float32' dtype is the correct one to use.
                  Instead if it uses 'ap_fixed<A,B>', 'np.intA' is the correct one to use (note that A cannot
                  any integer value, but it can assume {..., 8, 16, 32, ...} values. Check `numpy`
                  doc for more info).
                  In this case the encoding/decoding has to be computed by the PS. For example for
                  'ap_fixed<16,6>' type the following 2 functions are the correct one to use for encode/decode
                  'float' -> 'ap_fixed<16,6>':
                  ```
                    def encode(xi):
                        return np.int16(round(xi * 2**10)) # note 2**10 = 2**(A-B)
                    def decode(yi):
                        return yi * 2**-10
                    encode_v = np.vectorize(encode) # to apply them element-wise
                    decode_v = np.vectorize(decode)
                  ```
        - profile : boolean. Set it to `True` to print the performance of the algorithm in term of `inference/s`.
        - encode/decode: function pointers. See `dtype` section for more information.
        - return: an output array based on `np.ndarray` with a shape equal to `y_shape` and a `dtype` equal to
                  the namesake parameter.
        """
        if profile:
            timea = datetime.now()
        if encode is not None:
            X = encode(X)
        self.input_buffer[:] = X
        self.sendchannel.transfer(self.input_buffer)
        self.recvchannel.transfer(self.output_buffer)
        if debug:
            print("Transfer OK")
        self.sendchannel.wait()
        if debug:
            print("Send OK")
        self.recvchannel.wait()
        if debug:
            print("Receive OK")
        # result = self.output_buffer.copy()
        if decode is not None:
            self.output_buffer = decode(self.output_buffer)

        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))
            return self.output_buffer, dts, rate
        else:
            return self.output_buffer
