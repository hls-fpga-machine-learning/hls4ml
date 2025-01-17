import json
import os

import numpy as np

from hls4ml.model.layers import FixedPrecisionType, IntegerPrecisionType


class VitisAcceleratorIPFlowConfig:
    def __init__(self, config, model_inputs, model_outputs):
        self.config = config.config
        self.board = self.config.get('AcceleratorConfig', {}).get('Board', 'pynq-z2')
        self.supported_boards = json.load(open(os.path.dirname(__file__) + '/supported_boards.json'))
        if self.board in self.supported_boards.keys():
            board_info = self.supported_boards[self.board]
            self.part = board_info['part']
        else:
            raise Exception('The board does not appear in supported_boards.json file')

        if self.config.get('Part') is not None:
            if self.config.get('Part') != self.part:
                print(
                    'WARNING: You set a Part that does not correspond to the Board you specified. The correct '
                    'Part is now set.'
                )
                self.config['Part'] = self.part
        accel_config = self.config.get('AcceleratorConfig', None)
        if accel_config is not None:
            prec = accel_config.get('Precision')
            if prec is None:
                raise Exception('Precision must be provided in the AcceleratorConfig')
            else:
                if prec.get('Input') is None or prec.get('Output') is None:
                    raise Exception('Input and Output fields must be provided in the AcceleratorConfig->Precision')
        else:
            accel_config = {
                'Precision': {'Input': 'float', 'Output': 'float'},
                'Driver': 'python',
                'Interface': 'axi_stream',
            }
            config.config['AcceleratorConfig'] = accel_config

        self.interface = self.config['AcceleratorConfig'].get('Interface', 'axi_stream')  # axi_stream, axi_master, axi_lite
        self.driver = self.config['AcceleratorConfig'].get('Driver', 'python')  # python or c
        self.input_type = self.config['AcceleratorConfig']['Precision'].get(
            'Input', 'float'
        )  # float, double or ap_fixed<a,b>
        self.output_type = self.config['AcceleratorConfig']['Precision'].get(
            'Output', 'float'
        )  # float, double or ap_fixed<a,b>
        self.platform = self.config['AcceleratorConfig'].get(
            'Platform', 'xilinx_u250_xdma_201830_2'
        )  # Get platform folder name

        assert (
            len(model_inputs) == 1
        ), "Only models with one input tensor are currently supported by VitisAcceleratorIPFlowBackend"
        assert (
            len(model_outputs) == 1
        ), "Only models with one output tensor are currently supported by VitisAcceleratorIPFlowBackend"
        self.inp = model_inputs[0]
        self.out = model_outputs[0]
        inp_axi_t = self.input_type
        out_axi_t = self.output_type

        if inp_axi_t not in ['float', 'double']:
            self.input_type = self._next_factor8_type(config.backend.convert_precision_string(inp_axi_t))
        if out_axi_t not in ['float', 'double']:
            self.output_type = self._next_factor8_type(config.backend.convert_precision_string(out_axi_t))

        if self.input_type == 'float':
            self.input_bitwidth = 32
        elif self.input_type == 'double':
            self.input_bitwidth = 64
        else:
            self.input_bitwidth = config.backend.convert_precision_string(inp_axi_t).width

        if out_axi_t == 'float':
            self.output_bitwidth = 32
        elif out_axi_t == 'double':
            self.output_bitwidth = 64
        else:
            self.output_bitwidth = config.backend.convert_precision_string(out_axi_t).width

    def _next_factor8_type(self, p):
        '''Return a new type with the width rounded to the next factor of 8 up to p's width
        Args:
            p : IntegerPrecisionType or FixedPrecisionType
        Returns:
            An IntegerPrecisionType or FixedPrecisionType with the width rounder up to the next factor of 8
            of p's width. Other parameters (fractional bits, extra modes) stay the same.
        '''
        W = p.width
        newW = int(np.ceil(W / 8) * 8)
        if isinstance(p, FixedPrecisionType):
            return FixedPrecisionType(newW, p.integer, p.signed, p.rounding_mode, p.saturation_mode, p.saturation_bits)
        elif isinstance(p, IntegerPrecisionType):
            return IntegerPrecisionType(newW, p.signed)

    def get_io_bitwidth(self):
        return self.input_bitwidth, self.output_bitwidth

    def get_corrected_types(self):
        return self.input_type, self.output_type, self.inp, self.out

    def get_interface(self):
        return self.interface

    def get_board_info(self, board=None):
        if board is None:
            board = self.board
        if board in self.supported_boards.keys():
            return self.supported_boards[board]
        else:
            raise Exception('The board is still not supported')

    def get_part(self):
        return self.part

    def get_driver(self):
        return self.driver

    def get_board(self):
        return self.board

    def get_platform(self):
        return self.platform

    def get_clock_period(self):
        return self.clock_period

    def get_driver_path(self):
        if self.board.startswith('alveo'):
            return '../templates/vitis_accelerator_ip_flow/' + 'alveo/' + self.driver + '_drivers/' + self.get_driver_file()
        else:
            return (
                '../templates/vitis_accelerator_ip_flow/'
                + self.board
                + '/'
                + self.driver
                + '_drivers/'
                + self.get_driver_file()
            )

    def get_driver_file(self):
        driver_ext = '.py' if self.driver == 'python' else '.h'
        return self.interface + '_driver' + driver_ext

    def get_krnl_rtl_src_dir(self):
        return '../templates/vitis_accelerator_ip_flow/' + 'alveo/' + '/krnl_rtl_src'

    def get_input_type(self):
        return self.input_type

    def get_output_type(self):
        return self.output_type

    def get_tcl_file_path(self):
        board_info = self.get_board_info(self.board)
        tcl_scripts = board_info.get('tcl_scripts', None)
        if tcl_scripts is None:
            raise Exception('No tcl scripts definition available for the board in supported_board.json')
        tcl_script = tcl_scripts.get(self.interface, None)
        if tcl_script is None:
            raise Exception('No tcl script definition available for the desired interface in supported_board.json')
        if self.board.startswith('alveo'):
            return '../templates/vitis_accelerator_ip_flow/' + 'alveo/' + '/tcl_scripts/' + tcl_script
        else:
            return '../templates/vitis_accelerator_ip_flow/' + self.board + '/tcl_scripts/' + tcl_script
