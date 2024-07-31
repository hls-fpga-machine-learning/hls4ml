import json
from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.models import Sequential
import re

import hls4ml

test_root_path = Path(__file__).parent

# backends = ['Vivado', 'Vitis']
io_type_options = ['io_stream', 'io_parallel']
backend_options = ['Vitis']

import os

os.environ['XILINX_VITIS'] = "/opt/Xilinx/Vitis_HLS/2023.2/"
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']

def parse_cosim_report(project_path):
    prj_dir = None
    top_func_name = None

    project_tcl_path = project_path + '/project.tcl'

    with open(project_tcl_path) as f:
        for line in f.readlines():
            if 'set project_name' in line:
                top_func_name = line.split('"')[-2]
                prj_dir = top_func_name + '_prj'

    cosim_file_path = project_path + '/' + prj_dir + f'/solution1/sim/report/{top_func_name}_cosim.rpt'
    
    if os.path.isfile(cosim_file_path):
        return cosim_file_path
    else:
        raise FileNotFoundError("Co-simulation report not found.")

def fifo_depth_optimization_script(backend, profiling_fifo_depth, io_type):

    # build a keras model
    input_shape = (128, 128, 3)
    activation = 'relu'
    kernel_size = (3, 3)
    padding = 'same'

    model = Sequential()
    model.add(
        SeparableConv2D(filters=4, kernel_size=kernel_size, padding=padding, activation=activation, input_shape=input_shape)
    )
    model.add(SeparableConv2D(filters=8, kernel_size=kernel_size, padding=padding, activation=activation))

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(100, *input_shape)
    keras_prediction = model.predict(X_input)

    # execute fifo optimization
    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<8, 4>')
    config['Flows'] = ['vitis:fifo_depth_optimization']
    hls4ml.model.optimizer.get_optimizer('vitis:fifo_depth_optimization').configure(profiling_fifo_depth=profiling_fifo_depth)

    output_dir = str(test_root_path / f'hls4mlprj_fifo_depth_optimization_backend_{backend}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, io_type=io_type, hls_config=config, output_dir=output_dir, backend=backend
    )

    # build the new project with optimized depths
    hls_model.build(reset=False, csim=False, synth=True, cosim=True)
    hls4ml.report.read_vivado_report(output_dir)

    # checks if the fifo depths decreased
    fifo_depths = {}
    with open(hls_model.config.get_output_dir() + "/fifo_depths.json", "r") as fifo_depths_file:
        fifo_depths = json.load(fifo_depths_file)

    fifo_depths_descreased = True
    for fifo_name in fifo_depths.keys():
        if fifo_depths[fifo_name]['optimized'] >= fifo_depths[fifo_name]['initial']:
            fifo_depths_descreased = False

    # checks that cosimulation ran succesfully without detecting deadlocks
    cosim_report_path = parse_cosim_report(hls_model.config.get_output_dir())

    with open(cosim_report_path) as cosim_report_file:
        cosim_succesful = any("Pass" in line for line in cosim_report_file)

    # np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.001)
    assert cosim_succesful and fifo_depths_descreased
     
def expect_exception(error, message, backend, profiling_fifo_depth, io_type):
    with pytest.raises(error, match=re.escape(message)):
        fifo_depth_optimization_script(backend, profiling_fifo_depth, io_type)   
    
def expect_value_error(backend, profiling_fifo_depth):
    io_type = 'io_stream'
    value_error_expected_message = "The FIFO depth for profiling (profiling_fifo_depth variable) must be a non-negative integer."
    expect_exception(ValueError, value_error_expected_message, backend, profiling_fifo_depth, io_type)

def expect_runtime_error(backend, io_type):
    profiling_fifo_depth = 200_000
    runtime_error_expected_message = "To use this optimization you have to set `IOType` field to `io_stream` in the HLS config."
    expect_exception(RuntimeError, runtime_error_expected_message, backend, profiling_fifo_depth, io_type)

def expect_succeful_execution(backend):
    profiling_fifo_depth = 200_000
    io_type = 'io_stream'
    fifo_depth_optimization_script(backend, profiling_fifo_depth, io_type)

# @pytest.mark.skip(reason='Skipping synthesis tests for now')
@pytest.mark.parametrize('backend', backend_options)
def test_fifo_depth(backend):
    profiling_fifo_depth = -2
    expect_value_error(backend, profiling_fifo_depth)
    
    profiling_fifo_depth = "a"
    expect_value_error(backend, profiling_fifo_depth)
        
    io_type = 'io_parallel'
    expect_runtime_error(backend, io_type)

    expect_succeful_execution(backend)        
