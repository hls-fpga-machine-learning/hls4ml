import importlib
import os

import yaml

from hls4ml.converters.keras_to_hls import KerasFileReader  # noqa: F401
from hls4ml.converters.keras_to_hls import KerasModelReader  # noqa: F401
from hls4ml.converters.keras_to_hls import KerasReader  # noqa: F401
from hls4ml.converters.keras_to_hls import get_supported_keras_layers  # noqa: F401
from hls4ml.converters.keras_to_hls import parse_keras_model  # noqa: F401
from hls4ml.converters.keras_to_hls import keras_to_hls, register_keras_layer_handler
from hls4ml.converters.onnx_to_hls import get_supported_onnx_layers  # noqa: F401
from hls4ml.converters.onnx_to_hls import parse_onnx_model  # noqa: F401
from hls4ml.converters.onnx_to_hls import onnx_to_hls, register_onnx_layer_handler
from hls4ml.converters.pytorch_to_hls import (  # noqa: F401
    get_supported_pytorch_layers,
    pytorch_to_hls,
    register_pytorch_layer_handler,
)
from hls4ml.model import ModelGraph
from hls4ml.utils.config import create_config
from hls4ml.utils.dependency import requires
from hls4ml.utils.symbolic_utils import LUTFunction

# ----------Layer handling register----------#
model_types = ['keras', 'pytorch', 'onnx']

for model_type in model_types:
    for module in os.listdir(os.path.dirname(__file__) + f'/{model_type}'):
        if module == '__init__.py' or module[-3:] != '.py':
            continue
        try:
            lib = importlib.import_module(__name__ + f'.{model_type}.' + module[:-3])
            for _, func in list(lib.__dict__.items()):
                # if 'func' is callable (i.e., function, class...)
                # and has 'handles' attribute
                # and is defined in this module (i.e., not imported)
                if callable(func) and hasattr(func, 'handles') and func.__module__ == lib.__name__:
                    for layer in func.handles:  # type: ignore
                        if model_type == 'keras':
                            register_keras_layer_handler(layer, func)
                        elif model_type == 'pytorch':
                            register_pytorch_layer_handler(layer, func)
                        elif model_type == 'onnx':
                            register_onnx_layer_handler(layer, func)

        except ImportError as err:
            print(f'WARNING: Failed to import handlers from {module}: {err.msg}.')
            continue


def parse_yaml_config(config_file):
    """Parse conversion configuration from the provided YAML file.

    This function parses the conversion configuration contained in the YAML
    file provided as an argument. It ensures proper serialization of hls4ml
    objects and should be called on YAML files created by hls4ml. A minimal
    valid YAML file may look like this::

        KerasH5: my_keras_model.h5
        OutputDir: my-hls-test
        ProjectName: myproject
        Part: xcvu13p-flga2577-2-e
        ClockPeriod: 5
        IOType: io_stream
        HLSConfig:
            Model:
            Precision: ap_fixed<16,6>
            ReuseFactor: 10

    Please refer to the docs for more examples of valid YAML configurations.

    Arguments:
        config_file (str): Location of the file on the filesystem.

    Returns:
        dict: Parsed configuration.
    """

    def construct_keras_model(loader, node):
        model_str = loader.construct_scalar(node)
        import keras

        return keras.models.load_model(model_str)

    yaml.add_constructor('!keras_model', construct_keras_model, Loader=yaml.SafeLoader)

    print('Loading configuration from', config_file)
    with open(config_file) as file:
        parsed_config = yaml.safe_load(file)
    return parsed_config


def convert_from_config(config):
    """Convert to hls4ml model based on the provided configuration.

    Arguments:
        config: A string containing the path to the YAML configuration file on
            the filesystem or a dict containing the parsed configuration.

    Returns:
        ModelGraph: hls4ml model.
    """

    if isinstance(config, str):
        yamlConfig = parse_yaml_config(config)
    else:
        yamlConfig = config

    model = None
    if 'OnnxModel' in yamlConfig:
        model = onnx_to_hls(yamlConfig)
    elif 'PytorchModel' in yamlConfig:
        model = pytorch_to_hls(yamlConfig)
    else:
        model = keras_to_hls(yamlConfig)

    return model


def _check_hls_config(config, hls_config):
    """
    Check hls_config for to set appropriate parameters for config.
    """

    if 'LayerName' in hls_config:
        config['HLSConfig']['LayerName'] = hls_config['LayerName']

    if 'LayerType' in hls_config:
        config['HLSConfig']['LayerType'] = hls_config['LayerType']

    if 'Flows' in hls_config:
        config['HLSConfig']['Flows'] = hls_config['Flows']

    if 'Optimizers' in hls_config:
        config['HLSConfig']['Optimizers'] = hls_config['Optimizers']

    if 'SkipOptimizers' in hls_config:
        config['HLSConfig']['SkipOptimizers'] = hls_config['SkipOptimizers']

    return


def _check_model_config(model_config):
    if model_config is not None:
        if not all(k in model_config for k in ('Precision', 'ReuseFactor')):
            raise Exception('Precision and ReuseFactor must be provided in the hls_config')
    else:
        model_config = {}
        model_config['Precision'] = 'ap_fixed<16,6>'
        model_config['ReuseFactor'] = 1

    return model_config


@requires('_keras')
def convert_from_keras_model(
    model,
    output_dir='my-hls-test',
    project_name='myproject',
    input_data_tb=None,
    output_data_tb=None,
    backend='Vivado',
    hls_config=None,
    **kwargs,
):
    """Convert Keras model to hls4ml model based on the provided configuration.

    Args:
        model: Keras model to convert
        output_dir (str, optional): Output directory of the generated HLS
            project. Defaults to 'my-hls-test'.
        project_name (str, optional): Name of the HLS project.
            Defaults to 'myproject'.
        input_data_tb (str, optional): String representing the path of input data in .npy or .dat format that will be
            used during csim and cosim.
        output_data_tb (str, optional): String representing the path of output data in .npy or .dat format that will be
            used during csim and cosim.
        backend (str, optional): Name of the backend to use, e.g., 'Vivado'
            or 'Quartus' or 'Catapult'.
        board (str, optional): One of target boards specified in `supported_board.json` file. If set to `None` a default
            device of a backend will be used. See documentation of the backend used.
        part (str, optional): The FPGA part. If set to `None` a default part of a backend will be used.
            See documentation of the backend used. Note that if `board` is specified, the part associated to that board
            will overwrite any part passed as a parameter.
        clock_period (int, optional): Clock period of the design.
            Defaults to 5.
        clock_uncertainty (str, optional): Clock uncertainty
            Defaults to 12.5% for Vivado HLS, 27% for Vitis HLS; unused for others
        io_type (str, optional): Type of implementation used. One of
            'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
        hls_config (dict, optional): The HLS config.
        kwargs** (dict, optional): Additional parameters that will be used to create the config of the specified backend

    Raises:
        Exception: If precision and reuse factor are not present in 'hls_config'.

    Returns:
        ModelGraph: hls4ml model.
    """

    config = create_config(output_dir=output_dir, project_name=project_name, backend=backend, **kwargs)

    config['KerasModel'] = model
    config['InputData'] = input_data_tb
    config['OutputPredictions'] = output_data_tb
    config['HLSConfig'] = {}

    if hls_config is None:
        hls_config = {}

    model_config = hls_config.get('Model', None)
    config['HLSConfig']['Model'] = _check_model_config(model_config)

    _check_hls_config(config, hls_config)

    return keras_to_hls(config)


@requires('_torch')
def convert_from_pytorch_model(
    model,
    output_dir='my-hls-test',
    project_name='myproject',
    input_data_tb=None,
    output_data_tb=None,
    backend='Vivado',
    hls_config=None,
    **kwargs,
):
    """Convert PyTorch model to hls4ml model based on the provided configuration.

    Args:
        model: PyTorch model to convert.
        output_dir (str, optional): Output directory of the generated HLS project. Defaults to 'my-hls-test'.
        project_name (str, optional): Name of the HLS project. Defaults to 'myproject'.
        input_data_tb (str, optional): String representing the path of input data in .npy or .dat format that will be
            used during csim and cosim. Defaults to None.
        output_data_tb (str, optional): String representing the path of output data in .npy or .dat format that will be
            used during csim and cosim. Defaults to None.
        backend (str, optional): Name of the backend to use, e.g., 'Vivado' or 'Quartus' or 'Catapult'. Defaults to 'Vivado'.
        board (str, optional): One of target boards specified in `supported_board.json` file. If set to `None` a default
            device of a backend will be used. See documentation of the backend used.
        part (str, optional): The FPGA part. If set to `None` a default part of a backend will be used.
            See documentation of the backend used. Note that if `board` is specified, the part associated to that board
            will overwrite any part passed as a parameter.
        clock_period (int, optional): Clock period of the design.
            Defaults to 5.
        clock_uncertainty (str, optional): Clock uncertainty
            Defaults to 12.5% for Vivado HLS, 27% for Vitis HLS; unused for others
        io_type (str, optional): Type of implementation used. One of
            'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
        hls_config (dict, optional): The HLS config.
        kwargs** (dict, optional): Additional parameters that will be used to create the config of the specified backend.

    Raises:
        Exception: If precision and reuse factor are not present in 'hls_config'.

    Notes:
        Pytorch uses the "channels_first" data format for its tensors, while hls4ml expects the "channels_last" format
        used by keras. By default, hls4ml will automatically add layers to the model which transpose the inputs to the
        "channels_last" format. Not that this is not supported for the "io_stream" io_type, for which the user will have
        to transpose the input by hand before passing it to hls4ml. In that case the "channels_last_conversion" argument of
        the "config_from_pytorch_model" function needs to be set to "internal". This argument can be used to completely
        disable this internal conversion. By default, the output of the model remains
        in the "channels_last" data format. The "transpose_outputs" argument of the "config_from_pytorch_model" can be
        used to add a layer to the model that transposes back to "channels_first". As before, this will not work for
        io_stream.

    Returns:
        ModelGraph: hls4ml model.
    """

    config = create_config(output_dir=output_dir, project_name=project_name, backend=backend, **kwargs)

    config['PytorchModel'] = model
    config['InputData'] = input_data_tb
    config['OutputPredictions'] = output_data_tb
    config['HLSConfig'] = {}

    if hls_config is None:
        hls_config = {}

    model_config = hls_config.get('Model')
    config['HLSConfig']['Model'] = _check_model_config(model_config)
    config['InputShape'] = hls_config.get('InputShape')
    _check_hls_config(config, hls_config)

    return pytorch_to_hls(config)


@requires('onnx')
def convert_from_onnx_model(
    model,
    output_dir='my-hls-test',
    project_name='myproject',
    input_data_tb=None,
    output_data_tb=None,
    backend='Vivado',
    hls_config=None,
    **kwargs,
):
    """Convert Keras model to hls4ml model based on the provided configuration.

    Args:
        model: ONNX model to convert.
        output_dir (str, optional): Output directory of the generated HLS
            project. Defaults to 'my-hls-test'.
        project_name (str, optional): Name of the HLS project.
            Defaults to 'myproject'.
        input_data_tb (str, optional): String representing the path of input data in .npy or .dat format that will be
            used during csim and cosim.
        output_data_tb (str, optional): String representing the path of output data in .npy or .dat format that will be
            used during csim and cosim.
        backend (str, optional): Name of the backend to use, e.g., 'Vivado'
            or 'Quartus' or 'Catapult'.
        board (str, optional): One of target boards specified in `supported_board.json` file. If set to `None` a default
            device of a backend will be used. See documentation of the backend used.
        part (str, optional): The FPGA part. If set to `None` a default part of a backend will be used.
            See documentation of the backend used. Note that if `board` is specified, the part associated to that board
            will overwrite any part passed as a parameter.
        clock_period (int, optional): Clock period of the design.
            Defaults to 5.
        clock_uncertainty (str, optional): Clock uncertainty
            Defaults to 12.5% for Vivado HLS, 27% for Vitis HLS; unused for others
        io_type (str, optional): Type of implementation used. One of
            'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
        hls_config (dict, optional): The HLS config.
        kwargs** (dict, optional): Additional parameters that will be used to create the config of the specified backend

    Raises:
        Exception: If precision and reuse factor are not present in 'hls_config'.

    Returns:
        ModelGraph: hls4ml model.
    """

    config = create_config(output_dir=output_dir, project_name=project_name, backend=backend, **kwargs)

    config['OnnxModel'] = model
    config['InputData'] = input_data_tb
    config['OutputPredictions'] = output_data_tb
    config['HLSConfig'] = {}

    if hls_config is None:
        hls_config = {}

    model_config = hls_config.get('Model', None)
    config['HLSConfig']['Model'] = _check_model_config(model_config)

    _check_hls_config(config, hls_config)

    return onnx_to_hls(config)


@requires('sr')
def convert_from_symbolic_expression(
    expr,
    n_symbols=None,
    lut_functions=None,
    use_built_in_lut_functions=False,
    output_dir='my-hls-test',
    project_name='myproject',
    input_data_tb=None,
    output_data_tb=None,
    precision='ap_fixed<16,6>',
    **kwargs,
):
    """Converts a given (SymPy or string) expression to hls4ml model.

    Args:
        expr (str or sympy.Expr): Expression to convert. The variables in the expression should be in the form of
            ``x0, x1, x2, ...``.
        n_symbols (int, optional): Number of symbols in the expression. If not provided, the largest index of the variable
            will be used as the number of symbols. Useful if number of inputs differs from the number of variables used
            in the expression. Defaults to None.
        lut_functions (dict, optional): LUT function definitions. Defaults to None.
            The dictionary should have the form of::

                {
                    '<func_name>': {
                        'math_func': '<func>',
                        'table_size': <table_size>,
                        'range_start': <start>,
                        'range_end': <end>,
                    }
                }

            where ``<func_name>`` is a given name that can be used with PySR, ``<func>`` is the math function to
            approximate (`sin`, `cos`, `log`,...), ``<table_size>`` is the size of the lookup table, and ``<start>`` and
            ``<end>`` are the ranges in which the function will be approximated. It is **strongly** recommended to use a
            power-of-two as a range.
        use_built_in_lut_functions (bool, optional): Use built-in sin/cos LUT functions. Defaults to False.
        output_dir (str, optional): Output directory of the generated HLS
            project. Defaults to 'my-hls-test'.
        project_name (str, optional): Name of the HLS project.
            Defaults to 'myproject'.
        input_data_tb (str, optional): String representing the path of input data in .npy or .dat format that will be
            used during csim and cosim.
        output_data_tb (str, optional): String representing the path of output data in .npy or .dat format that will be
            used during csim and cosim.
        precision (str, optional): Precision to use. Defaults to 'ap_fixed<16,6>'.
        part (str, optional): The FPGA part. If set to `None` a default part of a backend will be used.
        clock_period (int, optional): Clock period of the design.
            Defaults to 5.
        compiler (str, optional): Compiler to use, ``vivado_hls`` or ``vitis_hls``. Defaults to ``vivado_hls``.
        hls_include_path (str, optional): Path to HLS inlcude files. If `None` the location will be inferred from the
            location of the `compiler` used. If an empty string is passed the HLS math libraries won't be used during
            compilation, meaning Python integration won't work unless all functions are LUT-based. Doesn't affect synthesis.
            Defaults to None.
        hls_libs_path (str, optional): Path to HLS libs files. If `None` the location will be inferred from the
            location of the `compiler` used. Defaults to None.

    Returns:
        ModelGraph: hls4ml model.
    """
    import sympy

    if not isinstance(expr, (list, set)):
        expr = [expr]
    for i, e in enumerate(expr):
        if not isinstance(e, sympy.Expr):
            expr[i] = sympy.parsing.sympy_parser.parse_expr(e)

    if n_symbols is None:
        n_symbols = 0
        for e in expr:
            symbols = max([int(d.name.replace('x', '')) for d in e.free_symbols]) + 1
            if symbols > n_symbols:
                n_symbols = symbols

    if lut_functions is None:
        lut_functions = []
    else:
        if isinstance(lut_functions, dict):
            lut_functions = [
                LUTFunction(name, params['math_func'], params['range_start'], params['range_end'], params['table_size'])
                for name, params in lut_functions.items()
            ]

    layer_list = []

    input_layer = {}
    input_layer['name'] = 'x'
    input_layer['class_name'] = 'InputLayer'
    input_layer['input_shape'] = [n_symbols]
    layer_list.append(input_layer)

    expr_layer = {}
    expr_layer['name'] = 'expr1'
    expr_layer['class_name'] = 'SymbolicExpression'
    expr_layer['expression'] = [str(e) for e in expr]
    expr_layer['n_symbols'] = n_symbols
    expr_layer['lut_functions'] = lut_functions
    expr_layer['use_built_in_luts'] = use_built_in_lut_functions
    layer_list.append(expr_layer)

    config = create_config(output_dir=output_dir, project_name=project_name, backend='SymbolicExpression', **kwargs)

    # config['Expression'] = str(expr)
    config['NSymbols'] = n_symbols
    config['InputData'] = input_data_tb
    config['OutputPredictions'] = output_data_tb

    config['HLSConfig'] = {'Model': {'Precision': precision, 'ReuseFactor': 1}}

    hls_model = ModelGraph(config, layer_list)

    return hls_model
