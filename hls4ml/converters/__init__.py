from __future__ import absolute_import
import os
import yaml
import importlib

from hls4ml.utils.config import create_vivado_config

from hls4ml.converters.keras_to_hls import keras_to_hls, get_supported_keras_layers, register_keras_layer_handler

#----------Make converters available if the libraries can be imported----------#       
try:
    from hls4ml.converters.pytorch_to_hls import pytorch_to_hls, get_supported_pytorch_layers, register_pytorch_layer_handler
    __pytorch_enabled__ = True
except ImportError:
    __pytorch_enabled__ = False

try:
    from hls4ml.converters.onnx_to_hls import onnx_to_hls, get_supported_onnx_layers, register_onnx_layer_handler
    __onnx_enabled__ = True
except ImportError:
    __onnx_enabled__ = False

try:
    from hls4ml.converters.tf_to_hls import tf_to_hls
    __tensorflow_enabled__ = True
except ImportError:
    __tensorflow_enabled__ = False

#----------Layer handling register----------#
model_types = ['keras', 'pytorch', 'onnx']

for model_type in model_types:
    for module in os.listdir(os.path.dirname(__file__) + '/{}'.format(model_type)):
        if module == '__init__.py' or module[-3:] != '.py':
            continue
        try:
            lib = importlib.import_module(__name__ + '.{}.'.format(model_type) + module[:-3])
            for name, func in list(lib.__dict__.items()):
                # if 'func' is callable (i.e., function, class...)
                # and has 'handles' attribute
                # and is defined in this module (i.e., not imported)
                if callable(func) and hasattr(func, 'handles') and func.__module__ == lib.__name__:
                    for layer in func.handles:
                        
                        if model_type == 'keras':
                            register_keras_layer_handler(layer, func)
                        elif model_type == 'pytorch':
                            register_pytorch_layer_handler(layer, func)
                        elif model_type == 'onnx':
                            register_onnx_layer_handler(layer, func)
                            
        except ImportError:
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
        XilinxPart: xcku115-flvb2104-2-i
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
        from tensorflow.keras.models import load_model

        model_str = loader.construct_scalar(node)
        return load_model(model_str)

    yaml.add_constructor(u'!keras_model', construct_keras_model, Loader=yaml.SafeLoader)

    print('Loading configuration from', config_file)
    with open(config_file, 'r') as file:
        parsed_config = yaml.load(file, Loader=yaml.SafeLoader)
    return parsed_config

def convert_from_config(config):
    """Convert to hls4ml model based on the provided configuration.

    Arguments:
        config: A string containing the path to the YAML configuration file on
            the filesystem or a dict containig the parsed configuration.

    Returns:
        HLSModel: hls4ml model.
    """

    if isinstance(config, str):
        yamlConfig = parse_yaml_config(config)
    else:
        yamlConfig = config
        
    model = None
    if 'OnnxModel' in yamlConfig:
        if __onnx_enabled__:
            model = onnx_to_hls(yamlConfig)
        else:
            raise Exception("ONNX not found. Please install ONNX.")
    elif 'PytorchModel' in yamlConfig:
        if __pytorch_enabled__:
            model = pytorch_to_hls(yamlConfig)
        else:
            raise Exception("PyTorch not found. Please install PyTorch.")
    elif 'TensorFlowModel' in yamlConfig:
        if __tensorflow_enabled__:
            model = tf_to_hls(yamlConfig)
        else:
            raise Exception("TensorFlow not found. Please install TensorFlow.")
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
        model_config['ReuseFactor'] = '1'
        
    return model_config
    

def convert_from_keras_model(model, output_dir='my-hls-test', project_name='myproject',
    fpga_part='xcku115-flvb2104-2-i', clock_period=5, io_type='io_parallel', hls_config={}):
    """Convert to hls4ml model based on the provided configuration.

    Args:
        model: Keras model to convert
        output_dir (str, optional): Output directory of the generated HLS
            project. Defaults to 'my-hls-test'.
        project_name (str, optional): Name of the HLS project.
            Defaults to 'myproject'.
        fpga_part (str, optional): The target FPGA device.
            Defaults to 'xcku115-flvb2104-2-i'.
        clock_period (int, optional): Clock period of the design.
            Defaults to 5.
        io_type (str, optional): Type of implementation used. One of
            'io_parallel' or 'io_serial'. Defaults to 'io_parallel'.
        hls_config (dict, optional): The HLS config.

    Raises:
        Exception: If precision and reuse factor are not present in 'hls_config'

    Returns:
        HLSModel: hls4ml model.
    """

    config = create_vivado_config(
        output_dir=output_dir,
        project_name=project_name,
        fpga_part=fpga_part,
        clock_period=clock_period,
        io_type=io_type
    )
    config['KerasModel'] = model

    model_config = hls_config.get('Model', None)
    config['HLSConfig']['Model'] = _check_model_config(model_config)
    
    _check_hls_config(config, hls_config)

    return keras_to_hls(config)


def convert_from_pytorch_model(model, input_shape, output_dir='my-hls-test', project_name='myproject',
    fpga_part='xcku115-flvb2104-2-i', clock_period=5, io_type='io_parallel', hls_config={}):
    """
    
    Convert a Pytorch model to a hls model.
    
    Parameters
    ----------
    model : Pytorch model object.
        Model to be converted to hls model object.
    output_dir : string, optional
        Output directory to write hls codes.
    project_name : string, optional
        hls project name.
    fpga_part : string, optional
        The particular FPGA part number that you are considering.
    clock_period : int, optional
        The clock period, in ns, at which your algorithm runs.
    io_type : string, optional
        Your options are 'io_parallel' or 'io_serial' where this really 
        defines if you are pipelining your algorithm or not.
    hls_config : dict, optional
        Additional configuration dictionary for hls model.
        
    Returns
    -------
    hls_model : hls4ml model object.
        
    See Also
    --------
    hls4ml.convert_from_keras_model, hls4ml.convert_from_onnx_model
    
    Examples
    --------
    >>> import hls4ml
    >>> config = hls4ml.utils.config_from_pytorch_model(model, granularity='model')
    >>> hls_model = hls4ml.converters.convert_from_pytorch_model(model, hls_config=config)
    """
    
    config = create_vivado_config(
        output_dir=output_dir,
        project_name=project_name,
        fpga_part=fpga_part,
        clock_period=clock_period,
        io_type=io_type
    )
    
    config['PytorchModel'] = model
    config['InputShape'] = input_shape

    model_config = hls_config.get('Model', None)
    config['HLSConfig']['Model'] = _check_model_config(model_config)
    
    _check_hls_config(config, hls_config)
    
    return pytorch_to_hls(config)


def convert_from_onnx_model(model, output_dir='my-hls-test', project_name='myproject',
    fpga_part='xcku115-flvb2104-2-i', clock_period=5, io_type='io_parallel', hls_config={}):
    """
    
    Convert an ONNX model to a hls model.
    
    Parameters
    ----------
    model : ONNX model object.
        Model to be converted to hls model object.
    output_dir : string, optional
        Output directory to write hls codes.
    project_name : string, optional
        hls project name.
    fpga_part : string, optional
        The particular FPGA part number that you are considering.
    clock_period : int, optional
        The clock period, in ns, at which your algorithm runs.
    io_type : string, optional
        Your options are 'io_parallel' or 'io_serial' where this really 
        defines if you are pipelining your algorithm or not.
    hls_config : dict, optional
        Additional configuration dictionary for hls model.
        
    Returns
    -------
    hls_model : hls4ml model object.
        
    See Also
    --------
    hls4ml.convert_from_keras_model, hls4ml.convert_from_pytorch_model
    
    Examples
    --------
    >>> import hls4ml
    >>> config = hls4ml.utils.config_from_onnx_model(model, granularity='model')
    >>> hls_model = hls4ml.converters.convert_from_onnx_model(model, hls_config=config)
    """
    
    config = create_vivado_config(
        output_dir=output_dir,
        project_name=project_name,
        fpga_part=fpga_part,
        clock_period=clock_period,
        io_type=io_type
    )
    
    config['OnnxModel'] = model

    model_config = hls_config.get('Model', None)
    config['HLSConfig']['Model'] = _check_model_config(model_config)
    
    _check_hls_config(config, hls_config)
    
    return onnx_to_hls(config)


