from __future__ import print_function
import numpy as np
import math
from onnx import ModelProto, GraphProto, NodeProto, TensorProto
from onnx import optimizer, helper, numpy_helper, shape_inference

from hls4ml.model import HLSModel

MAXMULT = 4096

class ONNXDataReader:
    """
    ONNX data reader to be used for extracting relevant information during conversion.
    """
    def __init__(self, model):
        self.model = model
        self.input_map = {}
        self.index_map = {
            # Dense
            'kernel' : 1,
            'bias'   : 2,
            # BatchNormalization
            'gamma'  : 1,
            'beta'   : 2,
            'moving_mean'   : 3,
            'moving_variance' : 4,
        }

    def get_weights_data(self, layer_name, var_name):
        """Extract weights data from ONNX model.
        
        Parameters
        ----------
        layer_name : string
            layer's name in the ONNX model
        var_name : string
            variable to be extracted

        Returns
        -------
        data : numpy array
            extracted weights data 
        
        """
        
        inputs = self.input_map[layer_name]
        inp_idx = self.index_map[var_name]
        
        if inp_idx >= len(inputs['inputs']):
            # Input not found, likely a bias tensor is not available
            return None

        tensor = next((x for x in self.model.graph.initializer if x.name == inputs['inputs'][inp_idx]), None)
        
        if tensor is not None:
            data = numpy_helper.to_array(tensor)
            if inputs['transpose']:
                if inputs['perm'] is not None and len(data.shape) == len(inputs['perm']):
                    data = data.transpose(inputs['perm'])
                else:
                    data = data.transpose()

        return data
    
    def add_input(self, layer_name, inputs, transpose=True, perm=None):
        self.input_map[layer_name] = { 'inputs': inputs, 'transpose': transpose, 'perm': perm }
    
####----------------------Helpers---------------------######
def sanitize_layer_name(layer):
    new_name = layer['name']
    if new_name[0].isdigit():
        new_name = layer['class_name'].lower() + new_name
    
    layer['name'] = new_name

def get_onnx_attribute(operation, name, default=None):
    attr = next((x for x in operation.attribute if x.name == name), None)
    if attr is None:
        value = default
    else:
        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode()
    return value

def get_input_shape(model, operation, input_idx=0):
    value_info_idx = next((i for i, x in enumerate(model.graph.value_info) if x.name == operation.input[input_idx]), 0)
    return [d.dim_value for d in model.graph.value_info[value_info_idx].type.tensor_type.shape.dim]

def compute_pads_1d(operation, layer):
    auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
    if auto_pad != 'NOTSET':
        if (layer['y_in'] % layer['stride'] == 0):
            pad_along_width = max(layer['y_filt'] - layer['stride'], 0)
        else:
            pad_along_width = max(layer['y_filt'] - (layer['y_in'] % layer['stride']), 0)

        pads = [pad_along_width // 2, pad_along_width - (pad_along_width // 2)]

        if auto_pad == 'SAME_UPPER':
            pads = sorted(pads)
        elif auto_pad == 'SAME_LOWER':
            pads = sorted(pads, reverse=True)
        else: # 'VALID' padding
            pads = [0, 0]
    else:
        pads = get_onnx_attribute(operation, 'pads', [0, 0])
    
    return pads

def compute_pads_2d(operation, layer):
    auto_pad = get_onnx_attribute(operation, 'auto_pad', 'NOTSET')
    if auto_pad != 'NOTSET':
        #Height
        if (layer['in_height'] % layer['stride_height'] == 0):
            pad_along_height = max(layer['filt_height'] - layer['stride_height'], 0)
        else:
            pad_along_height = max(layer['filt_height'] - (layer['in_height'] % layer['stride_height']), 0)
        pad_height = [pad_along_height // 2, pad_along_height - pad_along_height // 2]

        #Width
        if (layer['in_width'] % layer['stride_width'] == 0):
            pad_along_width = max(layer['filt_width'] - layer['stride_width'], 0)
        else:
            pad_along_width = max(layer['filt_width'] - (layer['in_width'] % layer['stride_width']), 0)
        pad_width = [pad_along_width // 2, pad_along_width - pad_along_width // 2]

        if auto_pad == 'SAME_UPPER':
            pads = [min(pad_height), min(pad_width), max(pad_height), max(pad_width)]
        elif auto_pad == 'SAME_LOWER':
            pads = [max(pad_height), max(pad_width), min(pad_height), min(pad_width)]
        else: # 'VALID' padding
            pads = [0, 0, 0, 0]
    else:
        pads = get_onnx_attribute(operation, 'pads', [0, 0, 0, 0])
    
    return pads

def _hls4ml_onnx_optimizer(graph):
    """
    Optimize onnx's model graph.
    """
    
    layer_index = 0
    initializer_list = [x for x in graph.initializer]
    
    print("Optimizing hls4ml ONNX model ...")
    for layer in graph.node:
        
        if layer_index != 0:
            input_idx = layer.input[0] #input layer index of this node, only support 1 input for now
            before_layer = [x for x in graph.node if input_idx in x.output]
            after_layer = [x for x in graph.node if layer.output[0] in x.input]
            
            if not before_layer or not after_layer:
                continue
            else:
                #Pick the first layer
                before_layer = before_layer[0]
                after_layer = after_layer[0]
            
            #-------------CASE 1--------------#
            #If there is transpose before and after the layer remove both of them
            # (tranpose) -> (current_layer) -> (transpose)
            #The tranpose layers are likely some sort of internal operation that we
            #don't need during conversion.
            if before_layer.op_type == after_layer.op_type == 'Transpose':
                graph.node.remove(before_layer)
                graph.node.remove(after_layer)
                print("Layer {} has transpose before and after it. Optimized!".format(layer.name))
        
        layer_index += 1
        
    return graph

####----------------------Layer handling---------------------######
layer_handlers = {}

def register_onnx_layer_handler(layer_name, handler_func):
    if layer_name in layer_handlers:
        raise Exception('Layer {} already registered'.format(layer_name))
    else:
        layer_handlers[layer_name] = handler_func

def get_supported_onnx_layers():
    return list(layer_handlers.keys())

def onnx_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function
    return decorator

####---------------Main processing function------------------######
def onnx_to_hls(config):
    """ Convert onnx model to hls model from configuration.
    
    Parameters
    ----------
    config: dict
        onnx configuration from yaml file or passed through API.
        
    Returns
    -------
    hls_model : hls4ml model object
        
    """

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    #Extract model architecture
    print('Interpreting Model ...')
    
    model =  onnx.load(config['OnnxModel']) if isinstance(config['OnnxModel'], str) else config['OnnxModel']
          
    #Optimizie the model graph's before conversion
    model = shape_inference.infer_shapes(model) # have to infer shapes before optimizing the model
    graph =  _hls4ml_onnx_optimizer(model.graph)
    
    #Map inputs of skipped layers
    inputs_map = {}
    reader = ONNXDataReader(model)
    
    #Input layer info (assuming that there is only one input layer)
    input_layer = {}
    input_layer['name'] = 'Input'
    input_layer['class_name'] = 'InputLayer'
    input_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    input_layer['input_shape'] = input_shape

    if len(input_layer['input_shape']) > 1:
        input_layer['input_shape'][0] = None #Firt dim is batch

    sanitize_layer_name(input_layer)
    layer_list.append(input_layer)

    # Defined supported layers and check for unsupported layer type
    skip_layers = ['Squeeze', 'Unsqueeze', 'Dropout', 'Identity', 'Flatten', 'Reshape']
    
    #Map inputs of skipped layers
    inputs_map = {}
    
    supported_layers = get_supported_onnx_layers() + skip_layers
    
    # Get input shape
    current_shape = [input_layer['input_shape']]
    print('Input shape:', current_shape[0])
    
    #Loop through layers
    layer_counter = 0
    
    #Output shape tracking
    output_shapes = {}
    output_shape = None

    print('Topology:')
    for node in graph.node:
        
        if node.op_type not in supported_layers:
            raise Exception('ERROR: Unsupported operation type: {}'.format(node.op_type))
        
        #If not the first layer then input shape is taken from last layer's output
        if layer_counter != 0:
            current_shape = [output_shape]
            
        if node.op_type in skip_layers:
            if node.op_type == 'Flatten':
                output_shape = [current_shape[0][0], np.prod(current_shape[0][1:])]
            
            else:
                #Currently supported skipped layers have only one input and output
                #Skipped layers can follow each other (e.g., Dropout -> Flatten)
                
                #Mapping inputs
                input_name = inputs_map.get(node.input[0], node.input[0])
                output_name = node.output[0]
                inputs_map[output_name] = input_name
                
                output_shape = current_shape[0]
            continue 
        
        if node.op_type in supported_layers:
            layer_counter = layer_counter + 1
        
        #Process the layer
        layer, output_shape = layer_handlers[node.op_type](reader, node, inputs_map, current_shape, graph, config)
        
        sanitize_layer_name(layer)
        print('Layer name: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], current_shape))
        layer_list.append(layer)


    #################
    ## Generate HLS
    #################

    print('Creating HLS model')
    hls_model = HLSModel(config, reader, layer_list)
    return hls_model
