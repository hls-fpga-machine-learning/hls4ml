from __future__ import print_function
import numpy as np
import os
import yaml
import sys
import torch
import pickle
import re

from hls4ml.model import HLSModel

class PyTorchModelReader(object):
    """
    Pytorch data reader to be used for extracting relevant information during conversion.
    """
    def __init__(self, config):
        self.torch_model = config['PytorchModel']
        self.state_dict = self.torch_model.state_dict()
        self.input_shape = config['InputShape']
    
    def get_weights_data(self, layer_name, var_name):
        """Get weights data from layers.
        
        The hls layer classes are based on Keras's default parameters.
        Thus, this function will also need to account for some differences
        between Keras and Pytorch terminology.
        
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
        
        data = None
        
        #Parameter mapping from pytorch to keras
        torch_paramap = {
        #Conv
        'kernel': 'weight', 
        #Batchnorm
        'gamma': 'weight',
        'beta': 'bias',
        'moving_mean':'running_mean',
        'moving_variance': 'running_var'}
            
        if var_name not in list(torch_paramap.keys()) + ['weight', 'bias']:
            raise Exception('Pytorch parameter not yet supported!')
        
        elif var_name in list(torch_paramap.keys()):
            var_name = torch_paramap[var_name]
            
        data = self.state_dict[layer_name + '.' + var_name].numpy().transpose() #Look at transpose when systhesis produce lousy results. Might need to remove it.
        
        return data
    
class PyTorchFileReader(PyTorchModelReader): #Inherit get_weights_data method
    def __init__(self, config):
        self.config = config

        if not torch.cuda.is_available():
            self.torch_model = torch.load(config['PytorchModel'], map_location=lambda storage, loc: storage)
        else:
            self.torch_model = torch.load(config['PytorchModel'])
        
        #Get input tensor's shape
        self.input_shape = config.get('InputShape')
        
        if self.input_shape == None:
            raise Exception('Must specify input shape ("InputShape") in config!')
        
        #Convert it to a list
        self.input_shape = self.input_shape.strip('(,)').split(',')
        self.input_shape = [None if n == 'None' else int(n) for n in self.input_shape]

        self.state_dict = self.torch_model.state_dict()
        
        return data

####----------------------Layer handling---------------------######
layer_handlers = {}

def register_pytorch_layer_handler(layer_name, handler_func):
    if layer_name in layer_handlers:
        raise Exception('Layer {} already registered'.format(layer_name))
    else:
        layer_handlers[layer_name] = handler_func

def get_supported_pytorch_layers():
    return list(layer_handlers.keys())

def pytorch_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function
    return decorator
####----------------------------------------------------------------

def pytorch_to_hls(config):
    """ Convert Pytorch model to hls model from configuration.
    
    Parameters
    ----------
    config: dict
        pytorch configuration from yaml file or passed through API.
        
    Returns
    -------
    hls_model : hls4ml model object
        
    """
    
    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    print('Interpreting Model ...')
    
    reader = PyTorchFileReader(config) if isinstance(config['PytorchModel'],str) else PyTorchModelReader(config)
    input_shapes = [list(reader.input_shape)]
        
    model = reader.torch_model

    #Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Flatten', 'Sequential']
    
    #All supported layers
    supported_layers = get_supported_pytorch_layers() + skip_layers
    
    #Map inputs of skipped and split (activation) layers
    inputs_map = {}

    input_layers = None
    output_layers = None
    
    layer_config = None
    
    #Output shape tracking
    output_shapes = {}
    output_shape = None
    
    #Loop through layers
    print('Topology:')
    layer_counter = 0
    
    #First add input layer
    input_layer = {}
    input_layer['name'] = 'input1'
    input_layer['class_name'] = 'InputLayer'
    input_layer['input_shape'] = input_shapes[0][1:]
    layer_list.insert(0, input_layer)
    print("Input Shape: ", input_shapes)
    
    for layer_name, pytorch_layer in model.named_modules():
        
        pytorch_class = pytorch_layer.__class__.__name__
        
        #First module is the whole model's class
        if pytorch_class == model.__class__.__name__:
            continue
        
        if pytorch_class not in supported_layers:
            raise Exception('Unsupported layer {}'.format(pytorch_class))
                
        #If not the first layer then input shape is taken from last layer's output
        if layer_counter != 0:
            input_shapes = [output_shape] #In case there are multiple inputs
        
        #Handle skipped layers
        if pytorch_class in skip_layers:
            if pytorch_class == 'Sequential': #Ignore the mother module's class name
                continue

            if pytorch_class == 'Flatten':
                output_shapes[layer_name] = [input_shapes[0][0], np.prod(input_shapes[0][1:])]
            else:
                output_shapes[layer_name] = input_shapes[0]
            continue #!!
        
        #Increment the layer counter after initial screenings
        if pytorch_class in supported_layers:
            layer_counter += 1
        
        #Process the layer
        layer, output_shape = layer_handlers[pytorch_class](pytorch_layer, layer_name, input_shapes, reader, config)

        print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
        layer_list.append(layer)
        
        assert(output_shape is not None)
        output_shapes[layer['name']] = output_shape
           

    #################
    ## Generate HLS
    #################
    
    print('Creating HLS model')
    hls_model = HLSModel(config, reader, layer_list)
    return hls_model
