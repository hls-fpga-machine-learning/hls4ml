from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_attribute

@onnx_handler('Gemm')
def parse_gemm_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
   
    layer['class_name'] = 'Dense'
    layer['name'] = node.name
    
    layer['n_in'] = next((x.type.tensor_type.shape.dim[-1].dim_value for x in graph.input if x.name == node.input[0]), None)
    layer['n_out'] = next((x.type.tensor_type.shape.dim[-1].dim_value for x in graph.value_info if x.name == node.output[0]), None)
    
    output_shape = [input_shapes[0][0], layer['n_out']]
    
    tran_weight = get_onnx_attribute(node, 'transB', 0)
    reader.add_input(layer['name'], node.input, tran_weight)
    
    return layer, output_shape

#------------------Global paras for activations
activation_layers = ['Relu', 'Tanh', 'Sigmoid', 'LeakyRelu', 'ThresholdedRelu', 'HardSigmoid', 'Elu', 'Selu', 'PRelu', 'Softmax', 'Softsign', 'Softplus']

activation_map = {'Relu':'ReLU', 'Tanh':'Activation', 'Sigmoid':'Activation',
    'LeakyRelu':'LeakyReLU', 'ThresholdedRelu':'ThresholdedReLU', 'HardSigmoid':'Activation',
    'Elu':'ELU', 'Selu':'Activation', 'PRelu':'PReLU', 'Softmax':'Softmax', 'Softsign':'Activation', 'Softplus':'Activation'}
#---------

@onnx_handler(*activation_layers)
def parse_activation_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
    
    layer['name'] = node.name
    layer['class_name'] = activation_map[node.op_type]
    
    if layer['class_name'] != 'Activation':
        
        if layer['class_name'] == 'Softmax':
            layer['activation'] = 'softmax'
        
        else:
            layer['activation'] = layer['class_name']
            layer['class_name'] = 'Activation'
       
    return layer, [shape for shape in input_shapes[0]]
    
@onnx_handler('BatchNormalization')
def parse_batchnorm_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
   
    layer['class_name'] = 'BatchNormalization'
    layer['data_format'] = 'channels_first'
    layer['name'] = node.name
    
    #Other attributes
    layer['epsilon'] = get_onnx_attribute(node, 'epsilon')
    layer['momentum'] = get_onnx_attribute(node, 'momentum')
            
    reader.add_input(layer['name'], node.input)
    
    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim
        
    layer['n_in'] = layer['n_out'] = in_size
    
    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) > 2:
        layer['n_filt']= input_shapes[0][1] #Always channel first for onnx
    
    return layer, [shape for shape in input_shapes[0]]