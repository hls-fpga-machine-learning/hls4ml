import math
from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_attribute, compute_pads_1d, compute_pads_2d

pool_operations = ['AveragePool', 'MaxPool']
@onnx_handler(*pool_operations)
def parse_pool_layer(reader, node, inputs_map, input_shapes, graph, config):
            
    info = layer['class_name'].replace('Pool', '')
    strides = get_onnx_attribute(node, 'strides')
    kernel_shape = get_onnx_attribute(node, 'kernel_shape')
    
    if len(current_shape) == 3: # 1D
        layer['class_name'] = info + 'Pooling1D'
        layer['stride'] = strides[0]
        layer['pool_size'] = layer['y_filt'] = kernel_shape[0]
        
        #Padding
        pads = compute_pads_1d(node, layer)
        layer['pad_left'] = pads[0]
        layer['pad_right'] = pads[1]

        if all(x == 0 for x in pads): # No padding, i.e., 'VALID' padding
            layer['padding'] = 'valid'
        else:
            layer['padding'] = 'same'
            
        (layer['out_width'],_,_) = compute_padding_1d(layer['padding'],
                                                      layer['in_width'],
                                                      layer['stride_width'],
                                                      layer['filt_width'])

        output_shape = [input_shapes[0][0], layer['n_filt'], layer['n_out']]
    
    elif len(current_shape) == 4: # 2D
        layer['class_name'] = info + 'Pooling2D'

        layer['n_filt'] = input_shapes[0][1]
        layer['in_height'] = input_shapes[0][2]
        layer['in_width'] = input_shapes[0][3]

        layer['stride_height'] = strides[0]
        layer['stride_width'] = strides[1]
        layer['pool_height'] = layer['filt_height'] = kernel_shape[0]
        layer['pool_width'] = layer['filt_width'] = kernel_shape[1]
        
        pads = compute_pads_2d(node, layer)
        layer['pad_top'] = pads[0]
        layer['pad_bottom'] = pads[2]
        layer['pad_left'] = pads[1]
        layer['pad_right'] = pads[3]

        if all(x == 0 for x in pads): # No padding, i.e., 'VALID' padding in Keras/Tensorflow
            layer['padding'] = 'valid'
        else: #Only 'valid' and 'same' padding are available in Keras
            layer['padding'] = 'same'
            
        (layer['out_height'], layer['out_width'],_,_,_,_) = compute_padding_2d(layer['padding'],
                                                                               layer['in_height'],
                                                                               layer['in_width'],
                                                                               layer['stride_height'],
                                                                               layer['stride_width'],
                                                                               layer['filt_height'],
                                                                               layer['filt_width'])
        
        output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]
        
        return layer, output_shape
