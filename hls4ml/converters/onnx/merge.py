from hls4ml.converters.onnx_to_hls import onnx_handler, get_onnx_attribute

merge_layers = ['Add', 'Sub', 'Mul', 'Average', 'Max', 'Min', 'Concat', 'Sum']
@onnx_handler(*merge_layers)
def parse_merge_layer(reader, node, inputs_map, input_shapes, graph, config):
    
    layer = {}
    
    layer['op'] = layer['class_name'].lower()
    
    if layer['class_name'] == 'Concatenate':
        rank = len(input_shapes[0][1:])
        if rank > 3:
            raise Exception('ERROR: Concatenation of tensors with rank > 3 is not yet supported.')
        layer['op'] = layer['class_name'].lower() + '{}d'.format(rank)
        layer['axis'] = get_onnx_attribute(node, 'axis')
    else:
        layer['class_name'] = 'Merge'
    
    layer['inputs'] = [inputs_map.get(x, x) for x in node.input]
    
    if len(layer['inputs']) > 2:
        raise Exception('ERROR: Merging more than two tensors is not yet supported.')
    
    return layer, input_shapes[0]