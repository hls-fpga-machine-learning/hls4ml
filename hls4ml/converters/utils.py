import math

def parse_data_format(input_shape, data_format='channels_last'):
    if input_shape[0] is None:
        # Ignore batch size
        input_shape = input_shape[1:]
    
    if data_format.lower() == 'channels_last':
        if len(input_shape) == 2: # 1D, (n_in, n_filt)
            return (input_shape[0], input_shape[1])
        elif len(input_shape) == 3: # 2D, (in_height, in_width, n_filt)
            return (input_shape[0], input_shape[1], input_shape[2])
        
    elif data_format.lower() == 'channels_first':
        if len(input_shape) == 2: # 1D, (n_filt, n_in)
            return (input_shape[1], input_shape[0])
        elif len(input_shape) == 3: # 2D, (n_filt, in_height, in_width)
            return (input_shape[1], input_shape[2], input_shape[0])
    else:
        raise Exception('Unknown data format: {}'.format(data_format))

def compute_padding_1d(pad_type, in_size, stride, filt_size):
    if pad_type.lower() == 'same':
        n_out = int(math.ceil(float(in_size) / float(stride)))
        if (in_size % stride == 0):
            pad_along_size = max(filt_size - stride, 0)
        else:
            pad_along_size = max(filt_size - (in_size % stride), 0)
        pad_left  = pad_along_size // 2
        pad_right  = pad_along_size - pad_left
    elif pad_type.lower() == 'valid':
        n_out = int(math.ceil(float(in_size - filt_size + 1) / float(stride)))
        pad_left = 0
        pad_right = 0
    else:
        raise Exception('Unknown padding type: {}'.format(pad_type))

    return (n_out, pad_left, pad_right)

def compute_padding_2d(pad_type, in_height, in_width, stride_height, stride_width, filt_height, filt_width):
    if pad_type.lower() == 'same':
        #Height
        out_height = int(math.ceil(float(in_height) / float(stride_height)))
        if (in_height % stride_height == 0):
            pad_along_height = max(filt_height - stride_height, 0)
        else:
            pad_along_height = max(filt_height - (in_height % stride_height), 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        #Width
        out_width = int(math.ceil(float(in_width) / float(stride_width)))
        if (in_width % stride_width == 0):
            pad_along_width = max(filt_width - stride_width, 0)
        else:
            pad_along_width = max(filt_width - (in_width % stride_width), 0)
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
    elif pad_type.lower() == 'valid':
        out_height = int(math.ceil(float(in_height - filt_height + 1) / float(stride_height)))
        out_width = int(math.ceil(float(in_width - filt_width + 1) / float(stride_width)))
        
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
    else:
        raise Exception('Unknown padding type: {}'.format(pad_type))

    return (out_height, out_width, pad_top, pad_bottom, pad_left, pad_right)