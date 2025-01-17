from hls4ml.model.layers import Conv1D, Conv2D, SeparableConv1D, SeparableConv2D
from hls4ml.model.optimizer import OptimizerPass


class GenerateConvStreamingInstructions(OptimizerPass):
    '''Generates the instructions for streaming implementation of CNNs'''

    def match(self, node):
        is_match = (
            isinstance(node, (Conv1D, SeparableConv1D, Conv2D, SeparableConv2D))
            and node.model.config.get_config_value('IOType').lower() == 'io_stream'
            and node.get_attr('implementation').lower() == 'encoded'
        )
        return is_match

    def transform(self, model, node):
        node_class = node.__class__.__name__
        if '1D' in node_class:
            self._generate_1d_instructions(node)
        elif '2D' in node_class:
            self._generate_2d_instructions(node)
        else:
            raise Exception(f'Cannot generate instructions for node {node.name} ({node_class})')

    def _generate_1d_instructions(self, node):
        min_w, instructions = node.model.config.backend.compute_conv1d_instructions(
            node.get_input_variable().shape[0],
            node.get_input_variable().shape[1],
            node.get_attr('filt_width'),
            node.get_attr('stride_width'),
        )
        instructions_str = ','.join(str(i) for i in instructions)
        node.set_attr('min_width', min_w)
        node.set_attr('instructions', instructions_str)

    def _generate_2d_instructions(self, node):
        min_h, min_w, instructions = node.model.config.backend.compute_conv2d_instructions(
            node.get_input_variable().shape[0],
            node.get_input_variable().shape[1],
            node.get_input_variable().shape[2],
            node.get_attr('filt_height'),
            node.get_attr('stride_height'),
        )
        instructions_str = ','.join(str(i) for i in instructions)
        node.set_attr('min_height', min_h)
        node.set_attr('min_width', min_w)
        node.set_attr('instructions', instructions_str)
