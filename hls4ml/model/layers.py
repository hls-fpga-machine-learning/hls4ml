import typing

import numpy as np

from hls4ml.model.attributes import (
    Attribute,
    AttributeDict,
    ChoiceAttribute,
    CodeMapping,
    ConfigurableAttribute,
    TypeAttribute,
    TypeMapping,
    VariableMapping,
    WeightAttribute,
    WeightMapping,
)
from hls4ml.model.types import (
    CompressedWeightVariable,
    ExponentPrecisionType,
    ExponentWeightVariable,
    FixedPrecisionType,
    IntegerPrecisionType,
    NamedType,
    TensorVariable,
    WeightVariable,
    find_minimum_width,
)
from hls4ml.utils.string_utils import convert_to_snake_case


# TODO move this to some utility module
class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, owner):
        return self.func(owner)


class Layer:
    """The base class for all layers, which are the nodes in the model graph.
    Note:  they don't necessarily correspond 1:1 with the network layers.

    The expected attributes are `index`, `trace` (configurable), and `result` (type)

    Args:
        model (ModelGraph):  The ModelGraph that this Layer is part of
        name (str): The node name
        attributes (dict): Initial set of attributes required to construct the node (Layer)
        inputs (list): List of inputs to the layer
        outputs (list, optional): The optional list of named outputs of the node
    """

    _expected_attributes = [
        Attribute('index'),
        ConfigurableAttribute('trace', default=False, value_type=bool),
        TypeAttribute('result'),
    ]
    """"""

    @classproperty
    def expected_attributes(cls):
        """Returns the expected attributes of a class."""
        all_attributes = []
        for base_cls in reversed(cls.mro()):  # Iterate over all base classes in the hierarchy
            if cls == base_cls:  # Skip adding attributes from self
                continue
            if hasattr(base_cls, '_expected_attributes'):  # Only consider classes with '_expected_attributes' defined
                all_attributes.extend(base_cls._expected_attributes)
        if '_expected_attributes' in cls.__dict__:
            # Add new attributes defined in the class
            # TODO Support overriding attributes from parent class
            all_attributes.extend(cls._expected_attributes)
        return all_attributes

    def __init__(self, model, name, attributes, inputs, outputs=None):
        if name == 'input':
            raise RuntimeError(
                "No model layer should be named 'input' because that is a reserved;"
                + "layer name in ModelGraph; Please rename the layer in your model"
            )
        self.model = model
        self.name = name
        self.index = model.next_layer()
        self.inputs = inputs
        self.outputs = outputs
        if self.outputs is None:
            self.outputs = [self.name]

        self.attributes = AttributeDict(self)
        self.attributes.update(attributes)

        self.set_attr('index', self.index)

        self.weights = WeightMapping(self.attributes)
        self.variables = VariableMapping(self.attributes)
        self.types = TypeMapping(self.attributes)
        self.code = CodeMapping(self.attributes)

        self._set_accum_t()

        layer_config = self.model.config.get_layer_config(self)
        for config_key, config_value in layer_config.items():
            config_key = convert_to_snake_case(config_key)
            if config_key in self.attributes:
                print(
                    'WARNING: Config parameter "{}" overwrites an existing attribute in layer "{}" ({})'.format(
                        config_key, self.name, self.class_name
                    )
                )
            if config_key.endswith('_t') and isinstance(
                config_value, str
            ):  # TODO maybe move this to __setitem__ of AttributeDict?
                precision = self.model.config.backend.convert_precision_string(config_value)
                config_value = NamedType(self.name + '_' + config_key, precision)
            self.attributes[config_key] = config_value

        self.initialize()
        self._validate_attributes()

    @property
    def class_name(self, include_wrapped=False):
        if include_wrapped:
            return self.__class__.__name__
        else:
            if hasattr(self, '_wrapped'):
                return self.__class__.__bases__[0].__name__
            else:
                return self.__class__.__name__

    def initialize(self):
        raise NotImplementedError

    def set_attr(self, key, value):
        self.attributes[key] = value

    def get_attr(self, key, default=None):
        return self.attributes.get(key, default)

    def _validate_attributes(self):
        all_attributes = {}
        for attr in self.expected_attributes:
            all_attributes[attr.name] = attr

        # Validate existing attributes
        for attr_name, attr_value in self.attributes.items():
            exp_attr = all_attributes.pop(attr_name, None)
            if exp_attr is not None:
                if not exp_attr.validate_value(attr_value):
                    raise Exception(
                        'Unexpected value of attribute "{}" of layer "{}" ({}). Expected {}, got {} ({})'.format(
                            attr_name, self.name, self.class_name, exp_attr.value_type, type(attr_value), attr_value
                        )
                    )
            else:
                pass  # TODO layer contains attribute that is not expected. we can log this for debugging

        # If any expected attributes remain, try adding their default values
        for attr_name, attr in all_attributes.items():
            if attr.default is not None:
                if isinstance(attr, TypeAttribute):
                    self.set_attr(attr_name, self._wrap_precision_to_type(self.name + '_' + attr_name, attr.default))
                else:
                    self.set_attr(attr_name, attr.default)
            else:
                raise Exception(
                    'Attribute "{}" of layer {} ({}) not set and no default value is specified.'.format(
                        attr_name, self.name, self.class_name
                    )
                )

    def _wrap_precision_to_type(self, name, precision):
        if isinstance(precision, str):
            precision = self.convert_precision_string(precision)
        return NamedType(name=name, precision=precision)

    def _set_accum_t(self):
        has_accum_t = any(a for a in self.expected_attributes if a.name == 'accum_t' and isinstance(a, TypeAttribute))
        if has_accum_t:
            accum_t = NamedType(*reversed(self.model.config.get_precision(self, 'accum')))
            self.set_attr('accum_t', accum_t)

    def get_input_node(self, input_name=None):
        if input_name is None:
            if len(self.inputs) > 0:
                input_name = self.inputs[0]
            else:
                return None
        nodes = [node for node in self.model.graph.values() if input_name in node.outputs]
        if len(nodes) == 0:
            return None
        else:
            return nodes[0]

    def get_input_variable(self, input_name=None):
        if input_name is not None:
            return self.model.get_layer_output_variable(input_name)
        else:
            return self.model.get_layer_output_variable(self.inputs[0])

    def get_output_use_map(self):
        output_map = {}
        for output in self.outputs:
            output_map[output] = []
            for layer in self.model.get_layers():
                for inp in layer.inputs:
                    if output == inp:
                        output_map[output].append(layer)
        return output_map

    def get_output_nodes(self, output_name=None):
        output_nodes = []
        if output_name is not None:
            outputs = [output_name]
        else:
            outputs = self.outputs
        for output in outputs:
            for layer in self.model.get_layers():
                for inp in layer.inputs:
                    if output == inp:
                        output_nodes.append(layer)
        return output_nodes

    def get_output_variable(self, output_name=None):
        if output_name is not None:
            return self.variables[output_name]
        else:
            return next(iter(self.variables.values()))

    def get_weights(self, var_name=None):
        if var_name:
            return self.weights[var_name]

        return self.weights.values()

    def get_variables(self):
        return self.variables.values()

    def add_output_variable(
        self, shape, dim_names, out_name=None, var_name='layer{index}_out', type_name='layer{index}_t', precision=None
    ):
        if out_name is None:
            out_name = self.outputs[0]

        if precision is None:
            precision, _ = self.model.config.get_precision(self, var='result')

        out = TensorVariable(shape, dim_names, var_name=var_name, type_name=type_name, precision=precision, index=self.index)

        self.set_attr(out_name, out)

    def add_weights(self, quantizer=None, compression=False):
        self.add_weights_variable(
            name='weight', var_name='w{index}', data='weight', quantizer=quantizer, compression=compression
        )

    def add_bias(self, quantizer=None):
        data = self.get_attr('bias_data', None)
        precision = None
        type_name = None
        if data is None:
            data = np.zeros(self.get_output_variable().shape[-1])
            precision = IntegerPrecisionType(width=1, signed=False)
            type_name = 'bias{index}_t'
            quantizer = None  # Don't quantize non-existant bias

        self.add_weights_variable(
            name='bias', var_name='b{index}', type_name=type_name, precision=precision, data=data, quantizer=quantizer
        )

    def add_weights_variable(
        self, name, var_name=None, type_name=None, precision=None, data=None, quantizer=None, compression=False
    ):
        if var_name is None:
            var_name = name + '{index}'

        if precision is None:
            precision, _ = self.model.config.get_precision(self, var=name)
        elif type_name is None:
            # If precision is specified but no type name is given, assign a dedicated
            # type name made from variable name and layer index
            type_name = name + '{index}_t'

        if type_name is None:
            _, type_name = self.model.config.get_precision(self, var=name)

        if data is None:
            data = self.get_attr(name + '_data')
        elif isinstance(data, str):
            data = self.get_attr(data + '_data')

        data_unquantized = data
        exponent_type = False
        if quantizer is not None:
            precision = quantizer.hls_type
            type_name = name + '{index}_t'
            data = quantizer(data)
            if isinstance(quantizer.hls_type, ExponentPrecisionType):
                exponent_type = True

        if compression:
            # TODO reuse factor may not be available here
            var = CompressedWeightVariable(
                var_name,
                type_name=type_name,
                precision=precision,
                quantizer=quantizer,
                data=data,
                reuse_factor=self.get_attr('reuse_factor', 1),
                index=self.index,
            )
        elif exponent_type:
            var = ExponentWeightVariable(
                var_name, type_name=type_name, precision=precision, quantizer=quantizer, data=data, index=self.index
            )
        else:
            var = WeightVariable(
                var_name, type_name=type_name, precision=precision, quantizer=quantizer, data=data, index=self.index
            )

        var.data_unquantized = data_unquantized

        self.set_attr(name, var)

    def get_layer_precision(self):
        precision = {}
        for data_type in self.types.values():
            precision[data_type.name] = data_type
        return precision


class Input(Layer):
    def initialize(self):
        shape = self.attributes['input_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = [f'N_INPUT_{i}_{self.index}' for i in range(1, len(shape) + 1)]
        if self.index == 1:
            default_type_name = 'input_t'
        else:
            default_type_name = f'input{self.index}_t'
        type_name = self.attributes.get('type_name', default_type_name)
        precision, _ = self.model.config.get_precision(self, var='result')
        self.add_output_variable(shape, dims, var_name=self.name, type_name=type_name, precision=precision)


class Reshape(Layer):
    _expected_attributes = [
        Attribute('target_shape', value_type=typing.Sequence),
    ]

    def initialize(self):
        input_shape = self.get_input_variable(self.inputs[0]).shape
        target_shape = self.get_attr('target_shape')
        if target_shape is None:
            # need to get it from the input
            shape_node = self.get_input_node(self.inputs[1])
            # for QONNX, remove batch dimension
            if shape_node:
                target_shape = shape_node.value[1:]
            else:
                raise RuntimeError("Reshape for ONNX requires the target shape to be a second input.")

        # remove Nones -- is this ever triggered?
        if target_shape[0] is None:
            target_shape = target_shape[1:]

        # take care of -1 shapes
        shape = self._infer_output_shape(input_shape, target_shape)

        # update the target shape with chnges from above
        self.set_attr('target_shape', shape)

        dims = [f'N_SIZE_{i}_{self.index}' for i in range(len(shape))]

        self.add_output_variable(shape, dims)

    def _infer_output_shape(self, input_shape, target_shape):
        """Expand the shape that potentially includes -1 as one of the dimensions."""
        if -1 in target_shape:  # Need to infer shape for -1
            dummy_x = np.ones(input_shape)
            dummy_y = np.reshape(dummy_x, target_shape)
            return list(dummy_y.shape)
        return target_shape


class Dense(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_out'),
        WeightAttribute('weight'),
        WeightAttribute('bias'),
        TypeAttribute('weight'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        shape = self.get_input_variable().shape[:]
        shape[-1] = self.attributes['n_out']
        if len(shape) > 1:
            dims = [f'N_LAYER_{i}_{self.index}' for i in range(1, len(shape) + 1)]
        else:
            dims = [f'N_LAYER_{self.index}']
        self.add_output_variable(shape, dims)
        self.add_weights(quantizer=self.get_attr('weight_quantizer'), compression=self.model.config.get_compression(self))
        self.add_bias(quantizer=self.get_attr('bias_quantizer'))


class Conv1D(Layer):
    _expected_attributes = [
        Attribute('in_width'),
        Attribute('out_width'),
        Attribute('n_chan'),
        Attribute('n_filt'),
        Attribute('filt_width'),
        Attribute('stride_width'),
        Attribute('pad_left'),
        Attribute('pad_right'),
        WeightAttribute('weight'),
        WeightAttribute('bias'),
        TypeAttribute('weight'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_width'], self.attributes['n_filt']]
            dims = [f'N_OUTPUTS_{self.index}', f'N_FILT_{self.index}']
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_width']]
            dims = [f'N_FILT_{self.index}', f'N_OUTPUTS_{self.index}']

        self.add_output_variable(shape, dims)
        self.add_weights(quantizer=self.get_attr('weight_quantizer'))
        self.add_bias(quantizer=self.get_attr('bias_quantizer'))


class SeparableConv1D(Layer):
    _expected_attributes = [
        Attribute('in_width'),
        Attribute('out_width'),
        Attribute('n_chan'),
        Attribute('n_filt'),
        Attribute('filt_width'),
        Attribute('stride_width'),
        Attribute('pad_left'),
        Attribute('pad_right'),
        WeightAttribute('depthwise'),
        WeightAttribute('pointwise'),
        WeightAttribute('bias'),
        TypeAttribute('depthwise'),
        TypeAttribute('pointwise'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_width'], self.attributes['n_filt']]
            dims = [f'N_OUTPUTS_{self.index}', f'N_FILT_{self.index}']
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_width']]
            dims = [f'N_FILT_{self.index}', f'N_OUTPUTS_{self.index}']
        self.add_output_variable(shape, dims)

        self.add_weights_variable(name='depthwise', var_name='d{index}', quantizer=self.get_attr('depthwise_quantizer'))
        self.add_weights_variable(name='pointwise', var_name='p{index}', quantizer=self.get_attr('pointwise_quantizer'))

        zero_bias_data = np.zeros((self.attributes['n_chan'],))
        precision = IntegerPrecisionType(width=1, signed=False)
        self.add_weights_variable(name='zero_bias', var_name='z{index}', data=zero_bias_data, precision=precision)

        self.add_bias(quantizer=self.get_attr('bias_quantizer'))


class DepthwiseConv1D(Conv1D):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_width'], self.attributes['n_chan']]
            dims = [f'OUT_HEIGHT_{self.index}', f'N_CHAN_{self.index}']
        else:
            shape = [self.attributes['n_chan'], self.attributes['out_width']]
            dims = [f'N_CHAN_{self.index}', f'OUT_WIDTH_{self.index}']
        self.add_output_variable(shape, dims)

        self.add_weights_variable(
            name='weight', var_name='w{index}', data='depthwise', quantizer=self.get_attr('depthwise_quantizer')
        )

        self.add_bias(quantizer=self.get_attr('bias_quantizer'))


class Conv2D(Layer):
    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('out_height'),
        Attribute('out_width'),
        Attribute('n_chan'),
        Attribute('n_filt'),
        Attribute('filt_height'),
        Attribute('filt_width'),
        Attribute('stride_height'),
        Attribute('stride_width'),
        Attribute('pad_top'),
        Attribute('pad_bottom'),
        Attribute('pad_left'),
        Attribute('pad_right'),
        WeightAttribute('weight'),
        WeightAttribute('bias'),
        TypeAttribute('weight'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
            dims = [f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}', f'N_FILT_{self.index}']
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_height'], self.attributes['out_width']]
            dims = [f'N_FILT_{self.index}', f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}']
        self.add_output_variable(shape, dims)
        self.add_weights(quantizer=self.get_attr('weight_quantizer'))
        self.add_bias(quantizer=self.get_attr('bias_quantizer'))


class Conv2DBatchnorm(Conv2D):
    def _get_folded_weights(self):
        """
        Function to get the batchnorm folded weights.
        This function converts the weights by folding batchnorm parameters into
        the weight of QConv2D. The high-level equation:
        W_fold = gamma * W / sqrt(variance + epsilon)
        bias_fold = gamma * (bias - moving_mean) / sqrt(variance + epsilon) + beta
        """
        kernel = self.get_attr('weight_data')
        bias = self.get_attr('bias_data')
        if bias is None:
            bias = 0

        # get batchnorm weights and moving stats
        gamma = self.get_attr('gamma_data')
        beta = self.get_attr('beta_data')
        moving_mean = self.get_attr('mean_data')
        moving_variance = self.get_attr('variance_data')
        # get the inversion factor so that we replace division by multiplication
        inv = np.reciprocal(np.sqrt(moving_variance + self.get_attr('epsilon')))
        if gamma is not None:
            inv *= gamma

        # wrap conv kernel and bias with bn parameters
        folded_kernel = inv * kernel
        folded_bias = inv * (bias - moving_mean)
        if beta is not None:
            folded_bias += beta

        return [folded_kernel, folded_bias]

    def initialize(self):
        super().initialize()
        folded_weights, folded_bias = self._get_folded_weights()
        if self.model.config.is_resource_strategy(self) and self.model.config.backend.name in [
            'Vivado',
            'VivadoAccelerator',
        ]:
            self.weights['weight'].data_unquantized = np.transpose(folded_weights, axes=[3, 0, 1, 2])
            self.weights['weight'].data = self.get_attr('weight_quantizer')(self.weights['weight'].data_unquantized)

        else:
            self.weights['weight'].data_unquantized = folded_weights
            self.weights['weight'].data = self.get_attr('weight_quantizer')(folded_weights)
        self.weights['bias'].data_unquantized = folded_bias
        bias_q = self.get_attr('bias_quantizer')
        if bias_q is not None:
            self.weights['bias'].data = bias_q(folded_bias)


class SeparableConv2D(Layer):
    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('out_height'),
        Attribute('out_width'),
        Attribute('n_chan'),
        Attribute('n_filt'),
        Attribute('filt_height'),
        Attribute('filt_width'),
        Attribute('stride_height'),
        Attribute('stride_width'),
        Attribute('pad_top'),
        Attribute('pad_bottom'),
        Attribute('pad_left'),
        Attribute('pad_right'),
        WeightAttribute('depthwise'),
        WeightAttribute('pointwise'),
        WeightAttribute('bias'),
        TypeAttribute('depthwise'),
        TypeAttribute('pointwise'),
        TypeAttribute('bias'),
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
            dims = [f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}', f'N_FILT_{self.index}']
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_height'], self.attributes['out_width']]
            dims = [f'N_FILT_{self.index}', f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}']
        self.add_output_variable(shape, dims)

        self.add_weights_variable(name='depthwise', var_name='d{index}', quantizer=self.get_attr('depthwise_quantizer'))
        self.add_weights_variable(name='pointwise', var_name='p{index}', quantizer=self.get_attr('pointwise_quantizer'))

        zero_bias_data = np.zeros((self.attributes['n_chan'],))
        precision = IntegerPrecisionType(width=1, signed=False)
        self.add_weights_variable(name='zero_bias', var_name='z{index}', data=zero_bias_data, precision=precision)

        self.add_bias(quantizer=self.get_attr('bias_quantizer'))


class DepthwiseConv2D(Conv2D):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_chan']]
            dims = [f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}', f'N_CHAN_{self.index}']
        else:
            shape = [self.attributes['n_chan'], self.attributes['out_height'], self.attributes['out_width']]
            dims = [f'N_CHAN_{self.index}', f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}']
        self.add_output_variable(shape, dims)

        self.add_weights_variable(
            name='weight', var_name='w{index}', data='depthwise', quantizer=self.get_attr('depthwise_quantizer')
        )

        self.add_bias(quantizer=self.get_attr('bias_quantizer'))


class Pooling1D(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_out'),
        Attribute('n_filt'),
        Attribute('pool_width'),
        Attribute('stride_width'),
        Attribute('pad_left'),
        Attribute('pad_right'),
        Attribute('count_pad', value_type=bool, default=False),
        ChoiceAttribute('pool_op', ['Max', 'Average'], configurable=False),
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['n_out'], self.attributes['n_filt']]
            dims = [f'N_OUTPUTS_{self.index}', f'N_FILT_{self.index}']
        else:
            shape = [self.attributes['n_filt'], self.attributes['n_out']]
            dims = [f'N_FILT_{self.index}', f'N_OUTPUTS_{self.index}']
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])


class Pooling2D(Layer):
    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('out_height'),
        Attribute('out_width'),
        Attribute('n_filt'),
        Attribute('pool_height'),
        Attribute('pool_width'),
        Attribute('stride_height'),
        Attribute('stride_width'),
        Attribute('pad_top'),
        Attribute('pad_bottom'),
        Attribute('pad_left'),
        Attribute('pad_right'),
        Attribute('count_pad', value_type=bool, default=False),
        ChoiceAttribute('pool_op', ['Max', 'Average'], configurable=False),
    ]

    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
            dims = [f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}', f'N_FILT_{self.index}']
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_height'], self.attributes['out_width']]
            dims = [f'N_FILT_{self.index}', f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}']
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])


class GlobalPooling1D(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_filt'),
        ChoiceAttribute('pool_op', ['Max', 'Average'], configurable=False),
    ]

    def initialize(self):
        shape = [self.attributes['n_filt']]
        dims = [f'N_FILT_{self.index}']
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0].replace('Global', ''))


class GlobalPooling2D(Layer):
    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('n_filt'),
        ChoiceAttribute('pool_op', ['Max', 'Average'], configurable=False),
    ]

    def initialize(self):
        shape = [self.attributes['n_filt']]
        dims = [f'N_FILT_{self.index}']
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0].replace('Global', ''))


class ZeroPadding1D(Layer):
    _expected_attributes = [
        Attribute('in_width'),
        Attribute('out_width'),
        Attribute('n_chan'),
        Attribute('pad_left'),
        Attribute('pad_right'),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_width'], self.attributes['n_chan']]
            dims = [f'OUT_WIDTH_{self.index}', f'N_CHAN_{self.index}']
        else:
            shape = [self.attributes['n_chan'], self.attributes['out_width']]
            dims = [f'N_CHAN_{self.index}', f'OUT_WIDTH_{self.index}']
        self.add_output_variable(shape, dims, precision=inp.type.precision)


class ZeroPadding2D(Layer):
    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('out_height'),
        Attribute('out_width'),
        Attribute('n_chan'),
        Attribute('pad_top'),
        Attribute('pad_bottom'),
        Attribute('pad_left'),
        Attribute('pad_right'),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_chan']]
            dims = [f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}', f'N_CHAN_{self.index}']
        else:
            shape = [self.attributes['n_chan'], self.attributes['out_height'], self.attributes['out_width']]
            dims = [f'N_CHAN_{self.index}', f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}']
        self.add_output_variable(shape, dims, precision=inp.type.precision)


class Activation(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('activation', value_type=str),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)
        self.set_attr('n_in', self.get_input_variable().size())


class ParametrizedActivation(Activation):
    def _get_act_function_name(self):
        act = self.get_attr('activation').lower()
        if act == 'leakyrelu':
            return 'leaky_relu'
        elif act == 'thresholdedrelu':
            return 'thresholded_relu'
        else:
            return act  # ELU activation


class HardActivation(Activation):
    '''
    Implements the hard sigmoid and tan function in keras and qkeras
    (Default parameters in qkeras are different, so should be configured)
    The hard sigmoid unction is clip(slope * x + shift, 0, 1), and the
    hard tanh function is 2 * hard_sigmoid - 1
    '''

    _expected_attributes = [
        Attribute('slope', value_type=float, default=0.2, configurable=False),
        Attribute('shift', value_type=float, default=0.5, configurable=False),
        TypeAttribute('slope_t'),
        TypeAttribute('shift_t'),
    ]

    def initialize(self):
        super().initialize()
        slope_prec = self.get_attr('slope_prec', FixedPrecisionType(width=16, integer=0, signed=False))
        shift_prec = self.get_attr('shift_prec', FixedPrecisionType(width=1, integer=0, signed=False))
        index = self.get_attr('index')
        slope_t = NamedType(f'slope{index}_t', precision=slope_prec)
        shift_t = NamedType(f'shift{index}_t', precision=shift_prec)
        self.set_attr('slope_t', slope_t)
        self.set_attr('shift_t', shift_t)


class PReLU(Activation):
    def initialize(self):
        super().initialize()
        self.add_weights_variable(name='alpha', var_name='a{index}')


class Softmax(Activation):
    def initialize(self):
        super().initialize()


class TernaryTanh(Activation):
    def initialize(self):
        super().initialize()


class BatchNormalization(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_filt', default=0),
        WeightAttribute('scale'),
        WeightAttribute('bias'),
        TypeAttribute('scale'),
        TypeAttribute('bias'),
        Attribute('use_gamma', value_type=bool, default=True),
        Attribute('use_beta', value_type=bool, default=True),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)

        gamma = self.get_attr('gamma_data')
        beta = self.get_attr('beta_data')
        mean = self.get_attr('mean_data')
        var = self.get_attr('variance_data')

        scale = gamma / np.sqrt(var + self.get_attr('epsilon'))
        bias = beta - scale * mean

        self.add_weights_variable(name='scale', var_name='s{index}', data=scale)
        self.add_weights_variable(name='bias', var_name='b{index}', data=bias)


class Merge(Layer):
    def initialize(self):
        assert len(self.inputs) == 2
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        if np.prod(inp2.shape) > np.prod(inp1.shape):
            shape = inp2.shape.copy()
            dims = inp2.dim_names.copy()
        else:
            shape = inp1.shape.copy()
            dims = inp1.dim_names.copy()
        self.add_output_variable(shape, dims)


class Dot(Merge):
    def initialize(self):
        assert len(self.inputs) == 2
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        assert inp1.shape == inp2.shape
        if len(inp1.shape) > 1:
            raise Exception('ERROR: Dot of tensors with rank > 1 is not yet supported.')

        self.add_output_variable(shape=[1], dim_names=[f'OUT_DOT_{self.index}'])


class Concatenate(Merge):
    def initialize(self):
        assert len(self.inputs) == 2
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        axis = self.attributes['axis']
        if axis > 0:
            axis -= 1
        shape = inp1.shape[:]
        shape[axis] += inp2.shape[axis]
        rank = len(shape)
        if rank > 1:
            dims = [f'OUT_CONCAT_{i}_{self.index}' for i in range(rank)]
        else:
            dims = [f'OUT_CONCAT_{self.index}']
        self.add_output_variable(shape, dims)


class BiasAdd(Merge):  # TensorFlow's operator that gets merged into Dense/Conv
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        shape = inp.shape
        dims = inp.dim_names
        self.add_bias()
        self.add_output_variable(shape, dims)


class Resize(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        if len(inp.shape) == 2:  # 1D -> width + chan
            shape = [self.get_attr('out_width'), self.get_attr('n_chan')]
            dims = [f'OUT_WIDTH_{self.index}', f'N_CHAN_{self.index}']
        elif len(inp.shape) == 3:  # 2D -> height + width + chan
            shape = [self.get_attr('out_height'), self.get_attr('out_width'), self.get_attr('n_chan')]
            dims = [f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}', f'N_CHAN_{self.index}']
        self.add_output_variable(shape, dims, precision=inp.type.precision)


class Transpose(Layer):
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        perm = self.get_attr('perm')
        self.set_attr('dim', f'{len(inp.shape)}d')

        if len(perm) > 3:
            raise Exception('ERROR: Transpose of tensors with rank > 3 is not yet supported.')

        # ONNX double transpose specific, sometimes ONNX injects
        # useless double transpose layers when converting
        # from other frameworks
        if len(perm) == 1:
            shape = inp.shape  # dummy shape
            dims = ['DUMMY']  # dummy dims
            self.set_attr('perm', [0])
        else:
            shape = [inp.shape[i] for i in perm]

        self.set_attr('perm_str', ','.join([str(i) for i in perm]))

        if len(shape) == 2:
            self.set_attr('perm_str', ','.join(['0'] + [str(i + 1) for i in perm]))
            dims = [f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}']
            self.set_attr('depth', 1)
            self.set_attr('height', inp.shape[0])
            self.set_attr('width', inp.shape[1])
        elif len(shape) > 2:
            dims = [f'OUT_DEPTH_{self.index}', f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}']
            self.set_attr('depth', inp.shape[0])
            self.set_attr('height', inp.shape[1])
            self.set_attr('width', inp.shape[2])
        self.add_output_variable(shape, dims, precision=inp.type.precision)


class Embedding(Layer):
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_out'),
        Attribute('vocab_size'),
        WeightAttribute('embeddings'),
        TypeAttribute('embeddings'),
    ]

    def initialize(self):
        shape = self.get_input_variable().shape[:]
        shape += [self.attributes['n_out']]
        if len(shape) > 1:
            dims = [f'N_LAYER_{i}_{self.index}' for i in range(1, len(shape) + 1)]
        else:
            dims = [f'N_LAYER_{self.index}']
        self.add_output_variable(shape, dims)

        self.add_weights_variable(name='embeddings', var_name='e{index}')


class SimpleRNN(Layer):
    _expected_attributes = [
        Attribute('n_out'),
        Attribute('activation', value_type=str),
        Attribute('return_sequences', value_type=bool, default=False),
        Attribute('return_state', value_type=bool, default=False),
        ChoiceAttribute('direction', ['forward', 'backward'], default='forward'),
        WeightAttribute('weight'),
        WeightAttribute('bias'),
        WeightAttribute('recurrent_weight'),
        TypeAttribute('weight'),
        TypeAttribute('bias'),
        TypeAttribute('recurrent_weight'),
    ]

    def initialize(self):
        if self.attributes['return_sequences']:
            shape = [self.attributes['n_timesteps'], self.attributes['n_out']]
            dims = [f'N_TIME_STEPS_{self.index}', f'N_OUT_{self.index}']
        else:
            shape = [self.attributes['n_out']]
            dims = [f'N_OUT_{self.index}']

        self.add_output_variable(shape, dims)

        if self.attributes['return_state']:
            state_shape = [self.attributes['n_out']]
            state_dims = [f'N_OUT_{self.index}']
            self.add_output_variable(
                state_shape, state_dims, out_name=self.outputs[1], var_name='layer{index}_h', type_name='layer{index}_h_t'
            )
            self.add_output_variable(
                state_shape, state_dims, out_name=self.outputs[2], var_name='layer{index}_c', type_name='layer{index}_c_t'
            )

        # weights
        self.add_weights()

        # recurrent weights
        self.add_weights_variable(name='recurrent_weight', var_name='wr{index}')

        # biases
        self.add_weights_variable(name='bias', var_name='b{index}')


class LSTM(Layer):
    _expected_attributes = [
        Attribute('n_out'),
        Attribute('activation', value_type=str),
        Attribute('recurrent_activation', value_type=str),
        Attribute('return_sequences', value_type=bool, default=False),
        Attribute('return_state', value_type=bool, default=False),
        ChoiceAttribute('direction', ['forward', 'backward'], default='forward'),
        Attribute('time_major', value_type=bool, default=False),
        WeightAttribute('weight'),
        WeightAttribute('bias'),
        WeightAttribute('recurrent_weight'),
        WeightAttribute('recurrent_bias'),
        TypeAttribute('weight'),
        TypeAttribute('bias'),
        TypeAttribute('recurrent_weight'),
        TypeAttribute('recurrent_bias'),
    ]

    def initialize(self):
        if self.attributes['return_sequences']:
            shape = [self.attributes['n_timesteps'], self.attributes['n_out']]
            dims = [f'N_TIME_STEPS_{self.index}', f'N_OUT_{self.index}']
        else:
            shape = [self.attributes['n_out']]
            dims = [f'N_OUT_{self.index}']

        self.add_output_variable(shape, dims)

        if self.attributes['return_state']:
            state_shape = [self.attributes['n_out']]
            state_dims = [f'N_OUT_{self.index}']
            self.add_output_variable(
                state_shape, state_dims, out_name=self.outputs[1], var_name='layer{index}_h', type_name='layer{index}_h_t'
            )
            self.add_output_variable(
                state_shape, state_dims, out_name=self.outputs[2], var_name='layer{index}_c', type_name='layer{index}_c_t'
            )

        # weights
        self.add_weights()

        # recurrent weights
        recurrent_weight = self.get_attr('recurrent_weight_data')
        self.add_weights_variable(name='recurrent_weight', var_name='wr{index}', data=recurrent_weight)

        # biases
        self.add_weights_variable(name='bias', var_name='b{index}')

        recurrent_bias = np.zeros(recurrent_weight.shape[1])
        self.add_weights_variable(name='recurrent_bias', var_name='br{index}', data=recurrent_bias)


class GRU(Layer):
    _expected_attributes = [
        Attribute('n_out'),
        Attribute('activation', value_type=str),
        Attribute('recurrent_activation', value_type=str),
        Attribute('return_sequences', value_type=bool, default=False),
        Attribute('return_state', value_type=bool, default=False),
        ChoiceAttribute('direction', ['forward', 'backward'], default='forward'),
        Attribute('time_major', value_type=bool, default=False),
        ChoiceAttribute('apply_reset_gate', ['before', 'after'], default='after'),
        WeightAttribute('weight'),
        WeightAttribute('bias'),
        WeightAttribute('recurrent_weight'),
        WeightAttribute('recurrent_bias'),
        TypeAttribute('weight'),
        TypeAttribute('bias'),
        TypeAttribute('recurrent_weight'),
        TypeAttribute('recurrent_bias'),
    ]

    def initialize(self):
        if self.attributes['return_sequences']:
            shape = [self.attributes['n_timesteps'], self.attributes['n_out']]
            dims = [f'N_TIME_STEPS_{self.index}', f'N_OUT_{self.index}']
        else:
            shape = [self.attributes['n_out']]
            dims = [f'N_OUT_{self.index}']

        self.add_output_variable(shape, dims)

        if self.attributes['return_state']:
            state_shape = [self.attributes['n_out']]
            state_dims = [f'N_OUT_{self.index}']
            self.add_output_variable(
                state_shape, state_dims, out_name=self.outputs[1], var_name='layer{index}_h', type_name='layer{index}_h_t'
            )
            self.add_output_variable(
                state_shape, state_dims, out_name=self.outputs[2], var_name='layer{index}_c', type_name='layer{index}_c_t'
            )

        # weights
        self.add_weights()

        # recurrent weights
        self.add_weights_variable(name='recurrent_weight', var_name='wr{index}')

        # biases
        self.add_weights_variable(name='bias', var_name='b{index}')
        self.add_weights_variable(name='recurrent_bias', var_name='br{index}')


class GarNet(Layer):
    ref_impl = False

    def initialize(self):
        reuse_factor = self.model.config.get_reuse_factor(self)
        if self.attributes['n_vertices'] % reuse_factor != 0:
            raise Exception(
                'GarNet vertex loop has no bound check;'
                f'number of vertices must be divisible by the reuse factor ({reuse_factor}).'
            )

        self._initialize_transforms()

        if self.attributes['collapse']:
            shape = [self._output_features]
            dims = [f'OUT_FEATURES_{self.index}']
        else:
            shape = [self.attributes['n_vertices'], self._output_features]
            dims = [f'VERTICES_{self.index}', f'OUT_FEATURES_{self.index}']

        self.add_output_variable(shape, dims)

    def _initialize_transforms(self):
        n_propagate = self.attributes['n_propagate']
        n_aggregators = self.attributes['n_aggregators']
        n_out_features = self.attributes['n_out_features']

        if self.ref_impl:
            weights_source = [
                ('input_transform', 'FLR', 'kernel'),
                ('input_transform', 'FLR', 'bias'),
                ('aggregator_distance', 'S', 'kernel'),
                ('aggregator_distance', 'S', 'bias'),
                ('output_transform', 'Fout', 'kernel'),
                ('output_transform', 'Fout', 'bias'),
            ]

        else:
            quantize = self.get_attr('quantizer') is not None
            kernel, bias = self._make_input_transform_weights(n_propagate, n_aggregators, n_out_features, quantize=quantize)

            self._add_variable(
                'input_transform_weights', 'input_transform_w{index}', kernel, frac_width=10, quantize=quantize
            )
            self._add_variable('input_transform_biases', 'input_transform_b{index}', bias, frac_width=10, quantize=quantize)
            # dummy
            self.add_weights_variable(name='output_transform_weights', var_name='output_transform_w{index}', data=np.ones(1))

            weights_source = [
                ('aggregator_distance', 'S', 'kernel'),
                ('aggregator_distance', 'S', 'bias'),
                ('output_transform', 'Fout', 'bias'),
            ]

        for op_name, lname, wtype in weights_source:
            data = self.get_attr(f'{lname}_{wtype}_data')
            if wtype == 'kernel':
                data = data.transpose((1, 0))
                vtype = 'weights'
            else:
                vtype = 'biases'

            name = f'{op_name}_{vtype}'
            var_name = f'{op_name}_{vtype[0]}{{index}}'

            self._add_variable(name, var_name, data, frac_width=10, quantize=False)

        self._output_features = self.attributes['n_out_features']

    def _make_input_transform_weights(self, n_propagate, n_aggregators, n_out_features, quantize=False, sublayer=''):
        # Due to linearity of the input transform, input weights and biases can be contracted away at conversion time
        output_transform_kernel = self.get_attr(
            f'Fout{sublayer}_kernel_data'
        )  # [(n_aggregators, n_propagate), n_out_features]
        output_transform_kernel = output_transform_kernel.reshape((n_aggregators, n_propagate, n_out_features))
        if quantize:
            output_transform_kernel = self.get_attr('quantizer')(output_transform_kernel)

        input_transform_kernel = self.get_attr(f'FLR{sublayer}_kernel_data')  # [n_in_features, n_propagate]
        if quantize:
            input_transform_kernel = self.get_attr('quantizer')(input_transform_kernel)
        data = np.dot(input_transform_kernel, output_transform_kernel)  # [n_in_features, n_aggregators, n_out_features]
        kernel = data.transpose((2, 1, 0))

        input_transform_bias = self.get_attr(f'FLR{sublayer}_bias_data')  # [n_propagate]
        if quantize:
            input_transform_bias = self.get_attr('quantizer')(input_transform_bias)
        data = np.dot(input_transform_bias, output_transform_kernel)  # [n_aggregators, n_out_features]
        bias = data.transpose((1, 0))

        return kernel, bias

    def _add_variable(self, name, var_name, data, frac_width=10, quantize=False):
        # Wrapper for add_weights_variable with precision determination from data

        # automatically make the variable unsigned if data are all positive
        signed = np.amin(data) < 0.0

        int_width = find_minimum_width(data, signed=signed)

        if quantize:
            precision = IntegerPrecisionType(width=int_width, signed=signed)
        else:
            width = int_width + frac_width
            precision = FixedPrecisionType(
                width=width, integer=int_width, signed=signed, rounding_mode='AP_RND', saturation_mode='AP_SAT'
            )

        self.add_weights_variable(name=name, var_name=var_name, data=data, precision=precision)


class GarNetStack(GarNet):
    def _initialize_transforms(self):
        self._sublayer_weights = []

        quantize = self.get_attr('quantizer') is not None

        for il in range(self.attributes['n_sublayers']):
            sublayer_weights = {}

            n_aggregators = self.attributes['n_aggregators'][il]
            n_out_features = self.attributes['n_out_features'][il]
            n_propagate = self.attributes['n_propagate'][il]

            kernel, bias = self._make_input_transform_weights(
                n_propagate, n_aggregators, n_out_features, quantize=quantize, sublayer=il
            )

            name = f'input_transform_{il}_weights'
            self._add_variable(name, f'input_transform_{il}_w{{index}}', kernel, frac_width=10, quantize=quantize)
            sublayer_weights['input_transform_weights'] = self.weights[name]

            name = f'input_transform_{il}_biases'
            self._add_variable(name, f'input_transform_{il}_b{{index}}', bias, frac_width=10, quantize=quantize)
            sublayer_weights['input_transform_biases'] = self.weights[name]

            weights_source = [
                ('aggregator_distance', f'S{il}', 'kernel'),
                ('aggregator_distance', f'S{il}', 'bias'),
                ('output_transform', f'Fout{il}', 'bias'),
            ]

            for op_name, lname, wtype in weights_source:
                data = self.get_attr(f'{lname}_{wtype}_data')
                if wtype == 'kernel':
                    data = data.transpose((1, 0))
                    vtype = 'weights'
                else:
                    vtype = 'biases'

                name = f'{op_name}_{il}_{vtype}'
                var_name = f'{op_name}_{il}_{vtype[0]}{{index}}'

                self._add_variable(name, var_name, data, frac_width=10, quantize=False)
                sublayer_weights[f'{op_name}_{vtype}'] = self.weights[name]

            self._sublayer_weights.append(sublayer_weights)

        self._output_features = self.attributes['n_out_features'][-1]


class LayerGroup(Layer):
    _expected_attributes = [
        Attribute('layer_list', value_type=list),
        Attribute('input_layers', value_type=list),
        Attribute('output_layers', value_type=list),
        Attribute('data_reader', value_type=object),
        Attribute('output_shape', value_type=list),
    ]

    def initialize(self):
        shape = self.get_attr('output_shape')
        if shape[0] is None:
            shape.pop(0)
        dims = [f'N_INPUT_{self.index}_{i+1}' for i in range(len(shape))]

        self.add_output_variable(shape, dims)


layer_map = {
    'Input': Input,
    'InputLayer': Input,
    'Activation': Activation,
    'QActivation': Activation,
    'LeakyReLU': ParametrizedActivation,
    'ThresholdedReLU': ParametrizedActivation,
    'ELU': ParametrizedActivation,
    'PReLU': PReLU,
    'Softmax': Softmax,
    'TernaryTanh': TernaryTanh,
    'HardActivation': HardActivation,
    'Reshape': Reshape,
    'Dense': Dense,
    'BinaryDense': Dense,
    'TernaryDense': Dense,
    'QDense': Dense,
    'Conv1D': Conv1D,
    'QConv1D': Conv1D,
    'Conv2D': Conv2D,
    'BinaryConv2D': Conv2D,
    'QConv2D': Conv2D,
    'QConv2DBatchnorm': Conv2DBatchnorm,
    'SeparableConv1D': SeparableConv1D,
    'DepthwiseConv1D': DepthwiseConv1D,
    'SeparableConv2D': SeparableConv2D,
    'DepthwiseConv2D': DepthwiseConv2D,
    'QDepthwiseConv2D': DepthwiseConv2D,
    'BatchNormalization': BatchNormalization,
    'QBatchNormalization': BatchNormalization,
    'MaxPooling1D': Pooling1D,
    'AveragePooling1D': Pooling1D,
    'MaxPooling2D': Pooling2D,
    'AveragePooling2D': Pooling2D,
    'GlobalMaxPooling1D': GlobalPooling1D,
    'GlobalAveragePooling1D': GlobalPooling1D,
    'GlobalMaxPooling2D': GlobalPooling2D,
    'GlobalAveragePooling2D': GlobalPooling2D,
    'ZeroPadding1D': ZeroPadding1D,
    'ZeroPadding2D': ZeroPadding2D,
    'Merge': Merge,
    'Dot': Dot,
    'Concatenate': Concatenate,
    'Resize': Resize,
    'UpSampling1D': Resize,
    'UpSampling2D': Resize,
    'Transpose': Transpose,
    'Embedding': Embedding,
    'SimpleRNN': SimpleRNN,
    'LSTM': LSTM,
    'GRU': GRU,
    'GarNet': GarNet,
    'GarNetStack': GarNetStack,
    'LayerGroup': LayerGroup,
    # TensorFlow-specific layers:
    'BiasAdd': BiasAdd,
}


def register_layer(name, clazz):
    global layer_map
    layer_map[name] = clazz
