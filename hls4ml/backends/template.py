from hls4ml.model.optimizer.optimizer import OptimizerPass


class Template(OptimizerPass):
    """The Template base class, should not be instantiated directly

    Args:
        name (str): Name of the template.
        layer_class (Layer or list, tuple, or aet of Layers): The Layers that this template handles.
        attribute_name (str):  The type of attribute provided
    """

    def __init__(self, name, layer_class, attribute_name):
        self.name = name
        self.layer_class = layer_class
        if not isinstance(self.layer_class, (list, tuple, set)):
            self.layer_class = [self.layer_class]
        self.attribute_name = attribute_name

    def match(self, node):
        for layer_cls in self.layer_class:
            if node.class_name == layer_cls.__name__:
                return True
        return False

    def transform(self, model, node):
        formatted_template = self.format(node)
        node.set_attr(self.attribute_name, formatted_template)
        return False

    def format(self, node):
        raise NotImplementedError

    def get_name(self):
        return self.name

    def _default_params(self, node):
        params = {}
        params.update(node.attributes)
        # Convert all bool attributes to lowercase strings
        params = {key: str(val).lower() if isinstance(val, bool) else val for key, val in params.items()}

        return params


class LayerConfigTemplate(Template):
    """Base class for layer config templates:  provides the 'config_cpp' attribute

    Args:
        layer_class (Layer or list, tuple, or set of Layers): The Layers that this template handles.
    """

    def __init__(self, layer_class):
        if isinstance(layer_class, (list, tuple, set)):
            name = '_'.join([cls.__name__.lower() for cls in layer_class])
        else:
            name = layer_class.__name__.lower()
        name += '_config_template'
        super().__init__(name, layer_class, 'config_cpp')

    def _default_config_params(self, layer):
        params = self._default_params(layer)
        params['iotype'] = layer.model.config.get_config_value('IOType')
        params['reuse'] = layer.get_attr('reuse_factor')
        params['namespace'] = layer.model.config.get_writer_config().get('Namespace', None) or 'nnet'

        return params


class FunctionCallTemplate(Template):
    """Base class for function call templates:  provides the 'function_cpp' attribute

    Args:
        layer_class (Layer or list, tuple, or set of Layers): The Layers that this template handles.
        include_header (list, tuple, or set of str, or None):  The list of needed include files
    """

    def __init__(self, layer_class, include_header=None):
        if isinstance(layer_class, (list, tuple, set)):
            name = '_'.join([cls.__name__.lower() for cls in layer_class])
        else:
            name = layer_class.__name__.lower()
        name += '_function_template'
        super().__init__(name, layer_class, 'function_cpp')
        if include_header is None:
            self.include_header = ()
        else:
            self.include_header = include_header

    def _default_function_params(self, layer):
        params = self._default_params(layer)
        params['config'] = f'config{layer.index}'
        params['input_t'] = layer.get_input_variable().type.name
        params['output_t'] = layer.get_output_variable().type.name
        params['input'] = layer.get_input_variable().name
        params['output'] = layer.get_output_variable().name

        return params

    def transform(self, model, node):
        node.set_attr('include_header', self.include_header)
        return super().transform(model, node)
