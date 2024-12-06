"""
All information about a layer is stored in the attributes of a layer instance. This information can be properties of
a layer, like a number of hidden units in Dense layer or number of filters in a convolutional layer, but also includes
information about weight variables, output variables and all data types defined. The attribute system provides a mechanism
that ensures layers are correctly initialized, have the valid information stored and have configurable endpoints exposed.

This module contains the definitions of classes for handling attributes. The ``Attribute`` class and its subclasses provide
information about an expected attribute, but the actual value will be stored within the instance's ``attribute`` dict. This
provides an unified view (mapping) of all attributes, but for convenience there are mappings that expose only certain types
of attributes, such as types, variables, weights etc, via the ``AttributeMapping`` class.
"""

from collections.abc import MutableMapping
from numbers import Integral

from hls4ml.model.types import NamedType, Source, TensorVariable, WeightVariable
from hls4ml.utils.string_utils import convert_to_pascal_case

# region Attribute class definitions


class Attribute:
    """
    Base attribute class.

    Attribute consists of a name, the type of value it will store, the optional default if no value is specified during
    layer creation, and a flag indicating if the value can be modified by the user. This class is generally expected to
    exist only as part of the ``expected_attributes`` property of the layer class.

    Args:
        name (str): Name of the attribute
        value_type (optional): Type of the value expected to be stored in the attribute.
            If not specified, no validation of the stored value will be performed. Defaults to ``int``.
        default (optional): Default value if no value is specified during layer creation. Defaults to ``None``.
        configurable (bool, optional): Specifies if the attribute can be modified after creation. Defaults to ``False``.

    """

    def __init__(self, name, value_type: type = Integral, default=None, configurable=False, description=None):
        self.name = name
        self.value_type = value_type
        self.default = default
        self.configurable = configurable
        self.description = description

    def validate_value(self, value):
        if self.value_type is not None:
            return issubclass(type(value), self.value_type)
        else:
            return True  # Meaning we don't care

    @property
    def config_name(self):
        """Returns the name of the attribute as it will appear in the ``attribute`` dict of the layer instance.

        The format will be in pascal case, e.g., ``AttributeName`` -> ``attribute_name``.

        Returns:
            str: The pascal_case of the name of the attribute.
        """
        return convert_to_pascal_case(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Attribute):
            return NotImplemented
        return (
            self.name == other.name
            and self.value_type == other.value_type
            and self.default == other.default
            and self.configurable == other.configurable
            and self.description == other.description
        )

    def __hash__(self) -> int:
        return hash((self.name, self.value_type, self.default, self.configurable, self.description))


class ConfigurableAttribute(Attribute):
    """
    Represents a configurable attribute, i.e., the attribute whose value can be modified by the user.

    This is a convenience class. It is advised to use ``ConfigurableAttribute`` over ``Attribute(..., configurable=True)``
    when defining the expected attributes of layer classes.
    """

    def __init__(self, name, value_type=Integral, default=None, description=None):
        super().__init__(name, value_type, default, configurable=True, description=description)


class TypeAttribute(Attribute):
    """
    Represents an attribute that will store a type, i.e., an instance of ``NamedType`` or its subclasses.

    As a convention, the name of the attribute storing a type will end in ``_t``.
    """

    def __init__(self, name, default=None, configurable=True, description=None):
        if not name.endswith('_t'):
            name += '_t'
        super().__init__(name, value_type=NamedType, default=default, configurable=configurable, description=description)


class ChoiceAttribute(Attribute):
    """
    Represents an attribute whose value can be one of several predefined values.
    """

    def __init__(self, name, choices, default=None, configurable=True, description=None):
        super().__init__(name, value_type=list, default=default, configurable=configurable, description=description)
        assert len(choices) > 0
        if default is not None:
            assert default in choices
        self.choices = choices

    def validate_value(self, value):
        return value in self.choices

    def __eq__(self, other: object) -> bool:
        base_eq = super().__eq__(other)
        return base_eq and hasattr(other, 'choices') and set(self.choices) == set(other.choices)

    def __hash__(self) -> int:
        return super().__hash__() ^ hash(tuple(sorted(self.choices)))


class WeightAttribute(Attribute):
    """
    Represents an attribute that will store a weight variable.
    """

    def __init__(self, name, description=None):
        super().__init__(name, value_type=WeightVariable, default=None, configurable=False, description=description)


class CodeAttrubute(Attribute):
    """
    Represents an attribute that will store generated source code block.
    """

    def __init__(self, name, description=None):
        super().__init__(name, value_type=Source, default=None, configurable=False, description=description)


# endregion

# region Attribute mapping definitions


class AttributeDict(MutableMapping):
    """
    Class containing all attributes of a given layer.

    Instances of this class behave like a dictionary. Upon insertion, the key/value may trigger additional actions,
    such as registering variables or modifying the key name to ensure it follows the convention.

    Specific "views" (mappings) of this class can be used to filter desired attributes via the ``AttributeMapping`` class.
    """

    def __init__(self, layer):
        self.layer = layer
        self.attributes = {}
        self._expected_attributes = [a.name for a in self.layer.expected_attributes]

    def __getitem__(self, key):
        return self.attributes[key]

    def __len__(self):
        return len(self.attributes)

    def __iter__(self):
        yield from self.attributes.keys()

    def __setitem__(self, key, value):
        if isinstance(value, TensorVariable):
            self.layer.model.register_output_variable(key, value)
            self.attributes['result_t'] = value.type
            if key in self._expected_attributes and key in self.layer.outputs:
                key = 'out_' + key
        elif isinstance(value, WeightVariable):
            self.attributes[key + '_t'] = value.type

        self.attributes[key] = value

    def __delitem__(self, key):
        self.attributes.remove(key)


class AttributeMapping(MutableMapping):
    """
    Base class used to filter attributes based on their expected class.
    """

    def __init__(self, attributes, clazz):
        self.attributes = attributes
        self.clazz = clazz

    def __getitem__(self, key):
        return self.attributes[key]

    def __len__(self):
        return sum(map(lambda x: isinstance(x, self.clazz), self.attributes.values()))

    def __iter__(self):
        precision_keys = [k for k, v in self.attributes.items() if isinstance(v, self.clazz)]
        yield from precision_keys

    def __setitem__(self, key, value):
        self.attributes[key] = value

    def __delitem__(self, key):
        self.attributes.remove(key)


class WeightMapping(AttributeMapping):
    """
    Mapping that only sees ``WeightVariable`` instances (i.e., weights).
    """

    def __init__(self, attributes):
        super().__init__(attributes, WeightVariable)


class VariableMapping(AttributeMapping):
    """
    Mapping that only sees ``TensorVariable`` instances (i.e., activation tensors).
    """

    def __init__(self, attributes):
        super().__init__(attributes, TensorVariable)

    def __getitem__(self, key):
        if 'out_' + key in self.attributes:
            return self.attributes['out_' + key]
        else:
            return self.attributes[key]

    def __iter__(self):
        precision_keys = [k for k, v in self.attributes.items() if isinstance(v, self.clazz)]
        for key in precision_keys:
            if key.startswith('out_'):
                yield key[len('out_') :]
            else:
                yield key
        super().__iter__()


class TypeMapping(AttributeMapping):
    """
    Mapping that only sees ``NamedType`` instances (i.e., defined types).
    """

    def __init__(self, attributes):
        super().__init__(attributes, NamedType)


class CodeMapping(AttributeMapping):
    """
    Mapping that only sees ``Source`` instances (i.e., generated source code blocks).
    """

    def __init__(self, attributes):
        super().__init__(attributes, Source)


# endregion
