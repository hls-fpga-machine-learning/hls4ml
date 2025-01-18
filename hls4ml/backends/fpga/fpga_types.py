import numpy as np

from hls4ml.model.types import (
    CompressedType,
    ExponentPrecisionType,
    ExponentType,
    FixedPrecisionType,
    IntegerPrecisionType,
    NamedType,
    PackedType,
    XnorPrecisionType,
)

# region Precision types


class PrecisionDefinition:
    def definition_cpp(self):
        raise NotImplementedError


class APIntegerPrecisionDefinition(PrecisionDefinition):
    def definition_cpp(self):
        typestring = 'ap_{signed}int<{width}>'.format(signed='u' if not self.signed else '', width=self.width)
        return typestring


class APFixedPrecisionDefinition(PrecisionDefinition):
    def _rounding_mode_cpp(self, mode):
        if mode is not None:
            return 'AP_' + str(mode)

    def _saturation_mode_cpp(self, mode):
        if mode is not None:
            return 'AP_' + str(mode)

    def definition_cpp(self):
        args = [
            self.width,
            self.integer,
            self._rounding_mode_cpp(self.rounding_mode),
            self._saturation_mode_cpp(self.saturation_mode),
            self.saturation_bits,
        ]
        if args[2] == 'AP_TRN' and args[3] == 'AP_WRAP' and args[4] == 0:
            # This is the default, so we won't write the full definition for brevity
            args[2] = args[3] = args[4] = None

        args = ','.join([str(arg) for arg in args if arg is not None])
        typestring = 'ap_{signed}fixed<{args}>'.format(signed='u' if not self.signed else '', args=args)
        return typestring


class ACIntegerPrecisionDefinition(PrecisionDefinition):
    def definition_cpp(self):
        typestring = f'ac_int<{self.width}, {str(self.signed).lower()}>'
        return typestring


class ACFixedPrecisionDefinition(PrecisionDefinition):
    def _rounding_mode_cpp(self, mode):
        if mode is not None:
            return 'AC_' + str(mode)

    def _saturation_mode_cpp(self, mode):
        if mode is not None:
            return 'AC_' + str(mode)

    def definition_cpp(self):
        args = [
            self.width,
            self.integer,
            str(self.signed).lower(),
            self._rounding_mode_cpp(self.rounding_mode),
            self._saturation_mode_cpp(self.saturation_mode),
            self.saturation_bits,
        ]
        if args[3] == 'AC_TRN' and args[4] == 'AC_WRAP':
            # This is the default, so we won't write the full definition for brevity
            args[3] = args[4] = None
        if args[5] > 0:
            print(
                f'WARNING: Invalid setting of saturation bits ({args[5]}) for ac_fixed type, only 0 is allowed.'
                'Ignoring set value.'
            )
            args[5] = None

        args = ','.join([str(arg) for arg in args[:5] if arg is not None])
        typestring = f'ac_fixed<{args}>'
        return typestring


class PrecisionConverter:
    def convert(self, precision_type):
        raise NotImplementedError


class FixedPrecisionConverter(PrecisionConverter):
    def __init__(self, type_map, prefix):
        self.type_map = type_map
        self.prefix = prefix

    def convert(self, precision_type):
        type_cls = type(precision_type)
        type_cls_name = type_cls.__name__
        type_cls_fqn = type_cls.__module__ + '.' + type_cls.__qualname__

        # If the type is already converted, do nothing
        if type_cls_name.startswith(self.prefix):
            return precision_type

        definition_cls = self.type_map.get(type_cls, None)

        if definition_cls is not None:
            precision_type.__class__ = type(
                self.prefix + type_cls_name, (type_cls, definition_cls), {'_wrapped': type_cls_fqn}
            )
            return precision_type
        else:
            raise Exception(f'Cannot convert precision type to {self.prefix}: {precision_type.__class__.__name__}')


class APTypeConverter(FixedPrecisionConverter):
    def __init__(self):
        super().__init__(
            type_map={
                FixedPrecisionType: APFixedPrecisionDefinition,
                IntegerPrecisionType: APIntegerPrecisionDefinition,
                ExponentPrecisionType: APIntegerPrecisionDefinition,
                XnorPrecisionType: APIntegerPrecisionDefinition,
            },
            prefix='AP',
        )


class ACTypeConverter(FixedPrecisionConverter):
    def __init__(self):
        super().__init__(
            type_map={
                FixedPrecisionType: ACFixedPrecisionDefinition,
                IntegerPrecisionType: ACIntegerPrecisionDefinition,
                ExponentPrecisionType: ACIntegerPrecisionDefinition,
                XnorPrecisionType: ACIntegerPrecisionDefinition,
            },
            prefix='AC',
        )


# endregion

# region Data types


class TypeDefinition:
    def definition_cpp(self):
        raise NotImplementedError


class TypePrecisionConverter:
    def convert_precision(self, precision_converter):
        self.precision = precision_converter.convert(self.precision)


class NamedTypeConverter(TypeDefinition, TypePrecisionConverter):
    def definition_cpp(self):
        return f'typedef {self.precision.definition_cpp()} {self.name};\n'


class CompressedTypeConverter(TypeDefinition, TypePrecisionConverter):
    def definition_cpp(self):
        cpp_fmt = 'typedef struct {name} {{' '{index} row_index;' '{index} col_index;' '{precision} weight; }} {name};\n'
        return cpp_fmt.format(name=self.name, index=self.index_precision, precision=self.precision.definition_cpp())

    def convert_precision(self, precision_converter):
        super().convert_precision(precision_converter)
        self.index_precision = precision_converter.convert(self.index_precision)


class ExponentTypeConverter(TypeDefinition, TypePrecisionConverter):
    def definition_cpp(self):
        cpp_fmt = 'typedef struct {name} {{' '{sign} sign;' '{precision} weight; }} {name};\n'
        return cpp_fmt.format(name=self.name, precision=self.precision.definition_cpp(), sign=self.sign.definition_cpp())

    def convert_precision(self, precision_converter):
        super().convert_precision(precision_converter)
        self.sign = precision_converter.convert(self.sign)


class PackedTypeConverter(TypeDefinition, TypePrecisionConverter):
    def definition_cpp(self):
        n_elem_expr = '/' if self.unpack else '*'
        return 'typedef nnet::array<{precision}, {n_elem}> {name};\n'.format(
            name=self.name,
            precision=self.precision.definition_cpp(),
            n_elem=str(self.n_elem) + n_elem_expr + str(self.n_pack),
        )


class HLSTypeConverter:
    def __init__(self, precision_converter):
        self.precision_converter = precision_converter
        self.type_map = {
            NamedType: NamedTypeConverter,
            CompressedType: CompressedTypeConverter,
            ExponentType: ExponentTypeConverter,
            PackedType: PackedTypeConverter,
        }

    def convert(self, atype):
        type_cls = type(atype)
        type_cls_name = type_cls.__name__
        type_cls_fqn = type_cls.__module__ + '.' + type_cls.__qualname__

        # If the type is already converted, do nothing
        if type_cls_name.startswith('HLS'):
            return atype

        conversion_cls = self.type_map.get(type_cls, None)

        if conversion_cls is not None:
            atype.__class__ = type('HLS' + type_cls_name, (type_cls, conversion_cls), {'_wrapped': type_cls_fqn})
            atype.convert_precision(self.precision_converter)
            return atype
        else:
            raise Exception(f'Cannot convert type: {atype.__class__.__name__}')


# endregion

# region Variables


class VariableDefinition:
    def definition_cpp(self, name_suffix='', as_reference=False):
        raise NotImplementedError


# region ArrayVariable


class ArrayVariableConverter:
    def __init__(self, type_converter, prefix, definition_cls):
        self.type_converter = type_converter
        self.prefix = prefix
        self.definition_cls = definition_cls

    def convert(self, tensor_var, pragma='partition'):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        tensor_var.pragma = pragma
        tensor_var.type = self.type_converter.convert(tensor_var.type)
        tensor_cls_fqn = tensor_var.__class__.__module__ + '.' + tensor_var.__class__.__qualname__

        tensor_var.__class__ = type(
            self.prefix + 'ArrayVariable', (type(tensor_var), self.definition_cls), {'_wrapped': tensor_cls_fqn}
        )
        return tensor_var


# endregion

# region StructMemberVariable


class StructMemberVariableConverter:
    def __init__(self, type_converter, prefix, definition_cls):
        self.type_converter = type_converter
        self.prefix = prefix
        self.definition_cls = definition_cls

    def convert(self, tensor_var, pragma='partition', struct_name=None):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        tensor_var.pragma = pragma
        tensor_var.type = self.type_converter.convert(tensor_var.type)

        assert struct_name is not None, 'struct_name must be provided when creating a StructMemberVariable'
        tensor_var.struct_name = str(struct_name)
        tensor_var.member_name = tensor_var.name
        tensor_var.name = tensor_var.struct_name + '.' + tensor_var.member_name
        type_cls_fqn = tensor_var.__class__.__module__ + '.' + tensor_var.__class__.__qualname__

        tensor_var.__class__ = type(
            self.prefix + 'StructMemberVariable', (type(tensor_var), self.definition_cls), {'_wrapped': type_cls_fqn}
        )
        return tensor_var


# endregion

# region StreamVariable


class StreamVariableConverter:
    def __init__(self, type_converter, prefix, definition_cls):
        self.type_converter = type_converter
        self.prefix = prefix
        self.definition_cls = definition_cls

    def convert(self, tensor_var, n_pack=1, depth=0):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        if depth == 0:
            depth = np.prod(tensor_var.shape) // tensor_var.shape[-1]
        tensor_var.pragma = ('stream', depth)
        tensor_var.type = self.type_converter.convert(
            PackedType(tensor_var.type.name, tensor_var.type.precision, tensor_var.shape[-1], n_pack)
        )
        tensor_cls_fqn = tensor_var.__class__.__module__ + '.' + tensor_var.__class__.__qualname__

        tensor_var.__class__ = type(
            self.prefix + 'StreamVariable', (type(tensor_var), self.definition_cls), {'_wrapped': tensor_cls_fqn}
        )
        return tensor_var


# endregion

# region InplaceStreamVariable


class InplaceStreamVariableConverter(StreamVariableConverter):
    def convert(self, tensor_var, n_pack=1, depth=0):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        tensor_var.pragma = None
        tensor_var.type = self.type_converter.convert(
            PackedType(tensor_var.type.name, tensor_var.type.precision, tensor_var.input_var.shape[-1], n_pack)
        )
        tensor_cls_fqn = tensor_var.__class__.__module__ + '.' + tensor_var.__class__.__qualname__

        tensor_var.__class__ = type(
            self.prefix + 'StreamVariable', (type(tensor_var), self.definition_cls), {'_wrapped': tensor_cls_fqn}
        )
        return tensor_var


# endregion

# region WeightsVariable


class StaticWeightVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return f'{self.type.name} {self.name}[{self.data_length}]'


class StaticWeightVariableConverter:
    def __init__(self, type_converter):
        self.type_converter = type_converter

    def convert(self, weight_var):
        if isinstance(weight_var, StaticWeightVariableDefinition):  # Already converted
            return weight_var

        weight_var.weight_class = weight_var.__class__.__name__
        weight_var.storage = 'register'
        weight_var.type = self.type_converter.convert(weight_var.type)
        tensor_cls_fqn = weight_var.__class__.__module__ + '.' + weight_var.__class__.__qualname__

        weight_var.__class__ = type(
            'StaticWeightVariable', (type(weight_var), StaticWeightVariableDefinition), {'_wrapped': tensor_cls_fqn}
        )
        return weight_var


class BramWeightVariableConverter:
    @classmethod
    def convert(cls, weight_var):
        weight_var.storage = 'bram'
        return weight_var


# endregion

# endregion
