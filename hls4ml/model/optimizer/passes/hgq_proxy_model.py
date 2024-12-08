import re
import typing
from copy import copy
from warnings import warn

import numpy as np

from hls4ml.backends.fpga.fpga_types import NamedType
from hls4ml.model.layers import Layer, Reshape, register_layer
from hls4ml.model.optimizer import OptimizerPass, register_pass
from hls4ml.model.optimizer.passes.bit_exact import get_input_layers, get_output_layers
from hls4ml.model.types import FixedPrecisionType, UnspecifiedPrecisionType, WeightVariable

if typing.TYPE_CHECKING:
    from hls4ml.model import ModelGraph

re_purge_prefix = re.compile(r'(?<!\w)(?:ap_|ac_)', re.IGNORECASE)
re_parse_fixed = re.compile(r'\s*(u?)fixed<([^>]+)>\s*', re.IGNORECASE)


class FixedPointQuantizer(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)
        self.set_attr('n_in', self.get_input_variable().size())
        self.overrides = self.attributes['overrides']
        self.fusible = self.attributes['fusible']
        self.SAT, self.RND = self.attributes['SAT'], self.attributes['RND']
        self.mask_kbi = self.attributes['mask_kbi']


class UnaryLUT(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)
        self.set_attr('n_in', inp.size())
        self.table = self.attributes['table']
        self.table_size = self.attributes['table_size']

        table_t = to_hls4ml_fixed(self.attributes['table_t'])
        self.add_weights_variable(name='table', var_name='table{index}', precision=table_t, data=self.table)


def to_hls4ml_fixed(fixed: str):
    matched = re_parse_fixed.match(re_purge_prefix.sub('', fixed))
    assert matched is not None, f'Cannot parse {fixed}'
    signed = matched.group(1) != 'u'
    b, i, *args = matched.group(2).split(',')
    b, i = int(b), int(i)
    args = [arg.upper() for arg in args]
    new_type = FixedPrecisionType(b, i, signed, *args)
    # For some reason, __class__ is overwritten in hls4ml
    return new_type


def userconf_ifdef(key: str, layer_name: str, model):
    hls_config: dict = model.config.config['HLSConfig']
    layer_confs: dict = hls_config.get('LayerName', None)
    if not layer_confs:
        return False
    layer_conf = layer_confs.get(layer_name, None)
    if not layer_conf:
        return False
    # return key in layer_conf # Ideal case. Not for now.
    if key.endswith('_t') and key != 'table_t':
        # table_t cannot be defined in Precision, for some reason.
        # On the other hand, result_t, weight_t, bias_t, accum_t cannot be decleared explicitly outside Precision, for now.
        # However, still assume that they can be defined explicitly outside Precision.
        precision_conf = layer_conf.get('Precision', None)
        if not precision_conf:
            return key in layer_conf
        return key[:-2] in precision_conf or key in layer_conf

    if key == 'parallelization_factor':
        # Irregular config key name.
        return 'ParallelizationFactor' in layer_conf

    return key in layer_conf


class FuseFixedPointQuantizer(OptimizerPass):
    def match(self, node: Layer):
        if not isinstance(node, FixedPointQuantizer):
            return False
        if any(np.unique(x).size > 1 for x in node.mask_kbi):
            return False
        return True

    def propagate(self, node: Layer, precision: FixedPrecisionType):
        node.attributes.attributes[node.name].type.precision = precision
        node.attributes.attributes['result_t'].precision = precision

        if not isinstance(node, Reshape):
            return

        inp_layer = get_input_layers(node)[0]
        can_propagate = len(get_output_layers(inp_layer)) == 1

        if not can_propagate:
            return

        new_precision = copy(precision)
        precision.saturation_bits = 0
        precision.rounding_mode = 'TRN'
        precision.saturation_mode = 'WRAP'
        self.propagate(inp_layer, new_precision)

    def transform(self, model: 'ModelGraph', node: FixedPointQuantizer):
        precision: FixedPrecisionType = copy(node.attributes[node.name].type.precision)
        # Rounding and saturation for FixedPointQuantizer are applied in generated code, thus not reflected in result_t.
        precision.rounding_mode = node.RND
        precision.saturation_mode = node.SAT
        ino_layer = get_input_layers(node)[0]
        can_fuse = len(get_output_layers(ino_layer)) == 1
        if not can_fuse:
            return False
        self.propagate(ino_layer, precision)
        model.remove_node(node)
        return True


class EnforceProxyModelEmbeddedConfig(OptimizerPass):
    def match(self, node: Layer):
        if not isinstance(node, FixedPointQuantizer):
            return False
        if not node.overrides:
            return False
        return True

    def transform(self, model, node: FixedPointQuantizer):
        if 'layers' not in node.overrides:
            return False

        graph_changed = False
        layers = node.overrides['layers']
        for name, conf in layers.items():
            conf: dict[str, str]
            name: str
            if name not in model.graph:
                # Some layer may be removed by other passes. (e.g. Final flatten layer)
                continue
            target_node: Layer = model.graph[name]

            # Invoke automatic precision derivation for pooling layers accum_t, if undefined.
            if 'pool' in target_node.__class__.__name__.lower():
                if not userconf_ifdef('accum_t', name, model):
                    target_node.attributes['accum_t'].precision = UnspecifiedPrecisionType()

            for k, v in conf.items():
                if userconf_ifdef(k, name, model):
                    warn(
                        f'Config key {k} is defined in hls_config for layer {name} by user. Proxy model config is ignored.',
                        stacklevel=1,
                    )
                    continue

                if k.endswith('_t'):
                    var_type = target_node.get_attr(k)  # type: ignore
                    if var_type is None:
                        continue
                    var_type: NamedType
                    precision = to_hls4ml_fixed(v)
                    var_type.precision = precision
                    if k == 'result_t':
                        type_name = f'{name}_t'
                    else:
                        type_name = f'{name}_{k}'
                    var_type.name = type_name
                    # Need to overwrite kernel/bias writing precision also, or written weights will likely be wrong.
                    if k[:-2] in target_node.attributes.keys():
                        weight_var: WeightVariable = target_node.attributes[k[:-2]]
                        # weight_var should be a StaticWeightVariable, which is again, defined with meta programming
                        # Type hinting using StaticWeightVariableDefinition which is the base class.
                        weight_var.update_precision(precision)
                    # Well, it turned out that there is yet ANOTHER copy saved in config.
                    model.config.layer_name_precision[f'{name}_{k[:-2]}'] = v
                elif k in target_node.attributes.attributes:
                    target_node.set_attr(k, v)
                elif k == 'parallelization_factor':
                    target_node.set_attr(k, int(v))

            if linear_node := model.graph.get(f'{name}_linear'):
                # Proxy model does not assume any extra linear layer.
                # Purge them on sight
                model.remove_node(linear_node)
                graph_changed = True

        return graph_changed


def register_hgq_proxy_model():
    register_layer('FixedPointQuantizer', FixedPointQuantizer)
    register_layer('HGQ>FixedPointQuantizer', FixedPointQuantizer)
    register_layer('UnaryLUT', UnaryLUT)
    register_layer('HGQ>UnaryLUT', UnaryLUT)
    # register_pass('enforce_proxy_model_embedded_config', EnforceProxyModelEmbeddedConfig)
    register_pass('fuse_fixed_point_quantizer', FuseFixedPointQuantizer)
