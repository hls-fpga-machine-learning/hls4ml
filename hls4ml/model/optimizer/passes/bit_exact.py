import typing
from copy import copy
from functools import reduce, singledispatch
from math import ceil, log2
from typing import Sequence
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from hls4ml.model.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    Einsum,
    EinsumDense,
    GlobalPooling1D,
    Input,
    Layer,
    Merge,
    Pooling1D,
    Reshape,
    Softmax,
)
from hls4ml.model.optimizer import ModelOptimizerPass, OptimizerPass
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer, UnaryLUT
from hls4ml.model.types import FixedPrecisionType, NamedType, RoundingMode, SaturationMode, WeightVariable
from hls4ml.utils.qinterval import QIntervalArray, einsum, minimal_kif

if typing.TYPE_CHECKING:
    from hls4ml.model import ModelGraph


KIF_t = tuple[NDArray[np.int8], NDArray[np.int8], NDArray[np.int8]]


def to_hls4ml_fixed(k, i, f, name, *args):
    signed, b, I = k != 0, int(k + i + f), int(k + i)
    if b <= 0:
        b = 1
        I = 0
    args = [arg.upper() for arg in args]
    ptype = FixedPrecisionType(b, I, signed, *args)
    return NamedType(name, ptype)


def get_input_layers(layer: Layer):
    model: 'ModelGraph' = layer.model
    inp_names = layer.inputs
    return [model.graph[name] for name in inp_names]


def get_output_layers(layer: Layer):
    model: 'ModelGraph' = layer.model
    return [l for l in model.graph.values() if layer.name in l.attributes.get('inputs', ())]


def get_output_shape(layer: Layer) -> tuple[int, ...]:
    return tuple(layer.get_output_variable().shape)


def get_input_shapes(layer: Layer) -> list[tuple[int, ...]]:
    return [get_output_shape(inp) for inp in get_input_layers(layer)]


def _maximum_kif_at_shape(shape: tuple[int, ...]):
    k = np.ones(shape, dtype=np.int8)
    i = np.full(shape, 126, dtype=np.int8)
    f = np.full(shape, 126, dtype=np.int8)
    return k, i, f


@singledispatch
def request_kif(layer: Layer) -> tuple[KIF_t, ...]:
    input_shapes = get_input_shapes(layer)
    return tuple(_maximum_kif_at_shape(shape) for shape in input_shapes)


@request_kif.register
def _(layer: FixedPointQuantizer):
    assert layer.mask_kbi is not None
    k, b, I = layer.mask_kbi
    k, i, f = k, I - k, b - I

    out_shape = get_output_shape(layer)
    k = np.broadcast_to(k[0], out_shape).astype(np.int8)
    i = np.broadcast_to(i[0], out_shape).astype(np.int8)
    f = np.broadcast_to(f[0], out_shape).astype(np.int8)

    if layer.SAT != 'WRAP':
        k[:] = 1
        i[:] = 126
    if layer.RND == 'TRN':
        pass
    elif layer.RND == 'RND':
        f += 1
    else:
        f += 3
    return ((k, i, f),)


@request_kif.register(Pooling1D)
# @request_kif.register(Pooling2D)
@request_kif.register(GlobalPooling1D)
# @request_kif.register(GlobalPooling2D)
def _(layer: Pooling1D | GlobalPooling1D):
    # inp_shape = get_input_shapes(layer)[0]
    out_shape = get_output_shape(layer)
    pool_width = layer.attributes.attributes['pool_width']
    stride_width = layer.attributes.attributes['stride_width']
    pool_op = layer.attributes.attributes['pool_op']
    if isinstance(layer, Pooling1D):
        pad_0_0: int = layer.attributes.attributes['pad_left']
    else:
        pad_0_0 = 0
    is_ch_last = layer.attributes.attributes['data_format'] == 'channels_last'

    k = np.ones(out_shape, dtype=np.int8)
    i = np.full(out_shape, -127, dtype=np.int8)
    f = np.full(out_shape, 126, dtype=np.int8)

    _, i_out, f_out = requested_kif(layer)

    if not is_ch_last:
        i = np.moveaxis(i, 0, -1)
        f = np.moveaxis(f, 0, -1)

    for idx_out in range(k.shape[-1]):
        i_in_0 = i_out * stride_width - pad_0_0
        i_in_1 = i_in_0 + pool_width
        if i_in_0 < 0:
            i_in_0 = 0
        i[..., i_in_0:i_in_1] = i_out[..., idx_out]
        f[..., i_in_0:i_in_1] = f_out[..., idx_out]

    if not is_ch_last:
        i = np.moveaxis(i, -1, 0)
        f = np.moveaxis(f, -1, 0)

    if pool_op == 'Average':
        ln2_size = np.log2(pool_width)
        i += np.ceil(ln2_size).astype(np.int8)
        if not ln2_size.is_integer():
            f[:] = 126
    return ((k, i, f),)


@request_kif.register
def _(layer: Reshape):
    inp_shape = get_input_shapes(layer)[0]
    k, i, f = requested_kif(layer)
    k = k.reshape(inp_shape)
    i = i.reshape(inp_shape)
    f = f.reshape(inp_shape)
    return ((k, i, f),)


@request_kif.register
def _(layer: Activation):
    fn_name = layer.attributes.attributes.get('activation')
    if fn_name == 'linear':
        return (requested_kif(layer),)
    if fn_name == 'relu':
        k, i, f = requested_kif(layer)
        k = np.ones_like(k)
        return ((k, i, f),)
    inp_shape = get_input_shapes(layer)[0]
    return (_maximum_kif_at_shape(inp_shape),)


def requested_kif(layer: Layer) -> KIF_t:
    out_layers = get_output_layers(layer)
    out_shape = get_output_shape(layer)
    if not out_layers:
        return _maximum_kif_at_shape(out_shape)

    k = np.zeros(out_shape, dtype=np.int8)
    i = np.full(out_shape, -127, dtype=np.int8)
    f = i.copy()
    for out_layer in out_layers:
        _kif_s = request_kif(out_layer)
        out_layer_inp_layers = get_input_layers(out_layer)
        idx = out_layer_inp_layers.index(layer)
        k = np.maximum(k, _kif_s[idx][0])
        i = np.maximum(i, _kif_s[idx][1])
        f = np.maximum(f, _kif_s[idx][2])

    return k, i, f


@singledispatch
def produce_kif(layer: Layer) -> KIF_t:
    raise NotImplementedError(f'No implementation of produce_kif for {layer.__class__}')


@produce_kif.register
def _(layer: Input):
    k = np.ones(get_output_shape(layer), dtype=np.int8)
    i = f = np.full(get_output_shape(layer), 126, dtype=np.int8)
    return k, i, f


def get_input_kifs(layer: Layer):
    return [produce_kif(l) for l in get_input_layers(layer)]


@produce_kif.register
def _(layer: FixedPointQuantizer):
    assert layer.mask_kbi is not None
    k, b, I = layer.mask_kbi
    k, i, f = k, I - k, b - I

    out_shape = get_output_shape(layer)
    k = np.broadcast_to(k[0], out_shape)
    i = np.broadcast_to(i[0], out_shape)
    f = np.broadcast_to(f[0], out_shape)

    return k, i, f


@produce_kif.register
def _(layer: Reshape):
    out_shape = get_output_shape(layer)
    k, i, f = produce_kif(get_input_layers(layer)[0])
    return k.reshape(out_shape), i.reshape(out_shape), f.reshape(out_shape)


@produce_kif.register
def _(layer: Merge):
    op = layer.attributes.attributes['op'].lower()
    kif_ins = get_input_kifs(layer)
    match op:
        case 'add':
            qint_ins = [QIntervalArray.from_kif(*kif) for kif in kif_ins]
            k, i, f = reduce(lambda a, b: a + b, qint_ins).to_kif()  # type: ignore
            return k.astype(np.int8), i, f
        case 'concatename':
            axis = layer.attributes.attributes['axis']
            _ks, _is, _fs = zip(*[kif for kif in kif_ins])
            k = np.concatenate(_ks, axis=axis)
            i = np.concatenate(_is, axis=axis)
            f = np.concatenate(_fs, axis=axis)
            return k, i, f
        case _:
            raise NotImplementedError(f'No implementation of Merge for {op}')


@produce_kif.register
def _(layer: EinsumDense):
    t_kernel = layer.attributes.attributes['weight'].data
    to_original_kernel = layer.attributes.attributes['to_original_kernel']
    kernel = to_original_kernel(t_kernel)
    _bias = layer.attributes.attributes['bias']
    eq = layer.attributes.attributes['equation']
    k_in, i_in, f_in = get_input_kifs(layer)[0]
    qint_in = QIntervalArray.from_kif(k_in, i_in, f_in)
    qint_out = einsum(eq, qint_in, kernel)
    if _bias is not None:
        t_bias = _bias.data
        bias = t_bias.transpose(layer.attributes.attributes['out_tpose_idxs'])
        qint_out = qint_out + bias
    k, i, f = qint_out.to_kif()
    return k.astype(np.int8), i, f


@produce_kif.register
def _(layer: Einsum):
    kif_in1, kif_in2 = get_input_kifs(layer)
    qint_in1 = QIntervalArray.from_kif(*kif_in1)
    qint_in2 = QIntervalArray.from_kif(*kif_in2)
    eq = layer.attributes.attributes['equation']
    qint_out = einsum(eq, qint_in1, qint_in2)
    k, i, f = qint_out.to_kif()
    return k.astype(np.int8), i, f


@produce_kif.register
def _(layer: Dense):
    kernel = layer.attributes.attributes['weight'].data
    _bias = layer.attributes.attributes['bias']
    k_in, i_in, f_in = get_input_kifs(layer)[0]
    qint_in = QIntervalArray.from_kif(k_in, i_in, f_in)
    qint_out = qint_in @ kernel
    if _bias is not None:
        qint_out = qint_out + _bias.data
    k, i, f = qint_out.to_kif()
    return k.astype(np.int8), i, f


def r_im2col(kernel_size: Sequence[int], arr: np.ndarray, buffer: np.ndarray, axis: int):
    w = kernel_size[0]
    if len(kernel_size) == 3:  # 1D
        for i in range(arr.shape[axis] - w + 1):
            patch = np.take(arr, range(i, i + w), axis=axis)
            buffer[i] = patch.flatten()
    else:  # 2D+
        for i in range(arr.shape[axis] - w + 1):
            patch = arr[i : i + w]
            r_im2col(kernel_size[1:], patch, buffer[i], axis + 1)


def _im2col(kernel_size: Sequence[int], arr: np.ndarray):
    if len(kernel_size) < 3:
        return arr
    shape = [inp_d - ker_d + 1 for inp_d, ker_d in zip(arr.shape, kernel_size[:-2])]
    shape.append(np.prod(kernel_size[:-1]))  # type: ignore
    buf = np.empty(shape, dtype=arr.dtype)
    r_im2col(kernel_size, arr, buf, 0)
    return buf


def im2col(kernel_size: Sequence[int], *arrs: np.ndarray):
    """im2col for multidimensional arrays. Assumes Channel Last format.

    Parameters
    ----------
    kernel_size : Sequence[int]
        The size of the kernel, in the form (*kernel_shape, ch_in, ch_out)

    *arrs : np.ndarray
        The input arrays to be transformed

    Returns
    -------
    list[np.ndarray]
        The transformed arrays
    """
    return [_im2col(kernel_size, arr) for arr in arrs]


def pad_arrs(node: Layer, pad_val: float = 0, *arrs: np.ndarray):
    out_arrs = []
    if node.class_name.endswith('Conv2D'):
        pad_top = node.attributes.attributes['pad_top']
        pad_bottom = node.attributes.attributes['pad_bottom']
        pad_left = node.attributes.attributes['pad_left']
        pad_right = node.attributes.attributes['pad_right']
        for arr in arrs:
            r = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=pad_val)
            out_arrs.append(r)
    elif node.class_name.endswith('Conv1D'):
        pad_left = node.attributes.attributes['pad_left']
        pad_right = node.attributes.attributes['pad_right']
        for arr in arrs:
            r = np.pad(arr, ((pad_left, pad_right), (0, 0)), constant_values=pad_val)
            out_arrs.append(r)
    else:
        raise ValueError(f'Layer {node.class_name} is not supported for pad_arrs')
    return tuple(out_arrs)


def stride_arrs(node: Layer, *arrs: np.ndarray):
    if node.class_name.endswith('Conv2D'):
        st_h = node.attributes.attributes['stride_height']
        st_w = node.attributes.attributes['stride_width']
        return tuple(arr[::st_h, ::st_w] for arr in arrs)
    if node.class_name.endswith('Conv1D'):
        st_w = node.attributes.attributes['stride_width']
        return tuple(arr[::st_w] for arr in arrs)
    raise ValueError(f'Layer {node.class_name} is not supported for stride_arrs')


@produce_kif.register(Conv1D)
@produce_kif.register(Conv2D)
def _(layer: Conv1D | Conv2D):
    kernel = layer.attributes.attributes['weight'].data
    _bias = layer.attributes.attributes['bias']
    bias = _bias.data if _bias is not None else 0
    k_in, i_in, f_in = get_input_kifs(layer)[0]
    k_in, i_in, f_in = pad_arrs(layer, 0, k_in, i_in, f_in)
    k_in, i_in, f_in = im2col(kernel.shape, k_in, i_in, f_in)
    k_in, i_in, f_in = stride_arrs(layer, k_in, i_in, f_in)
    kernel = kernel.reshape(-1, kernel.shape[-1])
    qint_in = QIntervalArray.from_kif(k_in, i_in, f_in)
    qint_out = qint_in @ kernel
    qint_out = qint_out + bias
    k, i, f = qint_out.to_kif()
    return k.astype(np.int8), i, f


@produce_kif.register
def _(layer: BatchNormalization):
    k_in, i_in, f_in = get_input_kifs(layer)[0]
    qint_in = QIntervalArray.from_kif(k_in, i_in, f_in)
    scale = layer.attributes.attributes['scale'].data

    _bias = layer.attributes.attributes['bias']
    bias = _bias.data if _bias is not None else 0

    qint_out = qint_in * scale + bias
    k, i, f = qint_out.to_kif()
    return k.astype(np.int8), i, f


@produce_kif.register
def _(layer: Softmax):
    out_shape = get_output_shape(layer)

    inv_table_t: FixedPrecisionType = layer.attributes['inv_table_t'].precision
    exp_table_t: FixedPrecisionType = layer.attributes['exp_table_t'].precision

    b_exp, I_exp = exp_table_t.width, exp_table_t.integer
    b_inv, I_inv = inv_table_t.width, inv_table_t.integer

    i_exp, f_exp = I_exp, b_exp - I_exp
    i_inv, f_inv = I_inv, b_inv - I_inv
    k = np.zeros(out_shape, dtype=np.int8)
    i = np.full(out_shape, min(i_exp + i_inv, 1), dtype=np.int8)
    f = np.full(out_shape, f_exp + f_inv, dtype=np.int8)

    return k, i, f


@produce_kif.register
def _(layer: Activation):
    fn_name = layer.attributes.attributes['activation']
    k, i, f = get_input_kifs(layer)[0]

    if fn_name == 'linear':
        return k, i, f
    if fn_name == 'relu':
        print(k.__class__)
        k = np.zeros_like(k)
        return k, i, f
    if fn_name == 'tanh':
        i = np.minimum(i, 1)
        f = np.full_like(f, 126)
        return k, i, f
    if fn_name == 'sigmoid':
        k = np.zeros_like(k)
        i = np.minimum(i, 1)
        f = np.full_like(f, 126)
        return k, i, f

    k = np.zeros_like(k)
    i = np.full_like(i, 1)
    f = np.full_like(f, 126)
    return k, i, f


@produce_kif.register
def _(layer: UnaryLUT):
    table_t = layer.attributes['table_t'].precision
    k, I, f = table_t.signed, table_t.integer, table_t.fractional
    i = I - k
    shape = get_output_shape(layer)
    k = np.full(shape, np.max(k), dtype=np.int8)
    i = np.full(shape, np.max(i), dtype=np.int8)
    f = np.full(shape, np.max(f), dtype=np.int8)
    return k, i, f


def kif_arrs_to_ints(arr: tuple[np.ndarray, np.ndarray, np.ndarray]):
    return tuple(int(np.max(a)) for a in arr)


def default_register_precision(layer: Layer):
    _pk, _pi, _pf = produce_kif(layer)
    _rk, _ri, _rf = requested_kif(layer)
    _out_kif = np.minimum(_pk, _rk), np.minimum(_pi, _ri), np.minimum(_pf, _rf)
    _out_kif[1][(_pf > _rf) & (_pi <= _ri)] += 1
    result_kif = kif_arrs_to_ints(_out_kif)
    result_t = to_hls4ml_fixed(*result_kif, f'{layer.name}_t')
    layer.attributes.attributes['result_t'] = result_t
    layer.get_output_variable().type = result_t

    overrides = {}

    if 'accum_t' in layer.attributes.attributes:
        accum_kif = kif_arrs_to_ints((_pk, _pi, _pf))
        accum_t = to_hls4ml_fixed(*accum_kif, f'{layer.name}_accum_t')
        overrides['accum_t'] = accum_t

    for w_name_t, v in layer.attributes.attributes.items():
        if isinstance(v, NamedType) and w_name_t.endswith('_t'):
            w_name = w_name_t[:-2]
            if w_name not in layer.attributes.attributes:
                continue
            _data = layer.attributes.attributes[w_name]
            if _data is None:
                precision = to_hls4ml_fixed(0, 0, 1, f'{layer.name}_{w_name_t}')
            else:
                data = _data.data
                if not isinstance(data, np.ndarray):
                    raise ValueError(f'Expected data to be np.ndarray, got {type(data)} on layer {layer.name}')
                k, i, f = kif_arrs_to_ints(minimal_kif(data))
                precision = to_hls4ml_fixed(k, i, f, f'{layer.name}_{w_name_t}')
            overrides[w_name_t] = precision

    for w_name_t, v in overrides.items():
        layer.attributes.attributes[w_name_t] = v
        if w_name_t[:-2] in layer.attributes.attributes:
            weight_var: WeightVariable = layer.attributes.attributes[w_name_t[:-2]]
            weight_var.type = v
            weight_var.update_precision(v.precision)
            layer.model.config.layer_name_precision[f'{layer.name}_{w_name_t[:-2]}'] = str(v.precision)

    return (_pk, _pi, _pf), (_rk, _ri, _rf), _out_kif


@singledispatch
def register_precision(node: Layer):
    default_register_precision(node)


@register_precision.register
def _(node: Softmax):
    inv_inp_t: FixedPrecisionType = node.attributes['inv_inp_t'].precision
    accum_t = copy(inv_inp_t)
    if inv_inp_t.saturation_mode != SaturationMode.WRAP:
        accum_t.saturation_mode = SaturationMode.WRAP
        n_in = node.attributes['n_in']
        scale = ceil(log2(n_in))
        accum_t.width += scale
        accum_t.integer += scale
    if inv_inp_t.rounding_mode == RoundingMode.TRN:
        pass
    elif inv_inp_t.rounding_mode == RoundingMode.RND:
        accum_t.width += 1
    else:
        accum_t.width += 3
    accum_t.rounding_mode = RoundingMode.TRN
    default_register_precision(node)
    impl = node.attributes['implementation']
    match impl:
        case 'latency':
            k, i, f = get_input_kifs(node)[0]
            b = np.max(k) + np.max(i) + np.max(f)
        case 'stable':
            inp_norm_t: FixedPrecisionType = node.attributes['inp_norm_t'].precision
            b = inp_norm_t.width
        case 'lagency':
            raise ValueError('lagency softmax is not supported')
        case 'argmax':
            b = 0
        case _:
            raise ValueError(f'Unknown softmax implementation {impl}')

    exp_table_size = 2 ** int(b)
    node.attributes['exp_table_size'] = exp_table_size
    node.attributes['accum_t'] = NamedType(f'{node.name}_accum_t', accum_t)


@register_precision.register
def _(node: UnaryLUT):
    k, i, f = minimal_kif(node.attributes['table'].data)  # type: ignore
    k, i, f = bool(np.max(k)), int(np.max(i)), int(np.max(f))
    table_t = to_hls4ml_fixed(k, i, f, f'{node.name}_table_t')
    node.attributes['table_t'] = table_t
    default_register_precision(node)


class BitExact(ModelOptimizerPass):
    def __init__(self):
        pass

    def _match(self, model: 'ModelGraph'):
        if not any(isinstance(node, FixedPointQuantizer) for node in model.graph.values()):
            return False
        return True

    def transform(self, model):
        if not self._match(model):
            return False

        for node in model.graph.values():
            if node.attributes.get('bit_exact_transformed'):
                return False
            register_precision(node)
            node.attributes['bit_exact_transformed'] = True

        return False


class FixInputPrecision(OptimizerPass):
    def match(self, node: Layer):
        if not isinstance(node, Input):
            return False

        # Unhandled input precision, usually by a heterogeneous quantizer with non-WRAP saturation
        return node.get_output_variable().type.precision.width > 120

    def transform(self, model, node: Layer):
        out_layers: list[FixedPointQuantizer] = get_output_layers(node)
        if not all(isinstance(l, FixedPointQuantizer) for l in out_layers):
            warn(f'Input {node.name} has unhandled high precision. Consider setting it manually before synthesising.')
            return False

        sat_modes = [l.SAT for l in out_layers]
        sat_modes_set = set(sat_modes)
        illegal_sat_modes = sat_modes_set - {'WRAP', 'SAT', 'SAT_SYM'}
        if illegal_sat_modes:
            raise ValueError(f'Input {node.name} has quantizer with illegal saturation mode {illegal_sat_modes} after.')

        kifs = [produce_kif(l) for l in out_layers]
        i = np.max([np.max(i) for _, i, _ in kifs])
        k = np.max([np.max(k) for k, _, _ in kifs])
        f = node.get_output_variable().type.precision.fractional
        new_type = to_hls4ml_fixed(k, i, f, f'{node.name}_t')
        new_type.precision.saturation_mode = 'SAT'
        node.get_output_variable().type = new_type
        node.model.config.layer_name_precision[node.name] = str(new_type)
        return False
