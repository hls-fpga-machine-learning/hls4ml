import typing
from functools import singledispatch
from typing import Sequence

import numpy as np

from hls4ml.model.layers import Conv1D, Conv2D, Dense, EinsumDense, GlobalPooling1D, Layer, Pooling1D, Reshape
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer

if typing.TYPE_CHECKING:
    from hls4ml.model import ModelGraph

from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType, NamedType
from hls4ml.utils.qinterval import QIntervalArray, einsum, minimal_kif


def to_hls4ml_fixed(k, i, f, name, *args):
    signed, b, i = k != 0, int(k + i + f), int(k + i)
    args = [arg.upper() for arg in args]
    ptype = FixedPrecisionType(b, i, signed, *args)
    return NamedType(name, ptype)


def get_input_layers(layer: Layer):
    model: 'ModelGraph' = layer.model
    inp_names = layer.attributes.attributes['inputs']
    return [model.graph[name] for name in inp_names]


def get_output_layers(layer: Layer):
    model: 'ModelGraph' = layer.model
    return [l for l in model.graph.values() if layer.name in l.attributes.attributes['inputs']]


def get_output_shape(layer: Layer):
    return layer.attributes.attributes[layer.name].shape


def get_input_shapes(layer: Layer):
    return [get_output_shape(inp) for inp in get_input_layers(layer)]


@singledispatch
def request_kif(layer: Layer):
    input_shape = get_input_shapes(layer)[0]
    k = np.ones(input_shape, dtype=np.int8)
    i = f = np.full(input_shape, 127, dtype=np.int8)
    return k, i, f


@request_kif.register
def _(layer: FixedPointQuantizer):
    assert layer.mask_kbi is not None
    k, b, I = layer.mask_kbi
    k, i, f = k, I - k, b - I
    if layer.SAT != 'WRAP':
        k[:] = 1
        i[:] = 127
    if layer.RND == 'TRN':
        pass
    elif layer.RND == 'RND':
        f += 1
    else:
        f += 2
    return k, i, f


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
    i = np.full(out_shape, -128, dtype=np.int8)
    f = np.full(out_shape, 127, dtype=np.int8)

    _, i_out, f_out = np.max([request_kif(next_layer) for next_layer in get_output_layers(layer)], axis=0)

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
            f[:] = 127
    return k, i, f


@request_kif.register
def _(layer: Reshape):
    inp_shape = get_input_shapes(layer)[0]
    k, i, f = np.max([request_kif(next_layer) for next_layer in get_output_layers(layer)], axis=0)
    return k.reshape(inp_shape), i.reshape(inp_shape), f.reshape(inp_shape)


def requested_kif(layer: Layer):
    out_layers = get_output_layers(layer)
    if not out_layers:
        out_shape = get_output_shape(layer)
        k = np.ones(out_shape, dtype=np.int8)
        i = f = np.full(out_shape, 127, dtype=np.int8)
        return k, i, f
    return tuple(np.max([request_kif(l) for l in out_layers], axis=0))


@singledispatch
def produce_kif(layer: Layer):
    k = np.ones(get_output_shape(layer), dtype=np.int8)
    i = f = np.full(get_output_shape(layer), 127, dtype=np.int8)
    return k, i, f


def get_input_kifs(layer: Layer):
    return [produce_kif(l) for l in get_input_layers(layer)]


@produce_kif.register
def _(layer: FixedPointQuantizer):
    assert layer.mask_kbi is not None
    k, b, I = layer.mask_kbi
    k, i, f = k, I - k, b - I
    return k[0], i[0], f[0]


@produce_kif.register
def _(layer: Reshape):
    out_shape = get_output_shape(layer)
    k, i, f = produce_kif(get_input_layers(layer)[0])
    return k.reshape(out_shape), i.reshape(out_shape), f.reshape(out_shape)


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


def pad_and_stride_inp_arr(node: Layer, arr: np.ndarray, pad_val: float = 0):
    if node.class_name.endswith('Conv2D'):
        pad_top = node.attributes.attributes['pad_top']
        pad_bottom = node.attributes.attributes['pad_bottom']
        pad_left = node.attributes.attributes['pad_left']
        pad_right = node.attributes.attributes['pad_right']
        st_h = node.attributes.attributes['stride_height']
        st_w = node.attributes.attributes['stride_width']
        return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=pad_val)[::st_h, ::st_w]
    if node.class_name.endswith('Conv1D'):
        pad_left = node.attributes.attributes['pad_left']
        pad_right = node.attributes.attributes['pad_right']
        st_w = node.attributes.attributes['stride_width']
        return np.pad(arr, ((pad_left, pad_right), (0, 0)), constant_values=pad_val)[::st_w]
    return arr


@produce_kif.register(Conv1D)
@produce_kif.register(Conv2D)
def _(layer: Conv1D | Conv2D):
    kernel = layer.attributes.attributes['weight'].data
    _bias = layer.attributes.attributes['bias']
    bias = _bias.data if _bias is not None else 0
    k_in, i_in, f_in = get_input_kifs(layer)[0]
    k_in, i_in, f_in = im2col(kernel.shape, k_in, i_in, f_in)
    k_in = pad_and_stride_inp_arr(layer, k_in, 0)
    i_in = pad_and_stride_inp_arr(layer, i_in, 0)
    f_in = pad_and_stride_inp_arr(layer, f_in, 0)
    kernel = kernel.reshape(-1, kernel.shape[-1])
    qint_in = QIntervalArray.from_kif(k_in, i_in, f_in)
    qint_out = qint_in @ kernel
    qint_out = qint_out + bias
    k, i, f = qint_out.to_kif()
    return k.astype(np.int8), i, f


def kif_arrs_to_ints(arr: tuple[np.ndarray, np.ndarray, np.ndarray]):
    return tuple(int(np.max(a)) for a in arr)


def register_precision(layer: Layer):
    _pk, _pi, _pf = produce_kif(layer)
    _rk, _ri, _rf = requested_kif(layer)
    _out_kif = np.minimum(_pk, _rk), np.minimum(_pi, _ri), np.minimum(_pf, _rf)
    _out_kif[1][(_pf > _rf) & (_pi <= _ri)] += 1
    result_kif = kif_arrs_to_ints(_out_kif)
    result_t = to_hls4ml_fixed(*result_kif, f'{layer.name}_result_t')
    layer.attributes.attributes['result_t'] = result_t
    layer.attributes.attributes[layer.name].type = result_t  # Why??????

    if 'accum_t' in layer.attributes.attributes:
        accum_kif = kif_arrs_to_ints((_pk, _pi, _pf))
        accum_t = to_hls4ml_fixed(*accum_kif, f'{layer.name}_accum_t')
        layer.attributes.attributes['accum_t'] = accum_t

    if 'weight_t' in layer.attributes.attributes:
        kernel_kif = kif_arrs_to_ints(minimal_kif(layer.attributes.attributes['weight'].data))
        kernel_t = to_hls4ml_fixed(*kernel_kif, f'{layer.name}_weight_t')
        layer.attributes.attributes['weight_t'] = kernel_t

    if 'bias_t' in layer.attributes.attributes:
        _bias = layer.attributes.attributes.get('bias')
        if _bias is None:
            bias_t = to_hls4ml_fixed(0, 0, 1, f'{layer.name}_bias_t')
        else:
            bias_kif = kif_arrs_to_ints(minimal_kif(_bias.data))
            bias_t = to_hls4ml_fixed(*bias_kif, f'{layer.name}_bias_t')
        layer.attributes.attributes['bias_t'] = bias_t


class BitExact(OptimizerPass):
    def match(self, node):
        return True

    def transform(self, model, node):
        register_precision(node)
        return False
