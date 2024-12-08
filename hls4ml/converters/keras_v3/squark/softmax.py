import typing
from math import prod
from typing import Sequence

from hls4ml.model.types import FixedPrecisionType, RoundingMode, SaturationMode

from ..core import KV3SoftmaxHandler
from ._base import SQLayerHandler, register

if typing.TYPE_CHECKING:
    import squark
    from keras.api import KerasTensor
    from squark.quantizer.internal import FixedPointQuantizerBase


def fixed_quantizer_to_hls4ml_t(q: 'FixedPointQuantizerBase', take_max=False):
    from keras import ops

    k, i, f = q.kif
    k = ops.convert_to_numpy(k)
    i = ops.convert_to_numpy(i)
    f = ops.convert_to_numpy(f)
    if not take_max:
        assert k.size == 1 and i.size == 1 and f.size == 1, 'Only homogeneous quantizer is supported'
        k = bool(k.ravel().item())
        i = int(i.ravel().item())
        f = int(f.ravel().item())
    else:
        k = bool(k.max())
        i = int(i.max())
        f = int(f.max())

    k, b, I = k, k + i + f, k + i  # noqa: E741
    round_mode = q.round_mode
    if round_mode.startswith('S_'):
        round_mode = round_mode[2:]  # stochastic rounding
    round_mode = getattr(RoundingMode, round_mode)
    sat_mode = getattr(SaturationMode, q.overflow_mode)
    return FixedPrecisionType(b, I, k, rounding_mode=round_mode, saturation_mode=sat_mode)


@register
class SQSoftmaxDenseHandler(SQLayerHandler, KV3SoftmaxHandler):
    handles = ('squark.layers.softmax.QSoftmax',)

    def handle(
        self,
        layer: 'squark.layers.QSoftmax',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        assert not layer._allow_heterogeneous_table, 'Heterogeneous table is not supported in QSoftmax layer'
        assert len(layer.axis) == 1, 'Support softmax along one axis. Use transpose & reshape as workaround.'

        from keras import ops
        from squark.quantizer.internal import FixedPointQuantizerBase

        impl = 'stable' if layer.stable else 'latency'

        if impl == 'stable':
            exp_table_size = 2 ** int(ops.convert_to_numpy(ops.max(layer.exp_table.iq.quantizer.bits)))
        else:
            exp_table_size = None

        exp_oq = layer.exp_table.oq.quantizer
        inv_oq = layer.inv_table.oq.quantizer
        inv_iq = layer.inv_table.iq.quantizer
        assert isinstance(exp_oq, FixedPointQuantizerBase), 'Only fixed-point quantizer is supported for exp_table'
        exp_table_t = fixed_quantizer_to_hls4ml_t(exp_oq)
        inv_table_t = fixed_quantizer_to_hls4ml_t(inv_oq)
        inv_inp_t = fixed_quantizer_to_hls4ml_t(inv_iq)
        exp_scale = layer.input_scaler

        inv_table_size = 2**inv_inp_t.width

        config = super().handle(layer, in_tensors, out_tensors)
        assert len(config) == 1
        parallelization_factor = layer.parallelization_factor

        ax = layer.axis[0]
        ax = ax if ax >= 0 else len(in_tensors[0].shape) + ax
        # io_stream asserts axis=-1, convert to -1 when it is
        n_outer: int = prod(in_tensors[0].shape[1:ax])  # type: ignore
        n_inner: int = prod(in_tensors[0].shape[ax + 1 :])  # type: ignore
        ax = -1 if ax == len(in_tensors[0].shape) - 1 else ax
        n_in: int = in_tensors[0].shape[ax]  # type: ignore
        if parallelization_factor < 0:
            parallelization_factor = n_outer * n_inner

        config[0].update(
            {
                'axis': ax,
                'n_in': n_in,
                'n_outer': n_outer,
                'n_inner': n_inner,
                'implementation': impl,
                'exp_table_t': exp_table_t,
                'exp_table_size': exp_table_size,
                'inv_table_t': inv_table_t,
                'inv_table_size': inv_table_size,
                'inv_inp_t': inv_inp_t,
                'exp_scale': exp_scale,
                'parallelization_factor': parallelization_factor,
            }
        )
        if layer.stable:
            inp_norm_t = fixed_quantizer_to_hls4ml_t(layer.exp_table.iq.quantizer)
            inp_norm_t.saturation_mode = SaturationMode.WRAP
            inp_norm_t.rounding_mode = RoundingMode.TRN
            config[0]['inp_norm_t'] = inp_norm_t
        return config
