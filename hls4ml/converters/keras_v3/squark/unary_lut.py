import typing
from math import prod
from typing import Sequence

import numpy as np
from quantizers import float_quantize_np, get_fixed_quantizer_np

from hls4ml.model.types import FixedPrecisionType

from ._base import SQLayerHandler, register

if typing.TYPE_CHECKING:
    import squark
    from keras.api import KerasTensor
    from squark.quantizer.internal import FixedPointQuantizerBase, FloatPointQuantizer

from decimal import Decimal

from hls4ml.utils.qinterval import minimal_kif


@register
class SQUnaryLUTHandler(SQLayerHandler):
    handles = ('squark.layers.activation.QUnaryFunctionLUT',)

    def handle(
        self,
        layer: 'squark.layers.QUnaryFunctionLUT',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):

        from keras import ops

        if not layer.enable_iq and not layer.enable_oq:
            raise ValueError('Currently only support input_quantizer enabled UnaryFunctionLUT layer')
        assert not layer._allow_heterogeneous_table, 'Heterogeneous table is not supported in QUnaryFunctionLUT layer'

        iq = layer.iq.quantizer
        _min = Decimal(float(ops.min(iq.min)))  # type: ignore
        _max = Decimal(float(ops.max(iq.max)))  # type: ignore
        _eps = Decimal(float(ops.min(iq.epsilon)))  # type: ignore
        N = (_max - _min) / _eps
        assert float(N).is_integer(), 'Invalid quantizer range'
        N = int(N)
        assert N <= 1e6, 'Too large quantizer range'
        assert np.log2(N).is_integer(), f'Invalid quantizer range: N must be power of 2, got {N}'

        all_inputs = iq(ops.linspace(_min, _max, N))
        all_inputs = ops.array(np.unique(ops.convert_to_numpy(all_inputs)))
        table = ops.convert_to_numpy(layer.activation(all_inputs))

        if isinstance(iq, FixedPointQuantizerBase) and _min < 0:
            # idx by binary repr, move the positive part to the front
            table_pos, table_neg = table[N // 2 :], table[: N // 2]
            table = np.concatenate([table_pos, table_neg])

        oq = layer.oq.quantizer
        if isinstance(oq, FixedPointQuantizerBase):
            round_mode = oq.round_mode
            if round_mode.startswith('S_'):
                round_mode = round_mode[2:]
            overflow_mode = oq.overflow_mode
            fixed_q = get_fixed_quantizer_np(round_mode, overflow_mode)
            k, i, f = (ops.convert_to_numpy(x).ravel().item() for x in oq.kif)
            table = fixed_q(table, k, i, f)  # type: ignore

            k, b, I = bool(k), k + i + f, k + i  # noqa: E741
            table_t = FixedPrecisionType(b, I, k)
        else:
            assert isinstance(oq, FloatPointQuantizer)
            m, e, e0 = (ops.convert_to_numpy(x).ravel().item() for x in (oq.m, oq.e, oq.e0))
            table = float_quantize_np(table, m, e, e0)
            k, i, f = (int(np.min(x)) for x in minimal_kif(table))

            raise NotImplementedError('FloatPointQuantizer is not supported yet')
            table_t = FixedPrecisionType(k + i + f, k + i, bool(k))

        return {
            'n_in': prod(layer.input_shape[1:]),
            'table': table,
            'table_t': table_t,
        }
