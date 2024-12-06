import typing
from typing import Sequence

from ..einsum_dense import strip_batch_dim
from ._base import SQLayerHandler, register

if typing.TYPE_CHECKING:
    import squark
    from keras.api import KerasTensor


@register
class SQEinsumDenseHandler(SQLayerHandler):
    handles = ('squark.layers.ops.einsum.QEinsum',)

    def handle(
        self,
        layer: 'squark.layers.QEinsum',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        assert len(in_tensors) == 2, 'EinsumDense layer must have exactly one input tensor'
        assert len(out_tensors) == 1, 'EinsumDense layer must have exactly one output tensor'

        inp0_shape: tuple[int, ...] = in_tensors[0].shape[1:]  # type: ignore
        inp1_shape: tuple[int, ...] = in_tensors[1].shape[1:]  # type: ignore
        out_shape: tuple[int, ...] = out_tensors[0].shape[1:]  # type: ignore

        # fmt: off
        assert all(d is not None for d in inp0_shape), \
            f'Error when processing {layer.name}: Einsum layer requires fully inp shapes, got {inp0_shape} for inp1'
        assert all(d is not None for d in inp1_shape), \
            f'Error when processing {layer.name}: Einsum layer requires fully inp shapes, got {inp1_shape} for inp2'
        assert all(d is not None for d in out_shape), \
            f'Error when processing {layer.name}: EinsumDense layer requires fully out shapes. got {out_shape} for output'
        # fmt: on

        equation = strip_batch_dim(layer.equation, einsum_dense=False)

        return {
            'class_name': 'Einsum',
            'equation': equation,
            'inp0_shape': inp0_shape,
            'inp1_shape': inp1_shape,
            'out_shape': out_shape,
        }
