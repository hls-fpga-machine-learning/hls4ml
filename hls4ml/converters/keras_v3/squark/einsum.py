import typing
from typing import Sequence

from ._base import SQLayerHandler, register

if typing.TYPE_CHECKING:
    import squark
    from keras.api import KerasTensor


def strip_batch_dim(equation: str, einsum_dense: bool = True):
    """Remove the batch dimension from the equation.

    Args:
        equation (str): The einsum equation.
        einsum_dense (bool): Whether the equation is for EinsumDense layer.

    Returns:
        str: The einsum equation without the batch dimension.
    """

    _inps, out = equation.split('->')
    inp0, inp1 = _inps.split(',')
    if einsum_dense:
        if inp0.startswith('...'):
            assert out.startswith('...'), f'Error in eq: {equation}: Batch dim mismatch for the input and output.'
        else:
            assert inp0[0] == out[0], f'Error in eq: {equation}: Batch dim mismatch for the input and output.'
            assert inp0[0] not in inp1, f'Error in eq: {equation}: Batch dim is used in the kernel.'
            inp0, out = inp0[1:], out[1:]
    else:
        assert inp0[0] == inp1[0] == out[0], f'Error in eq: {equation}: Batch dim mismatch for the inputs and output.'
        inp0, inp1, out = inp0[1:], inp1[1:], out[1:]
    return f'{inp0},{inp1}->{out}'


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
