from math import prod
from typing import TypedDict

import numpy as np


class EinsumRecipe(TypedDict):
    in_transpose_idxs: tuple[tuple[int, ...], tuple[int, ...]]
    L0: int
    L1: int
    I: int
    C: int
    out_interpert_shape: tuple[int, ...]
    out_transpose_idxs: tuple[int, ...]


def _validate_einsum_expr(fn: str, shape0: tuple[int, ...], shape1: tuple[int, ...]):
    """Validate, resolve broadcasting, and compute output shape for einsum string

    Parameters
    ----------
    fn : str
        einsum string, e.g. 'ij,jk->ik'
    shape0 : tuple[int,...]
        shape of input0
    shape1 : tuple[int,...]
        shape of input1

    Returns
    -------
    tuple[str, tuple[int,...]]
        einsum string w/o broadcasting, and output shape

    Raises
    ------
    ValueError
        If the einsum string is invalid, or if it is incompatible with the input shapes
    """
    inp, out = map(str.strip, fn.split('->'))
    in0, in1 = map(str.strip, inp.split(','))
    alphabets = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    s_alphabets = set(alphabets)

    # Invalid characters
    if not (s_alphabets >= set(in0.replace('...', '') + in1.replace('...', '') + out.replace('...', ''))):
        raise ValueError(f"einsum string {fn} is invalid: subscripts should be in [a-zA-Z] and '...' only")

    in0 = in0.replace('...', '0')
    in1 = in1.replace('...', '0')
    out = out.replace('...', '0')
    ax_in0, ax_in1, ax_out = list(in0), list(in1), list(out)
    sax_in0, sax_in1, sax_out = set(ax_in0), set(ax_in1), set(ax_out)
    free_indices = ''.join(sorted(s_alphabets - sax_in0 - sax_in1 - sax_out))

    # Repeated indices
    if len(sax_in0) != len(ax_in0):
        for a in in0:
            if in0.count(a) == 1:
                continue
            a = a if a != '0' else '...'
            raise ValueError(f"einsum string {fn} is invalid: input0 subscripts includes '{a}' multiple times")
    if len(sax_in1) != len(ax_in1):
        for a in in1:
            if in1.count(a) == 1:
                continue
            a = a if a != '0' else '...'
            raise ValueError(f"einsum string {fn} is invalid: input1 subscripts includes '{a}' multiple times")
    if len(sax_out) != len(ax_out):
        for a in out:
            if out.count(a) == 1:
                continue
            a = a if a != '0' else '...'
            raise ValueError(f"einsum string {fn} is invalid: output subscripts includes '{a}' multiple times")

    # Invalid broadcasting
    if '0' in sax_in0 or '0' in sax_in1 or '0' in sax_out:
        if '0' in sax_in0 and '0' in sax_in1:
            raise ValueError(f"einsum string {fn} is invalid: both input0 and input1 allows broadcasting")
        if '0' not in sax_out:
            raise ValueError(f"einsum string {fn} is invalid: output does not allow broadcasting, but inputs do")
        if '0' not in sax_in0 and '0' not in sax_in1:
            raise ValueError(f"einsum string {fn} is invalid: output allows broadcasting, but inputs do not")

    # Output index out of nowhere
    if remaining := sax_out - sax_in0 - sax_in1:
        raise ValueError(f"einsum string {fn} is invalid: output subscripts {remaining} not found in inputs")

    _common_in = sax_in0 & sax_in1

    # Invalid input dimensions
    if '0' in sax_in0:
        if len(sax_in0) - 1 > len(shape0):
            raise ValueError(f"Input0 requires at least {len(sax_in0)-1} dimensions, but only {len(shape0)} given")
        # Replace broadcasting indices with free indices
        n_broadcast = len(shape0) - len(sax_in0) + 1
        in0 = in0.replace('0', free_indices[:n_broadcast])
        out = out.replace('0', free_indices[:n_broadcast])
        ax_in0 = list(in0)
        ax_out = list(out)
    else:
        if len(sax_in0) != len(shape0):
            raise ValueError(f"Input0 requires {len(sax_in0)} dimensions, but {len(shape0)} is given")
    if '0' in sax_in1:
        if len(sax_in1) - 1 > len(shape1):
            raise ValueError(f"Input1 requires at least {len(sax_in1)-1} dimensions, but only {len(shape1)} given")
        # Replace broadcasting indices with free indices
        n_broadcast = len(shape1) - len(sax_in1) + 1
        in1 = in1.replace('0', free_indices[:n_broadcast])
        out = out.replace('0', free_indices[:n_broadcast])
        ax_in1 = list(in1)
        ax_out = list(out)
    else:
        if len(sax_in1) != len(shape1):
            raise ValueError(f"Input1 requires {len(sax_in1)} dimensions, but {len(shape1)} is given")

    # Input dimension mismatch
    for a in _common_in:
        ax_0 = ax_in0.index(a)
        ax_1 = ax_in1.index(a)
        if shape0[ax_0] != shape1[ax_1]:
            raise ValueError(
                f"Input dimension size mismatches for common subscript '{a}': {shape0[ax_0]} and {shape1[ax_1]}"
            )

    out_shape = tuple(shape0[ax_in0.index(a)] if a in ax_in0 else shape1[ax_in1.index(a)] for a in ax_out)
    return f'{in0},{in1}->{out}', out_shape


def parse_einsum(fn: str, input_shape0: tuple[int, ...], input_shape1: tuple[int, ...]) -> EinsumRecipe:
    """Execute einsum operation on two input arrays

    Parameters
    ----------
    fn : str
        einsum string, e.g. 'ij,jk->ik'
    input : np.ndarray
        input0, the first input array
    input1 : np.ndarray
        input1, the second input array

    Returns
    -------
    np.ndarray
        output array
    """

    fn, _ = _validate_einsum_expr(fn, input_shape0, input_shape1)

    _in, _out = fn.split('->')
    _in0, _in1 = _in.split(',')

    in0, in1, out = list(_in0), list(_in1), list(_out)
    s_in0, s_in1, s_out = set(in0), set(in1), set(out)
    _common = s_in0 & s_in1
    _contract = _common - s_out
    _inplace = _common & s_out
    contract = sorted(_contract, key=lambda x: in1.index(x))
    inplace = sorted(_inplace, key=lambda x: in1.index(x))
    invariant0 = sorted((s_out - _common) & s_in0, key=lambda x: in0.index(x))
    invariant1 = sorted((s_out - _common) & s_in1, key=lambda x: in1.index(x))

    contract_idxs = tuple(map(in0.index, contract)), tuple(map(in1.index, contract))
    inplace_idxs = tuple(map(in0.index, inplace)), tuple(map(in1.index, inplace))
    invariant_idxs = tuple(map(in0.index, invariant0)), tuple(map(in1.index, invariant1))

    inplace_shape = tuple(input_shape0[i] for i in inplace_idxs[0])
    inplace_size = prod(inplace_shape)
    contract_size = prod(input_shape0[i] for i in contract_idxs[0])
    invariant_shape0 = tuple(input_shape0[i] for i in invariant_idxs[0])
    invariant_shape1 = tuple(input_shape1[i] for i in invariant_idxs[1])
    invariant_size0, invariant_size1 = prod(invariant_shape0), prod(invariant_shape1)

    transpose_idx0 = inplace_idxs[0] + invariant_idxs[0] + contract_idxs[0]
    transpose_idx1 = inplace_idxs[1] + invariant_idxs[1] + contract_idxs[1]

    out_shape_pretranspose = inplace_shape + invariant_shape0 + invariant_shape1
    _out_transpose_idx = np.argsort(tuple(map(out.index, inplace + invariant0 + invariant1)))
    out_transpose_idx = tuple(int(i) for i in _out_transpose_idx)

    return EinsumRecipe(
        in_transpose_idxs=(transpose_idx0, transpose_idx1),
        out_interpert_shape=out_shape_pretranspose,
        out_transpose_idxs=out_transpose_idx,
        L0=invariant_size0,
        L1=invariant_size1,
        I=inplace_size,
        C=contract_size,
    )


def _exec_einsum(recipe: EinsumRecipe, input0: np.ndarray, input1: np.ndarray) -> np.ndarray:
    """Execute einsum operation on two input arrays

    Parameters
    ----------
    recipe : EinsumRecipe
        einsum recipe
    input0 : np.ndarray
        input0, the first input array
    input1 : np.ndarray
        input1, the second input array

    Returns
    -------
    np.ndarray
        output array
    """
    input0 = input0.transpose(recipe['in_transpose_idxs'][0]).ravel()
    input1 = input1.transpose(recipe['in_transpose_idxs'][1]).ravel()
    output = np.zeros(recipe['L0'] * recipe['L1'] * recipe['I'], dtype=input0.dtype)

    L0, L1, I, C = recipe['L0'], recipe['L1'], recipe['I'], recipe['C']

    for l0 in range(L0):
        for i in range(I):
            output[(i * L0 + l0) * L1 : (i * L0 + l0 + 1) * L1] = (
                input1[i * L1 * C : (i + 1) * L1 * C].reshape((L1, C)) @ input0[(i * L0 + l0) * C : (i * L0 + l0 + 1) * C]
            )

    return output.reshape(recipe['out_interpert_shape']).transpose(recipe['out_transpose_idxs'])


def einsum(fn: str, input0: np.ndarray, input1: np.ndarray) -> np.ndarray:
    """Execute einsum operation on two input arrays

    Parameters
    ----------
    fn : str
        einsum string, e.g. 'ij,jk->ik'
    input : np.ndarray
        input0, the first input array
    input1 : np.ndarray
        input1, the second input array

    Returns
    -------
    np.ndarray
        output array
    """
    recipe = parse_einsum(fn, input0.shape, input1.shape)
    return _exec_einsum(recipe, input0, input1)
