from functools import singledispatchmethod
from typing import Any, Sequence, overload

import numpy as np

from hls4ml.utils.einsum_utils import EinsumRecipe, parse_einsum


def _minimal_f(array: np.ndarray):
    _low, _high = np.full(array.shape, -32, dtype=np.int8), np.full(array.shape, 32, dtype=np.int8)
    while np.any(_low < _high - 1):
        _mid = (_low + _high) // 2
        scaled = array * 2.0**_mid
        mask = scaled != scaled.astype(np.int64)
        _low = np.where(mask, _mid, _low)
        _high = np.where(mask, _high, _mid)
    return _high


def minimal_kif(array: np.ndarray):
    """Given a constant array, determine the minimal k, i, f values that can contain it with no loss of precision.

    Parameters
    ----------
    array : np.ndarray
        The constant array to be represented.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The minimal k, i, f values that can contain the array with no loss of precision.
    """
    f = _minimal_f(array)
    with np.errstate(divide='ignore', invalid='ignore'):
        i = np.ceil(np.log2(np.maximum(array + 2.0**-f, -array))).astype(np.int8)
    k = array < 0
    null_mask = array == 0
    i, f = np.where(null_mask, 0, i), np.where(null_mask, 0, f)
    return k, i, f


class _QIntervalArray:
    def __init__(self, min: np.ndarray, max: np.ndarray, delta: np.ndarray):
        self.min = min.astype(np.float64)
        self.max = max.astype(np.float64)
        self.delta = delta.astype(np.float64)
        self._validate()

    def _validate(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            assert np.all(self.min <= self.max), "min must be less than or equal to max"
            assert np.all(
                (self.max % self.delta == 0) | ((self.max == 0) & (self.delta == 0))
            ), "max must be a multiple of delta"
            assert np.all(
                (self.min % self.delta == 0) | ((self.min == 0) & (self.delta == 0))
            ), "min must be a multiple of delta"


class QIntervalArray(_QIntervalArray):
    """Symbolic array for quantized interval arithmetic.

    Available operations are:
    - Addition
    - Subtraction
    - Multiplication
    - Division (not recommended)
    - Matrix multiplication

    Parameters
    ----------
    min : np.ndarray
        The minimum value of the interval.
    max : np.ndarray
        The maximum value of the interval.
    delta : np.ndarray
        The quantization step of the interval.
    """

    @singledispatchmethod
    def __add__(self, other):
        _min = self.min + other
        _max = self.max + other
        _delta = np.minimum(self.delta, 2.0 ** -_minimal_f(other))
        return QIntervalArray(_min, _max, _delta)

    @__add__.register
    def _(self, other: _QIntervalArray):
        _min = self.min + other.min
        _max = self.max + other.max
        _delta = np.minimum(self.delta, other.delta)
        return QIntervalArray(_min, _max, _delta)

    def __sub__(self, other):
        return self + (-other)

    @singledispatchmethod
    def __mul__(self, other):
        other = np.float64(other)
        v1 = self.min * other
        v2 = self.max * other
        _min = np.minimum(v1, v2)
        _max = np.maximum(v1, v2)
        _delta = self.delta * other
        return QIntervalArray(_min, _max, _delta)

    @__mul__.register
    def _(self, other: _QIntervalArray):
        v1 = self.min * other.min
        v2 = self.min * other.max
        v3 = self.max * other.min
        v4 = self.max * other.max
        _min = np.minimum(np.minimum(v1, v2), np.minimum(v3, v4))
        _max = np.maximum(np.maximum(v1, v2), np.maximum(v3, v4))
        _delta = self.delta * other.delta
        return QIntervalArray(_min, _max, _delta)

    def __truediv__(self, other):
        return self * (1 / other)

    def __neg__(self):
        return QIntervalArray(-self.max, -self.min, self.delta)

    @singledispatchmethod
    def __matmul__(self, other: np.ndarray):
        seq = ''.join(chr(ord('a') + i) for i in range(self.min.ndim))
        eq = f'{seq},{seq[-1]}...->{seq}...'
        ax = self.min.ndim - 1
        v1 = np.einsum(eq, self.min, other, optimize=True)
        v2 = np.einsum(eq, self.max, other, optimize=True)
        other_delta = 2.0 ** -_minimal_f(other)
        _delta = np.einsum(eq, self.delta, other_delta, optimize=True)
        delta = np.min(np.where(_delta == 0, np.inf, _delta), axis=ax)
        _min = np.sum(np.minimum(v1, v2), axis=ax)
        _max = np.sum(np.maximum(v1, v2), axis=ax)
        return QIntervalArray(_min, _max, delta)

    @__matmul__.register
    def _(self, other: _QIntervalArray):
        seq = ''.join(chr(ord('a') + i) for i in range(self.min.ndim))
        eq = f'{seq},{seq[-1]}...->{seq}...'
        ax = self.min.ndim - 1
        v1 = np.einsum(eq, self.min, other.min, optimize=True)
        v2 = np.einsum(eq, self.max, other.max, optimize=True)
        v3 = np.einsum(eq, self.min, other.max, optimize=True)
        v4 = np.einsum(eq, self.max, other.min, optimize=True)

        _max = np.sum(np.maximum(np.maximum(v1, v2), np.maximum(v3, v4)), axis=ax)
        _min = np.sum(np.minimum(np.minimum(v1, v2), np.minimum(v3, v4)), axis=ax)

        _delta = np.einsum(eq, self.delta, other.delta, optimize=True)
        delta = np.min(_delta, axis=ax)

        return QIntervalArray(_min, _max, delta)

    def __rmatmul__(self, other: np.ndarray):
        seq = ''.join(chr(ord('a') + i) for i in range(other.ndim))
        eq = f'{seq},{seq[-1]}...->{seq}...'
        ax = other.ndim - 1
        v1 = np.einsum(eq, other, self.min, optimize=True)
        v2 = np.einsum(eq, other, self.max, optimize=True)
        other_delta = 2.0 ** -_minimal_f(other)
        _delta = np.einsum(eq, other_delta, self.delta, optimize=True)
        delta = np.min(np.where(_delta == 0, np.inf, _delta), axis=ax)
        _min = np.sum(np.minimum(v1, v2), axis=ax)
        _max = np.sum(np.maximum(v1, v2), axis=ax)
        return QIntervalArray(_min, _max, delta)

    def transpose(self, axes: Sequence[int]):
        return QIntervalArray(self.min.transpose(axes), self.max.transpose(axes), self.delta.transpose(axes))

    @property
    def shape(self):
        return self.min.shape

    def reshape(self, shape: Sequence[int]):
        return QIntervalArray(self.min.reshape(shape), self.max.reshape(shape), self.delta.reshape(shape))

    def ravel(self):
        return QIntervalArray(self.min.ravel(), self.max.ravel(), self.delta.ravel())

    @property
    def dtype(self):
        return self.min.dtype

    def __getitem__(self, key):
        return QIntervalArray(self.min[key], self.max[key], self.delta[key])

    def __array_function__(self, func, types, args, kwargs):
        if func == np.concatenate:
            return QIntervalArray(
                np.concatenate([a.min for a in args[0]]),
                np.concatenate([a.max for a in args[0]]),
                np.concatenate([a.delta for a in args[0]]),
            )
        return NotImplemented

    def rmatmul(self, other: np.ndarray):
        """Right matrix multiplication (other @ self), with __rmatmul__ implemented in QIntervalArray.
        This is to avoid using the @ operator defined in np.ndarray.

        Parameters
        ----------
        other : np.ndarray
            The operand matrix multiplied from the left.

        Returns
        -------
        QIntervalArray
            The result
        """
        return self.__rmatmul__(other)

    @classmethod
    def from_kif(cls, k: np.ndarray | int | bool, i: np.ndarray | int, f: np.ndarray | int):
        """Create a QIntervalArray from k, i, f values.

        Parameters
        ----------
        k : np.ndarray | int | bool
            keep_negative
        i : np.ndarray | int
            integer_bits, excluding sign bit
        f : np.ndarray | int
            fractional_bits

        Returns
        -------
        QIntervalArray
            The created QIntervalArray.
        """

        _min = np.asarray(-(2.0**i) * k)
        _max = np.asarray(2.0**i - 2.0**-f)
        _delta = np.asarray(2.0**-f)
        return cls(_min, _max, _delta)

    def sample(self, n: int | None = None):
        if n is not None:
            rand = np.random.rand(n, *self.min.shape)
        else:
            rand = np.random.rand(*self.min.shape)
        v = rand * (self.max - self.min) + self.min
        v = np.round(v / self.delta) * self.delta
        return v

    def to_kif(self):
        f = -np.log2(self.delta).astype(np.int8)

        with np.errstate(divide='ignore', invalid='ignore'):
            i = np.ceil(np.log2(np.maximum(self.max + 2.0**-f, -self.min))).astype(np.int8)
        k = self.min < 0
        null_mask = (self.max == 0) & (self.min == 0)
        i, f = np.where(null_mask, 0, i), np.where(null_mask, 0, f)
        return k, i, f


def _exec_einsum(recipe: EinsumRecipe, input0: np.ndarray | QIntervalArray, input1: np.ndarray | QIntervalArray, operator):
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
    # output = np.zeros(recipe['L0'] * recipe['L1'] * recipe['I'], dtype=input0.dtype)
    output = []

    L0, L1, I, C = recipe['L0'], recipe['L1'], recipe['I'], recipe['C']

    for i in range(I):
        for l0 in range(L0):
            A = input1[i * L1 * C : (i + 1) * L1 * C].reshape((L1, C))
            B = input0[(i * L0 + l0) * C : (i * L0 + l0 + 1) * C]
            output.append(operator(A, B))
    output = np.concatenate(output, axis=0)

    return output.reshape(recipe['out_interpert_shape']).transpose(recipe['out_transpose_idxs'])


@overload
def einsum(fn: str, input0: QIntervalArray, input1: QIntervalArray, operator=None) -> QIntervalArray: ...


@overload
def einsum(fn: str, input0: np.ndarray, input1: QIntervalArray, operator=None) -> QIntervalArray: ...


@overload
def einsum(fn: str, input0: QIntervalArray, input1: np.ndarray, operator=None) -> QIntervalArray: ...


@overload
def einsum(fn: str, input0: np.ndarray, input1: np.ndarray, operator=None) -> np.ndarray: ...


def einsum(fn: str, input0: np.ndarray | QIntervalArray, input1: np.ndarray | QIntervalArray) -> Any:  # type: ignore
    """Execute einsum operation on two input arrays

    WARNING: Order of multiplication is reversed -- watchout if you are using non-commutative operators

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

    def operator(A, B):
        if isinstance(A, np.ndarray):
            return B.__rmatmul__(A)
        else:
            return A @ B

    recipe = parse_einsum(fn, input0.shape, input1.shape)
    return _exec_einsum(recipe, input0, input1, operator)
