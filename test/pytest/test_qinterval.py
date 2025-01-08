import numpy as np
import pytest
from quantizers.fixed_point import get_fixed_quantizer_np

from hls4ml.utils.qinterval import QIntervalArray, einsum, minimal_kif


def assert_is_represented(qinterval: QIntervalArray, data: np.ndarray):
    assert np.all(data <= qinterval.max), f'{np.max(data - qinterval.max)} > 0'
    assert np.all(data >= qinterval.min), f'{np.min(data - qinterval.min)} < 0'
    with np.errstate(divide='ignore', invalid='ignore'):
        is_zero = (qinterval.max == 0) & (qinterval.min == 0)
        assert np.all((data % qinterval.delta == 0) | is_zero)


@pytest.fixture(scope='module')
def data():
    arr = np.random.randint(-1024, 1024, size=1000000)
    arr = arr * 2.0 ** np.random.randint(-20, 20, size=1000000)
    return arr


def test_minimal_kif(data):
    k, i, f = minimal_kif(data)
    q = get_fixed_quantizer_np()
    assert np.all(data == q(data, k, i, f))
    assert np.all((data != q(data, k, i, f - 1)) | (data == 0))
    assert np.all((data != q(data, k, i - 1, f)) | (data == 0) | (i + f == 0))


def random_arr(seed=None):
    rng = np.random.default_rng(seed)
    shape = (64, 64)

    _delta = 2.0 ** rng.integers(-8, 8, shape)
    _min = rng.integers(-1024, 1024, shape) * _delta
    _max = rng.integers(0, 4096, shape) * _delta + _min
    interval_arr = QIntervalArray(_min, _max, _delta)
    return interval_arr


@pytest.fixture(scope='module')
def qint_arr1():
    return random_arr()


@pytest.fixture(scope='module')
def qint_arr2():
    return random_arr()


@pytest.mark.parametrize('oprstr', ['__add__', '__sub__', '__mul__', '__matmul__', '__rmatmul__'])
def test_qinterval_oprs(qint_arr1, qint_arr2, oprstr):

    sampled_arr1 = qint_arr1.sample(10000)
    const_arr = qint_arr2.sample()
    applied_symbolic = getattr(qint_arr1, oprstr)(const_arr)
    applied_sampled = getattr(sampled_arr1, oprstr)(const_arr)

    assert_is_represented(applied_symbolic, applied_sampled)

    if oprstr != '__rmatmul__':
        # rmatmul is only between const and intervals.

        sampled_arr2 = qint_arr2.sample(10000)
        rapplied_symbolic = getattr(qint_arr1, oprstr)(qint_arr2)
        rapplied_sampled = getattr(sampled_arr1, oprstr)(sampled_arr2)

        assert_is_represented(rapplied_symbolic, rapplied_sampled)


@pytest.mark.parametrize('eq', ['ij,jk->ik', 'ij,kj->ikj'])
def test_qinterval_einsum(qint_arr1, qint_arr2, eq):

    _in, out = eq.split('->', 1)
    in0, in1 = _in.split(',', 1)
    qint_arr1 = qint_arr1[:16, :16]
    qint_arr2 = qint_arr2[:16, :16]

    sampled_arr1 = qint_arr1.sample(10000)
    sampled_arr2 = qint_arr2.sample(10000)

    # symbolic - symbolic
    einsum_symbolic = einsum(eq, qint_arr1, qint_arr2)
    einsum_sampled = np.einsum(f'A{in0},A{in1}->A{out}', sampled_arr1, sampled_arr2)
    assert_is_represented(einsum_symbolic, einsum_sampled)

    # symbolic - sampled
    einsum_symbolic = einsum(eq, qint_arr1, sampled_arr2[0])
    einsum_sampled = np.einsum(f'A{in0},{in1}->A{out}', sampled_arr1, sampled_arr2[0])
    assert_is_represented(einsum_symbolic, einsum_sampled)

    # sampled - symbolic
    einsum_symbolic = einsum(eq, sampled_arr1[0], qint_arr2)
    einsum_sampled = np.einsum(f'{in0},A{in1}->A{out}', sampled_arr1[0], sampled_arr2)
    assert_is_represented(einsum_symbolic, einsum_sampled)


def test_qinterval_to_kif(qint_arr1):
    k, i, f = qint_arr1.to_kif()
    samples = qint_arr1.sample(10000)
    q = get_fixed_quantizer_np()
    assert np.all(samples == q(samples, k, i, f))
