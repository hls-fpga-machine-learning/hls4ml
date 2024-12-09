from pathlib import Path

import HGQ  # noqa: F401
import numpy as np
import pytest
import tensorflow as tf
from HGQ import get_default_paq_conf, set_default_paq_conf, trace_minmax
from HGQ.layers import (  # noqa: F401
    HConv1D,
    HDense,
    HQuantize,
    PAvgPool1D,
    PAvgPool2D,
    PConcatenate,
    PFlatten,
    PMaxPool1D,
    PMaxPool2D,
    PReshape,
    Signature,
)
from HGQ.proxy import to_proxy_model
from tensorflow import keras

from hls4ml.converters import convert_from_keras_model

# tf.config.experimental_run_functions_eagerly(True)  # noqa


test_path = Path(__file__).parent


def _run_synth_match_test(proxy: keras.Model, data, io_type: str, backend: str, dir: str, cond=None):

    output_dir = dir + '/hls4ml_prj'
    hls_model = convert_from_keras_model(
        proxy,
        io_type=io_type,
        output_dir=output_dir,
        backend=backend,
        hls_config={'Model': {'Precision': 'fixed<1,0>', 'ReuseFactor': 1}},
    )
    hls_model.compile()

    data_len = data.shape[0] if isinstance(data, np.ndarray) else data[0].shape[0]
    # Multiple output case. Check each output separately
    if len(proxy.outputs) > 1:  # type: ignore
        r_proxy: list[np.ndarray] = [x.numpy() for x in proxy(data)]  # type: ignore
        r_hls: list[np.ndarray] = hls_model.predict(data)  # type: ignore
        r_hls = [x.reshape(r_proxy[i].shape) for i, x in enumerate(r_hls)]
    else:
        r_proxy: list[np.ndarray] = [proxy(data).numpy()]  # type: ignore
        r_hls: list[np.ndarray] = [hls_model.predict(data).reshape(r_proxy[0].shape)]  # type: ignore

    errors = []
    for i, (p, h) in enumerate(zip(r_proxy, r_hls)):
        try:
            if cond is None:
                mismatch_ph = p != h
                assert (
                    np.sum(mismatch_ph) == 0
                ), f"Proxy-HLS4ML mismatch for out {i}: {np.sum(np.any(mismatch_ph, axis=1))} out of {data_len} samples are different. Sample: {p[mismatch_ph].ravel()[:5]} vs {h[mismatch_ph].ravel()[:5]}"  # noqa: E501
            else:
                cond(p, h)
        except AssertionError as e:
            errors.append(e)
    if len(errors) > 0:
        msgs = [str(e) for e in errors]
        raise AssertionError('\n'.join(msgs))


def run_model_test(
    model: keras.Model, cover_factor: float | None, data, io_type: str, backend: str, dir: str, aggressive: bool, cond=None
):
    data_len = data.shape[0] if isinstance(data, np.ndarray) else data[0].shape[0]
    if cover_factor is not None:
        trace_minmax(model, data, cover_factor=cover_factor, bsz=data_len)
    proxy = to_proxy_model(model, aggressive=aggressive, unary_lut_max_table_size=4096)
    _run_synth_match_test(proxy, data, io_type, backend, dir, cond=cond)


def create_hlayer_model(layer: str, rnd_strategy: str, io_type: str):
    pa_config = get_default_paq_conf()
    pa_config['rnd_strategy'] = rnd_strategy
    pa_config['skip_dims'] = 'all' if io_type == 'io_stream' else 'batch'
    set_default_paq_conf(pa_config)

    inp = keras.Input(shape=(16))
    if 'Add' in layer:
        _inp = [HQuantize()(inp)] * 2
    elif 'Conv2D' in layer:
        _inp = PReshape((4, 4, 1))(HQuantize()(inp))
    elif 'Conv1D' in layer:
        _inp = PReshape((16, 1))(HQuantize()(inp))
    elif 'Dense' in layer or 'Activation' in layer:
        _inp = HQuantize()(inp)
    else:
        raise Exception(f'Please add test for {layer}')

    _layer = eval('HGQ.layers.' + layer)
    if hasattr(_layer, 'bias') and _layer.bias is not None:
        bias: tf.Variable = _layer.bias
        bias.assign(tf.constant(np.random.uniform(-4, 4, _layer.bias.shape).astype(np.float32)))

    out = _layer(_inp)
    model = keras.Model(inp, out)

    for layer in model.layers:
        # Randomize weight bitwidths
        if hasattr(layer, 'kq'):
            fbw: tf.Variable = layer.kq.fbw
            fbw.assign(tf.constant(np.random.uniform(2, 6, fbw.shape).astype(np.float32)))
        # And activation bitwidths
        if hasattr(layer, 'paq'):
            fbw: tf.Variable = layer.paq.fbw
            fbw.assign(tf.constant(np.random.uniform(2, 6, fbw.shape).astype(np.float32)))

    return model


def get_data(shape: tuple[int, ...], v: float, max_scale: float):
    rng = np.random.default_rng()
    a1 = rng.uniform(-v, v, shape).astype(np.float32)
    a2 = rng.uniform(0, max_scale, (1, shape[1])).astype(np.float32)
    return (a1 * a2).astype(np.float32)


def softmax_cond(proxy, hls):
    match_precent = np.mean(np.argmax(proxy, axis=1) == np.argmax(hls, axis=1))
    assert (
        match_precent > 0.90
    ), f"Proxy-HLS4ML mismatch: {(1-match_precent) * 100}% of samples are different. Sample: {proxy[:5]} vs {hls[:5]}"


def custom_activation_fn(x):
    return tf.sin(x) ** 2.0 - x  # type: ignore


@pytest.mark.parametrize(
    'layer',
    [
        "HDense(10)",
        "HDense(10, use_bias=False)",
        "HDenseBatchNorm(10)",
        "HConv1D(2, 3, padding='same')",
        "HConv1D(2, 3, padding='valid')",
        "HConv1D(2, 3, padding='valid', use_bias=False)",
        "HConv1D(2, 3, padding='valid', strides=2)",
        "HConv1D(2, 3, padding='same', strides=2)",
        "HConv1DBatchNorm(2, 3, padding='valid')",
        "HConv2D(2, (3,3), padding='same')",
        "HConv2D(2, (3,3), padding='valid')",
        "HConv2D(2, (3,3), padding='valid', use_bias=False)",
        "HConv2D(2, (3,3), padding='valid', strides=2)",
        "HConv2D(2, (3,3), padding='same', strides=2)",
        "HConv2DBatchNorm(2, (3,3), padding='valid')",
        "HAdd()",
        "HActivation('relu')",
        #   "HActivation('leaky_relu')",
        "HActivation('tanh')",
        "HActivation('sigmoid')",
        # "HActivation('softmax')",
        "HActivation(custom_activation_fn)",
    ],
)
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("rnd_strategy", ['standard_round', 'floor'])
@pytest.mark.parametrize("io_type", ['io_parallel', 'io_stream'])
@pytest.mark.parametrize("cover_factor", [1.0])
@pytest.mark.parametrize("aggressive", [True, False])
@pytest.mark.parametrize("backend", ['vivado', 'vitis'])
def test_syn_hlayers(layer, N: int, rnd_strategy: str, io_type: str, cover_factor: float, aggressive: bool, backend: str):
    model = create_hlayer_model(layer=layer, rnd_strategy=rnd_strategy, io_type=io_type)
    data = get_data((N, 16), 7, 1)

    cond = None if 'softmax' not in layer else softmax_cond
    path = test_path / f'hls4mlprj_hgq_{layer}_{rnd_strategy}_{io_type}_{aggressive}_{backend}'

    run_model_test(model, cover_factor, data, io_type, backend, str(path), aggressive, cond=cond)
