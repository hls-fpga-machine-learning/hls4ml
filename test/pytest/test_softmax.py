from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import accuracy_score

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture()
def generate_data(input_shape):
    shape = (5000, *input_shape)
    d = np.random.normal(0, 2, shape)
    modify_entries = np.random.randint(0, 1, shape) < 0.05
    d[modify_entries] = d[modify_entries] * 5 + 10
    return np.clip(d, -32, 31)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'Catapult'])
@pytest.mark.parametrize('strategy', ['stable', 'latency', 'argmax'])
@pytest.mark.parametrize(
    'input_bits,input_shape,table_bits,io_type,custom_accum',
    [
        ('16,6', (8,), '18,8', 'io_parallel', False),
        ('16,6', (8,), '18,8', 'io_stream', False),
        ('16,6', (8,), '18,8', 'io_parallel', True),
        ('16,6', (8,), '18,8', 'io_stream', True),
        ('16,6', (8,), '9,6', 'io_parallel', False),
        ('16,6', (8,), '9,6', 'io_stream', False),
        ('9,6', (8,), '18,8', 'io_parallel', False),
        ('9,6', (8,), '18,8', 'io_stream', False),
        ('16,6', (8, 8, 3), '18,8', 'io_stream', False),
    ],
)
def test_softmax(backend, strategy, generate_data, input_bits, input_shape, table_bits, io_type, custom_accum):
    X = generate_data
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Activation(input_shape=input_shape, activation='softmax', name='softmax'))
    model.compile()

    table_type = f'fixed<{table_bits}, RND, SAT>'

    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    cfg['LayerName']['softmax']['Strategy'] = strategy
    cfg['LayerName']['softmax']['inv_table_t'] = table_type
    cfg['LayerName']['softmax']['exp_table_t'] = table_type
    cfg['LayerName']['softmax']['accum_t'] = table_type
    cfg['LayerName']['softmax']['inv_inp_t'] = table_type
    if custom_accum:
        if backend not in ['Vivado', 'Vitis']:
            pytest.skip('Custom accumulators are only supported for Vivado and Vitis backends')
        W, I = map(int, input_bits.split(','))  # noqa: E741
        cfg['LayerName']['softmax']['accum_t'] = f'fixed<{W+3},{I+3}>'
        cfg['LayerName']['softmax']['inv_inp_t'] = f'fixed<{W+2},{I+2}>'
    inp_layer_name = next(iter(cfg['LayerName'].keys()))
    cfg['LayerName'][inp_layer_name]['Precision']['result'] = f'fixed<{input_bits}>'

    odir = str(
        test_root_path
        / (
            f'hls4mlprj_softmax_{backend}_{io_type}_{strategy}_{input_shape}'
            f'_input-bits={input_bits}_table-bits={table_bits}_custom-accum={custom_accum}'
        )
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)
    acc_hls4ml = accuracy_score(np.argmax(y_keras, axis=-1).ravel(), np.argmax(y_hls4ml, axis=-1).ravel())

    print(f'Accuracy hls4ml relative to keras: {acc_hls4ml}')

    assert acc_hls4ml >= 0.98


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'Catapult'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_softmax_skipped(backend, io_type):
    X = np.random.rand(100, 10)
    dense = tf.keras.layers.Dense(14, input_shape=(10,), name='dense')
    softmax = tf.keras.layers.Activation(activation='softmax', name='softmax')
    model = tf.keras.models.Sequential([dense, softmax])
    model.compile()

    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    cfg['LayerName']['softmax']['skip'] = True

    odir = str(test_root_path / 'hls4mlprj_softmax_skipped_{}_{}').format(backend, io_type)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type=io_type, output_dir=odir, backend=backend
    )
    hls_model.compile()

    # Verify Softmax was removed
    hls_layers = list(hls_model.get_layers())  # 0 is Input, 1 is Dense, 2 is Softmax (if not removed)
    assert len(hls_layers) == 2

    # Verify hls4ml output is equal to Dense output
    y_keras_dense = dense(X).numpy()  # type: ignore
    y_hls4ml = hls_model.predict(X).reshape(y_keras_dense.shape)  # type: ignore
    np.testing.assert_allclose(y_hls4ml, y_keras_dense, rtol=0, atol=2e-2)
