from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Activation, Concatenate, Input, Permute
from tensorflow.keras.models import Model

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def data():
    X = np.random.rand(100, 2, 3)
    return X


@pytest.fixture(scope='module')
def keras_model():
    inp = Input(shape=(2, 3), name='input_1')
    x = Permute((2, 1))(inp)
    y = Concatenate(axis=1)([x, x])
    x = Activation('relu', name='relu')(x)
    out = Concatenate(axis=1)([x, y])
    model = Model(inputs=inp, outputs=out)
    return model


@pytest.fixture
@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
def hls_model(keras_model, backend, io_type):
    hls_config = hls4ml.utils.config_from_keras_model(
        keras_model, default_precision='ap_fixed<16,3,AP_RND_CONV,AP_SAT>', granularity='name', backend=backend
    )
    hls_config['LayerName']['relu']['Precision'] = 'ap_ufixed<17,3>'
    output_dir = str(test_root_path / f'hls4mlprj_transpose_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=hls_config, io_type=io_type, backend=backend, output_dir=output_dir
    )

    hls_model.compile()
    return hls_model


@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
def test_accuracy(data, keras_model, hls_model):
    X = data
    model = keras_model
    # model under test predictions and accuracy
    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X).reshape(y_keras.shape)
    # "accuracy" of hls4ml predictions vs keras
    np.testing.assert_allclose(y_keras, y_hls4ml, rtol=0, atol=1e-04, verbose=True)


@pytest.fixture(scope='module')
def keras_model_highdim():
    inp = Input(shape=(2, 3, 4, 5, 6), name='input_1')
    out = Permute((3, 5, 4, 1, 2))(inp)
    model = Model(inputs=inp, outputs=out)
    return model


@pytest.fixture(scope='module')
def data_highdim():
    X = np.random.randint(-128, 127, (100, 2, 3, 4, 5, 6)) / 128
    X = X.astype(np.float32)
    return X


@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'oneAPI'])
def test_highdim_permute(data_highdim, keras_model_highdim, io_type, backend):
    X = data_highdim
    model = keras_model_highdim

    model_hls = hls4ml.converters.convert_from_keras_model(
        model,
        io_type=io_type,
        backend=backend,
        output_dir=str(test_root_path / f'hls4mlprj_highdim_transpose_{backend}_{io_type}'),
    )
    model_hls.compile()
    y_keras = model.predict(X)
    y_hls4ml = model_hls.predict(X).reshape(y_keras.shape)  # type: ignore

    assert np.all(y_keras == y_hls4ml)
