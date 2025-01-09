from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

in_shape = (10, 8)
atol = 5e-2


@pytest.fixture(scope='module')
def data():
    np.random.seed(0)
    return np.random.rand(100, *in_shape)


@pytest.fixture(scope='module')
def model():
    model = Sequential()
    model.add(LayerNormalization(input_shape=in_shape))
    model.compile()
    return model


# Currently only Vivado in io_parallel mode is supported
def test_layernorm(model, data):
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vivado')
    output_dir = str(test_root_path / 'hls4mlprj_layernorm_Vivado_io_parallel')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend='Vivado', hls_config=config, io_type='io_parallel', output_dir=output_dir
    )
    hls_model.compile()

    # Predict
    y_keras = model.predict(data).flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
