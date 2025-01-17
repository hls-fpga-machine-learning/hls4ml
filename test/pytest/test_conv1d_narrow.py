from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def data():
    X = np.random.rand(10, 11, 3)
    return X


@pytest.fixture(scope='module')
def model():
    model = Sequential()
    model.add(Conv1D(5, 9, input_shape=(11, 3)))
    model.compile()
    return model


@pytest.mark.parametrize(
    'narrowset',
    [
        ('io_stream', 'latency', 'Encoded'),
        ('io_stream', 'resource', 'Encoded'),
        ('io_stream', 'latency', 'LineBuffer'),
        ('io_stream', 'resource', 'LineBuffer'),
        ('io_parallel', 'resource', 'Encoded'),
        ('io_parallel', 'latency', 'Encoded'),
        ('io_parallel', 'resource', 'LineBuffer'),
        ('io_parallel', 'latency', 'LineBuffer'),
    ],
)
@pytest.mark.filterwarnings("error")
def test_narrow(data, model, narrowset, capfd):
    '''
    Check that the implementation does not have leftover data.
    '''
    io_type = narrowset[0]
    strategy = narrowset[1]
    conv = narrowset[2]
    X = data

    output_dir = str(test_root_path / f'hls4mlprj_conv1d_narrow_{io_type}_{strategy}_{conv}')

    config = hls4ml.utils.config_from_keras_model(model)
    config['Model']['Strategy'] = strategy
    config['Model']['ConvImplementation'] = conv

    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, io_type=io_type, output_dir=output_dir)
    hls_model.compile()

    # model under test predictions and accuracy
    y_keras = model.predict(X)
    y_hls4ml = hls_model.predict(X)

    out, _ = capfd.readouterr()
    assert "leftover data" not in out
    np.testing.assert_allclose(y_keras.ravel(), y_hls4ml.ravel(), atol=0.05)
