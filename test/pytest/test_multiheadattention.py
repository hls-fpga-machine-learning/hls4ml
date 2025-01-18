from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, MultiHeadAttention

import hls4ml

test_root_path = Path(__file__).parent

batch_size = 100
seq_len = 10
num_heads = 2
key_dim = 4

atol = 2e-2


@pytest.fixture(scope='module')
def query_data():
    return np.random.rand(batch_size, seq_len, num_heads * key_dim)


@pytest.fixture(scope='module')
def key_value_data():
    return np.random.rand(batch_size, seq_len, num_heads * key_dim)


@pytest.fixture(scope='module')
def model():
    query_input = Input(shape=(seq_len, num_heads * key_dim))
    key_value_input = Input(shape=(seq_len, num_heads * key_dim))
    mha_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(query_input, key_value_input)
    model = Model(inputs=[query_input, key_value_input], outputs=mha_layer)
    model.compile()
    return model


# Currently only Vitis in io_parallel mode is supported
def test_multiheadattention(model, query_data, key_value_data):
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vitis')
    output_dir = str(test_root_path / 'hls4mlprj_multiheadattention_Vitis_io_parallel')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend='Vitis', hls_config=config, io_type='io_parallel', output_dir=output_dir
    )
    hls_model.compile()

    # Predict
    y_keras = model.predict([query_data, key_value_data]).flatten()
    y_hls = hls_model.predict([query_data, key_value_data]).flatten()
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
