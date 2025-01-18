from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

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
    model = nn.Sequential(nn.LayerNorm(in_shape[-1]))
    model.eval()
    return model


# Currently only Vivado/Vitis in io_parallel mode is supported
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_layernorm(model, data, backend):
    config = hls4ml.utils.config_from_pytorch_model(model, in_shape, granularity='name', backend=backend)
    output_dir = str(test_root_path / f'hls4mlprj_layernorm_pytorch_{backend}_io_parallel')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, backend=backend, hls_config=config, io_type='io_parallel', output_dir=output_dir
    )
    hls_model.compile()

    # Predict
    y_pytorch = model(torch.Tensor(data)).detach().numpy().flatten()
    y_hls = hls_model.predict(data).flatten()
    np.testing.assert_allclose(y_pytorch, y_hls, rtol=0, atol=atol, verbose=True)
