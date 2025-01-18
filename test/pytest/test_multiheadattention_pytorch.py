from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import hls4ml

test_root_path = Path(__file__).parent

batch_size = 100
seq_len = 10
num_heads = 2
embed_dim = 8

atol = 2e-2


@pytest.fixture(scope='module')
def query_data():
    return np.random.rand(batch_size, seq_len, embed_dim)


@pytest.fixture(scope='module')
def key_value_data():
    return np.random.rand(batch_size, seq_len, embed_dim)


class MultiHeadAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        output, _ = self.mha(query, key, value)
        return output


# Currently only Vitis in io_parallel mode is supported
def test_multiheadattention(query_data, key_value_data):
    model = MultiHeadAttentionModel()
    model.eval()

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        [(seq_len, embed_dim), (seq_len, embed_dim), (seq_len, embed_dim)],
        granularity='name',
        backend='Vitis',
        channels_last_conversion='off',
        transpose_outputs=False,
    )
    output_dir = str(test_root_path / 'hls4mlprj_multiheadattention_pytorch_Vitis_io_parallel')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, backend='Vitis', hls_config=config, io_type='io_parallel', output_dir=output_dir
    )
    hls_model.compile()

    # Predict
    y_pytorch = (
        model(torch.Tensor(query_data), torch.Tensor(key_value_data), torch.Tensor(key_value_data))
        .detach()
        .numpy()
        .flatten()
    )
    y_hls = hls_model.predict([query_data, key_value_data, key_value_data]).flatten()
    np.testing.assert_allclose(y_pytorch, y_hls, rtol=0, atol=atol, verbose=True)
