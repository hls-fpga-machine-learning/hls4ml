from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent


class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(10, 20, num_layers=1, batch_first=True, bias=True)

    def forward(self, x, h0):
        output, hnn = self.rnn(x, h0)
        return output


class GRUNetStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(10, 20, num_layers=1, batch_first=True, bias=True)

    def forward(self, x):
        output, hnn = self.rnn(x)
        return output


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_gru(backend, io_type):
    model = GRUNet()
    model.eval()

    X_input = torch.randn(1, 1, 10)
    h0 = torch.randn(1, 1, 20)

    pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(h0)).detach().numpy()

    config = config_from_pytorch_model(
        model, [(None, 1, 10), (None, 1, 20)], channels_last_conversion="off", transpose_outputs=False
    )
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_gru_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict([X_input.detach().numpy(), h0.detach().numpy()]), (1, 1, 20))

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=1e-1)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_stream'])
def test_gru_stream(backend, io_type):
    model = GRUNetStream()
    model.eval()

    X_input = torch.randn(1, 1, 10)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model, (None, 1, 10), channels_last_conversion="off", transpose_outputs=False)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_gru_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)

    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(X_input.detach().numpy()), (1, 1, 20))

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=1e-1)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(10, 20, num_layers=1, batch_first=True, bias=True)

    def forward(self, x, h0, c0):
        output, (_, _) = self.rnn(x, (h0, c0))
        return output


class LSTMStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(10, 20, num_layers=1, batch_first=True, bias=True)

    def forward(self, x):
        output, (_, _) = self.rnn(x)
        return output


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_lstm(backend, io_type):
    model = LSTM()
    model.eval()

    X_input = torch.randn(1, 1, 10)
    h0 = torch.randn(1, 1, 20)
    c0 = torch.randn(1, 1, 20)

    pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(h0), torch.tensor(c0)).detach().numpy()

    config = config_from_pytorch_model(
        model, [(None, 1, 10), (None, 1, 20), (None, 1, 20)], channels_last_conversion="off", transpose_outputs=False
    )
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_lstm_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )

    hls_model.compile()

    hls_prediction = np.reshape(
        hls_model.predict([X_input.detach().numpy(), h0.detach().numpy(), c0.detach().numpy()]), (1, 1, 20)
    )

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=1e-1)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_stream'])
def test_lstm_stream(backend, io_type):
    if not (backend in ('Quartus', 'oneAPI') and io_type == "io_stream"):
        model = LSTMStream()
        model.eval()

        X_input = torch.randn(1, 1, 10)

        pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

        config = config_from_pytorch_model(model, [(None, 1, 10)], channels_last_conversion="off", transpose_outputs=False)
        output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_lstm_{backend}_{io_type}')

        hls_model = convert_from_pytorch_model(
            model,
            hls_config=config,
            output_dir=output_dir,
            backend=backend,
            io_type=io_type,
        )

        hls_model.compile()

        hls_prediction = np.reshape(hls_model.predict(X_input.detach().numpy()), (1, 1, 20))

        np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=1e-1)


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(10, 20, num_layers=1, batch_first=True, bias=True)

    def forward(self, x, h0):
        output, _ = self.rnn(x, h0)
        return output


@pytest.mark.parametrize('backend', ['Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
def test_rnn(backend, io_type):
    if not (backend in ('Quartus', 'oneAPI') and io_type == "io_stream"):
        model = RNN()
        model.eval()

        X_input = torch.randn(1, 1, 10)
        h0 = torch.zeros(1, 1, 20)

        pytorch_prediction = model(torch.Tensor(X_input), torch.Tensor(h0)).detach().numpy()

        config = config_from_pytorch_model(
            model, [(1, 10), (1, 20)], channels_last_conversion="off", transpose_outputs=False
        )
        output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_rnn_{backend}_{io_type}')

        hls_model = convert_from_pytorch_model(
            model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
        )

        hls_model.compile()

        hls_prediction = np.reshape(hls_model.predict([X_input.detach().numpy(), h0.detach().numpy()]), (1, 1, 20))

        np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0, atol=1e-1)
