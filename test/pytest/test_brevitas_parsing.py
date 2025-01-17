from pathlib import Path

import brevitas.nn as qnn
import numpy as np
import pytest
import torch
from brevitas.quant import Int8WeightPerTensorFixedPoint
from torch import nn
from torch.nn import Module

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent


class QuantModelConv2d(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_quant=Int8WeightPerTensorFixedPoint)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out


class QuantModelConv1d(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = qnn.QuantConv1d(3, 6, 4, bias=True, weight_quant=Int8WeightPerTensorFixedPoint)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out


class QuantModelLinear(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = qnn.QuantLinear(4, 4, bias=True, weight_quant=Int8WeightPerTensorFixedPoint)
        self.relu1 = qnn.QuantReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return out


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_quantlinear(backend, io_type):
    model = QuantModelLinear()

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    pytorch_prediction = model(x).detach().numpy()
    config = config_from_pytorch_model(model, input_shape=(None, 4))
    output_dir = str(test_root_path / f'hls4mlprj_brevitas_linear_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )
    hls_model.compile()

    hls_prediction = np.reshape(hls_model.predict(x.detach().numpy()), pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0.0, atol=0.05)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_quantconv1d(backend, io_type):
    model = QuantModelConv1d()

    n_in = 3
    n_out = 6
    size_in = 5

    x = torch.randn(1, n_in, size_in)

    pytorch_prediction = model(x).detach().numpy()
    if io_type == 'io_stream':
        x = np.ascontiguousarray(x.permute(0, 2, 1))
        config = config_from_pytorch_model(
            model, (None, n_in, size_in), channels_last_conversion="internal", transpose_outputs=False
        )
    else:
        config = config_from_pytorch_model(
            model, (None, n_in, size_in), channels_last_conversion="full", transpose_outputs=True
        )

    output_dir = str(test_root_path / f'hls4mlprj_brevitas_conv1d_{backend}_{io_type}')

    from hls4ml.converters.pytorch_to_hls import CustomFXTracer

    tracer = CustomFXTracer()
    traced_model = tracer.trace(model)
    nNodes = 0
    convNode = None
    for _node in traced_model.nodes:
        nNodes += 1
        if nNodes == 2:
            convNode = _node

    children = {c[0]: c[1] for c in model.named_children()}
    class_object_conv = children[convNode.target]

    out_width = int(
        (
            size_in
            + 2 * class_object_conv.padding[0]
            - class_object_conv.dilation[0] * (class_object_conv.kernel_size[0] - 1)
            - 1
        )
        / class_object_conv.stride[0]
        + 1
    )  # following https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

    hls_model = convert_from_pytorch_model(model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type)
    hls_model.compile()

    if io_type == 'io_stream':
        hls_prediction = np.transpose(np.reshape(hls_model.predict(x), (1, out_width, n_out)), (0, 2, 1))
    else:
        hls_prediction = np.reshape(hls_model.predict(x.detach().numpy()), pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_quantconv2d(backend, io_type):
    model = QuantModelConv2d()

    n_in = 3
    n_out = 6
    size_in_width = 5
    size_in_height = 6

    x = torch.randn(1, n_in, size_in_height, size_in_width)

    pytorch_prediction = model(x).detach().numpy()
    if io_type == 'io_stream':
        x = np.ascontiguousarray(x.permute(0, 2, 3, 1))
        config = config_from_pytorch_model(
            model, (None, n_in, size_in_height, size_in_width), channels_last_conversion="internal", transpose_outputs=False
        )
    else:
        config = config_from_pytorch_model(
            model, (None, n_in, size_in_height, size_in_width), channels_last_conversion="full", transpose_outputs=True
        )

    output_dir = str(test_root_path / f'hls4mlprj_brevitas_conv2d_{backend}_{io_type}')

    from hls4ml.converters.pytorch_to_hls import CustomFXTracer

    tracer = CustomFXTracer()
    traced_model = tracer.trace(model)

    nNodes = 0
    convNode = None
    for _node in traced_model.nodes:
        nNodes += 1
        if nNodes == 2:
            convNode = _node

    children = {c[0]: c[1] for c in model.named_children()}
    class_object_conv = children[convNode.target]

    out_width = int(
        (
            size_in_width
            + 2 * class_object_conv.padding[1]
            - class_object_conv.dilation[1] * (class_object_conv.kernel_size[1] - 1)
            - 1
        )
        / class_object_conv.stride[1]
        + 1
    )  # following https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    out_height = int(
        (
            size_in_height
            + 2 * class_object_conv.padding[0]
            - class_object_conv.dilation[0] * (class_object_conv.kernel_size[0] - 1)
            - 1
        )
        / class_object_conv.stride[0]
        + 1
    )  # following https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    hls_model = convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )
    hls_model.compile()

    if io_type == 'io_stream':
        hls_prediction = np.transpose(np.reshape(hls_model.predict(x), (1, out_height, out_width, n_out)), (0, 3, 1, 2))
    else:
        hls_prediction = np.reshape(hls_model.predict(x.detach().numpy()), pytorch_prediction.shape)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0.0, atol=0.05)
