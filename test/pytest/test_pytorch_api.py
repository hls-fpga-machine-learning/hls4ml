from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.nn import AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d

from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils.config import config_from_pytorch_model

test_root_path = Path(__file__).parent


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_linear(backend, io_type):

    model = LinearModel()
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_linear_{backend}_{io_type}')

    hls_model = convert_from_pytorch_model(
        model, (None, 1), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)

    nNodes = 0
    for _node in traced_model.graph.nodes:
        nNodes += 1

    assert nNodes - 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes["class_name"] == "Dense"
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == [1]
    assert list(hls_model.get_layers())[1].attributes['n_in'] == 1
    assert list(hls_model.get_layers())[1].attributes['n_out'] == 1


# TODO: add ThresholdedReLU test when it can be made to pass
@pytest.mark.parametrize(
    "activation_function",
    [
        nn.ReLU(),
        nn.LeakyReLU(negative_slope=1.0),
        nn.ELU(alpha=1.0),
        nn.PReLU(init=0.25),
        nn.Sigmoid(),
        nn.Threshold(threshold=1.0, value=0.0),
    ],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_activations(activation_function, backend, io_type):

    model = torch.nn.Sequential(nn.Linear(1, 1), activation_function).to()
    model.eval()

    X_input = np.random.rand(1)

    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(
        test_root_path / f'hls4mlprj_pytorch_api_activations_{activation_function.__class__.__name__}_{backend}_{io_type}'
    )
    hls_model = convert_from_pytorch_model(
        model, (None, 1), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=1e-2, atol=0.01)

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)

    nNodes = 0
    for _node in traced_model.graph.nodes:
        nNodes += 1

    assert nNodes - 1 == len(hls_model.get_layers())

    if activation_function.__class__.__name__ == 'ReLU' or activation_function.__class__.__name__ == 'Sigmoid':
        assert list(hls_model.get_layers())[2].attributes['class_name'] == 'Activation'
    elif activation_function.__class__.__name__ == 'Threshold':
        assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ThresholdedReLU'
    else:
        assert list(hls_model.get_layers())[2].attributes['class_name'] == activation_function.__class__.__name__


padds_options = [0, 1]


@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_conv1d(padds, backend, io_type):

    n_in = 2
    n_out = 2
    kernel_size = 3
    size_in = 4

    model = torch.nn.Sequential(nn.Conv1d(n_in, n_out, kernel_size, padding=padds), nn.ReLU()).to()
    model.eval()

    X_input = np.random.rand(1, n_in, size_in)
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_conv1d_{padds}_{backend}_{io_type}')
    hls_model = convert_from_pytorch_model(
        model, (None, n_in, size_in), hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()
    if padds == 0:
        hls_prediction = np.reshape(hls_model.predict(X_input), (1, n_out, size_in - 2))
    else:
        hls_prediction = np.reshape(hls_model.predict(X_input), (1, n_out, size_in))
    # results are not very good at the moment
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0.20, atol=0)

    if not (backend == 'Vivado' and io_type == 'io_stream' and padds == 1):
        # Vivado inserts and additional layer for 'same' padding in io_stream

        from torch.fx import symbolic_trace

        traced_model = symbolic_trace(model)
        nNodes = 0
        convNode = None
        reluNode = None
        for _node in traced_model.graph.nodes:
            nNodes += 1
            if nNodes == 2:
                convNode = _node
            if nNodes == 3:
                reluNode = _node
        assert nNodes + 1 == len(hls_model.get_layers())

        children = {c[0]: c[1] for c in model.named_children()}
        class_object_conv = children[convNode.target]
        class_object_relu = children[reluNode.target]
        assert list(hls_model.get_layers())[2].attributes['name'] == 'layer' + convNode.name
        assert list(hls_model.get_layers())[2].attributes['class_name'] == 'Conv1D'
        assert list(hls_model.get_layers())[4].attributes['activation'] == class_object_relu.__class__.__name__
        assert list(hls_model.get_layers())[2].attributes["in_width"] == size_in
        assert list(hls_model.get_layers())[2].attributes['filt_width'] == class_object_conv.kernel_size[0]
        assert list(hls_model.get_layers())[2].attributes['n_chan'] == class_object_conv.in_channels
        assert list(hls_model.get_layers())[2].attributes['n_filt'] == class_object_conv.out_channels
        assert list(hls_model.get_layers())[2].attributes['stride_width'] == class_object_conv.stride[0]
        if list(hls_model.get_layers())[2].attributes['padding'] == 'valid':
            padding = 0
        else:
            padding = 1
        assert padding == class_object_conv.padding[0]
        assert list(hls_model.get_layers())[2].attributes['data_format'] == 'channels_last'
        out_width = int(
            (size_in + 2 * padds - class_object_conv.dilation[0] * (class_object_conv.kernel_size[0] - 1) - 1)
            / class_object_conv.stride[0]
            + 1
        )  # following https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        assert list(hls_model.get_layers())[2].attributes["out_width"] == out_width

        pad_along_width = max((out_width - 1) * class_object_conv.stride[0] + class_object_conv.kernel_size[0] - size_in, 0)
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        if padds == 1:
            assert list(hls_model.get_layers())[2].attributes['pad_left'] == pad_left
            assert list(hls_model.get_layers())[2].attributes['pad_right'] == pad_right
        elif padds == 0:
            assert list(hls_model.get_layers())[2].attributes['pad_left'] == 0
            assert list(hls_model.get_layers())[2].attributes['pad_right'] == 0


padds_options = [0, 1]


@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_conv2d(padds, backend, io_type):

    n_in = 2
    n_out = 2
    kernel_size = 3
    size_in_width = 4
    size_in_height = 4

    model = torch.nn.Sequential(nn.Conv2d(n_in, n_out, kernel_size, padding=padds), nn.ReLU()).to()
    model.eval()

    X_input = np.random.rand(1, n_in, size_in_height, size_in_width)
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_conv2d_{padds}_{backend}_{io_type}')
    hls_model = convert_from_pytorch_model(
        model,
        (None, n_in, size_in_height, size_in_width),
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )
    hls_model.compile()

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)
    nNodes = 0
    convNode = None
    reluNode = None
    for _node in traced_model.graph.nodes:
        nNodes += 1
        if nNodes == 2:
            convNode = _node
        if nNodes == 3:
            reluNode = _node
    assert nNodes + 1 == len(hls_model.get_layers())

    children = {c[0]: c[1] for c in model.named_children()}
    class_object_conv = children[convNode.target]
    class_object_relu = children[reluNode.target]

    out_width = int(
        (size_in_width + 2 * padds - class_object_conv.dilation[1] * (class_object_conv.kernel_size[1] - 1) - 1)
        / class_object_conv.stride[1]
        + 1
    )  # following https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    assert list(hls_model.get_layers())[2].attributes["out_width"] == out_width
    out_height = int(
        (size_in_height + 2 * padds - class_object_conv.dilation[0] * (class_object_conv.kernel_size[0] - 1) - 1)
        / class_object_conv.stride[0]
        + 1
    )  # following https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    assert list(hls_model.get_layers())[2].attributes["out_height"] == out_height

    if padds == 0:
        hls_prediction = np.reshape(hls_model.predict(X_input), (1, n_out, out_height, out_width))

    else:
        hls_prediction = np.reshape(hls_model.predict(X_input), (1, n_out, out_height, out_width))
    # results are not very good at the moment
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0.20, atol=0)

    if not (backend == 'Vivado' and io_type == 'io_stream' and padds == 1):
        # Vivado inserts and additional layer for 'same' padding in io_stream

        assert list(hls_model.get_layers())[2].attributes['name'] == 'layer' + convNode.name
        assert list(hls_model.get_layers())[2].attributes['class_name'] == 'Conv2D'
        assert list(hls_model.get_layers())[3].attributes['activation'] == class_object_relu.__class__.__name__
        assert list(hls_model.get_layers())[2].attributes["in_width"] == size_in_width
        assert list(hls_model.get_layers())[2].attributes["in_height"] == size_in_height
        assert list(hls_model.get_layers())[2].attributes['filt_width'] == class_object_conv.kernel_size[1]
        assert list(hls_model.get_layers())[2].attributes['filt_height'] == class_object_conv.kernel_size[0]
        assert list(hls_model.get_layers())[2].attributes['n_chan'] == class_object_conv.in_channels
        assert list(hls_model.get_layers())[2].attributes['n_filt'] == class_object_conv.out_channels
        assert list(hls_model.get_layers())[2].attributes['stride_width'] == class_object_conv.stride[1]
        assert list(hls_model.get_layers())[2].attributes['stride_height'] == class_object_conv.stride[0]
        if list(hls_model.get_layers())[2].attributes['padding'] == 'valid':
            padding = 0
        else:
            padding = 1
        assert padding == class_object_conv.padding[0]
        assert list(hls_model.get_layers())[2].attributes['data_format'] == 'channels_last'

        pad_along_width = max(
            (out_width - 1) * class_object_conv.stride[1] + class_object_conv.kernel_size[1] - size_in_width, 0
        )
        pad_along_height = max(
            (out_height - 1) * class_object_conv.stride[0] + class_object_conv.kernel_size[0] - size_in_height, 0
        )

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        if padds == 1:
            assert list(hls_model.get_layers())[2].attributes['pad_left'] == pad_left
            assert list(hls_model.get_layers())[2].attributes['pad_right'] == pad_right
            assert list(hls_model.get_layers())[2].attributes['pad_top'] == pad_top
            assert list(hls_model.get_layers())[2].attributes['pad_bottom'] == pad_bottom
        elif padds == 0:
            assert list(hls_model.get_layers())[2].attributes['pad_left'] == 0
            assert list(hls_model.get_layers())[2].attributes['pad_right'] == 0
            assert list(hls_model.get_layers())[2].attributes['pad_top'] == 0
            assert list(hls_model.get_layers())[2].attributes['pad_bottom'] == 0


pooling_layers = [MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d]


@pytest.mark.parametrize('pooling', pooling_layers)
@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Quartus'])
def test_pooling(pooling, padds, backend):
    assert '1d' in pooling.__name__ or '2d' in pooling.__name__

    if '2d' in pooling.__name__:
        n_in = 3
        size_in_height = 15
        size_in_width = 18
    else:
        n_in = 3
        size_in_width = 121
        size_in_height = 0

    input_shape = (100, n_in, size_in_height, size_in_width) if '2d' in pooling.__name__ else (100, n_in, size_in_width)
    input_shape_forHLS = (
        (None, n_in, size_in_height, size_in_width) if '2d' in pooling.__name__ else (None, n_in, size_in_width)
    )
    X_input = np.random.rand(*input_shape)

    model = torch.nn.Sequential(pooling(3, padding=padds)).to()
    model.eval()
    pytorch_prediction = model(torch.Tensor(X_input)).detach().numpy()

    config = config_from_pytorch_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_pytorch_api_pooling_{pooling.__name__}_padds_{padds}_backend_{backend}')
    hls_model = convert_from_pytorch_model(
        model, input_shape_forHLS, hls_config=config, output_dir=output_dir, backend=backend
    )
    hls_model.compile()

    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)
    nNodes = 0
    poolNode = None
    for _node in traced_model.graph.nodes:
        nNodes += 1
        if nNodes == 2:
            poolNode = _node
    assert nNodes + 1 == len(hls_model.get_layers())
    children = {c[0]: c[1] for c in model.named_children()}
    class_object_pool = children[poolNode.target]

    if "Max" in pooling.__name__:
        out_height = int(
            (size_in_height + 2 * padds - class_object_pool.dilation * (class_object_pool.kernel_size - 1) - 1)
            / class_object_pool.stride
            + 1
        )
        out_width = int(
            (size_in_width + 2 * padds - class_object_pool.dilation * (class_object_pool.kernel_size - 1) - 1)
            / class_object_pool.stride
            + 1
        )
    else:
        if '2d' in pooling.__name__:
            out_height = int((size_in_height + 2 * padds - class_object_pool.kernel_size) / class_object_pool.stride + 1)
            out_width = int((size_in_width + 2 * padds - class_object_pool.kernel_size) / class_object_pool.stride + 1)
        else:
            out_height = int(
                (size_in_height + 2 * padds - class_object_pool.kernel_size[0]) / class_object_pool.stride[0] + 1
            )
            out_width = int((size_in_width + 2 * padds - class_object_pool.kernel_size[0]) / class_object_pool.stride[0] + 1)

    if '2d' in pooling.__name__:
        hls_prediction = np.reshape(hls_model.predict(X_input), (100, n_in, out_height, out_width))

    else:
        hls_prediction = np.reshape(hls_model.predict(X_input), (100, n_in, out_width))

    # results are not very good at the moment
    np.testing.assert_allclose(hls_prediction, pytorch_prediction, rtol=0.20, atol=0)

    # Verify correct parsing of layer
    hls_pool = list(hls_model.get_layers())[-2]
    if '2d' in pooling.__name__:
        assert hls_pool.attributes['name'] == 'layer' + poolNode.name
        assert hls_pool.attributes['class_name'][-2] == str(2)
        assert hls_pool.attributes['stride_height'] == class_object_pool.stride
        assert hls_pool.attributes['stride_width'] == class_object_pool.stride
        assert hls_pool.attributes['pool_height'] == class_object_pool.kernel_size
        assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size
        assert hls_pool.attributes['padding'] == 'valid' if class_object_pool.padding == 0 else 'same'

        if hls_pool.attributes['padding'] == 'same':
            # Height
            assert out_height == hls_pool.attributes['out_height']
            if size_in_height % class_object_pool.stride == 0:
                pad_along_height = max(class_object_pool.kernel_size - class_object_pool.stride, 0)
            else:
                pad_along_height = max(class_object_pool.kernel_size[1] - (size_in_height % class_object_pool.stride), 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            assert pad_bottom == hls_pool.attributes['pad_bottom']
            assert pad_top == hls_pool.attributes['pad_top']

            # Width
            assert out_width == hls_pool.attributes['out_width']
            if size_in_width % class_object_pool.stride == 0:
                pad_along_width = max(class_object_pool.kernel_size - class_object_pool.stride, 0)
            else:
                pad_along_width = max(class_object_pool.kernel_size - (size_in_width % class_object_pool.stride), 0)
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            assert pad_left == hls_pool.attributes['pad_left']
            assert pad_right == hls_pool.attributes['pad_right']

        elif hls_pool.attributes['padding'] == 'valid':

            assert hls_pool.attributes['out_height'] == out_height
            assert hls_pool.attributes['out_width'] == out_width
            assert hls_pool.attributes['pad_top'] == 0
            assert hls_pool.attributes['pad_bottom'] == 0
            assert hls_pool.attributes['pad_left'] == 0
            assert hls_pool.attributes['pad_right'] == 0

    elif '1d' in pooling.__name__:
        if "Max" in pooling.__name__:
            assert hls_pool.attributes['name'] == 'layer' + poolNode.name
            assert hls_pool.attributes['class_name'][-2] == str(1)
            assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size
            assert hls_pool.attributes['stride_width'] == class_object_pool.stride
            assert hls_pool.attributes['padding'] == 'valid' if class_object_pool.padding == 0 else 'same'

            if hls_pool.attributes['padding'] == 'same':
                assert hls_pool.attributes['n_out'] == out_width
                if size_in_width % class_object_pool.stride == 0:
                    pad_along_width = max(class_object_pool.kernel_size - class_object_pool.stride, 0)
                else:
                    pad_along_width = max(class_object_pool.kernel_size - (size_in_width % class_object_pool.stride), 0)
                assert hls_pool.attributes['pad_left'] == pad_along_width // 2
                assert hls_pool.attributes['pad_right'] == pad_along_width - pad_along_width // 2

            elif hls_pool.attributes['padding'] == 'valid':
                assert hls_pool.attributes['n_out'] == out_width
                assert hls_pool.attributes['pad_left'] == 0
                assert hls_pool.attributes['pad_right'] == 0
        else:
            assert hls_pool.attributes['name'] == 'layer' + poolNode.name
            assert hls_pool.attributes['class_name'][-2] == str(1)
            assert hls_pool.attributes['pool_width'] == class_object_pool.kernel_size[0]
            assert hls_pool.attributes['stride_width'] == class_object_pool.stride[0]
            assert hls_pool.attributes['padding'] == 'same' if class_object_pool.padding == 0 else 'valid'

            if hls_pool.attributes['padding'] == 'same':
                assert hls_pool.attributes['n_out'] == out_width
                if size_in_width % class_object_pool.stride[0] == 0:
                    pad_along_width = max(class_object_pool.kernel_size[0] - class_object_pool.stride[0], 0)
                else:
                    pad_along_width = max(
                        class_object_pool.kernel_size[0] - (size_in_width % class_object_pool.stride[0]), 0
                    )
                assert hls_pool.attributes['pad_left'] == pad_along_width // 2
                assert hls_pool.attributes['pad_right'] == pad_along_width - pad_along_width // 2

            elif hls_pool.attributes['padding'] == 'valid':
                assert hls_pool.attributes['n_out'] == out_width
                assert hls_pool.attributes['pad_left'] == 0
                assert hls_pool.attributes['pad_right'] == 0
