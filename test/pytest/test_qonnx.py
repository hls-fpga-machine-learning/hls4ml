import os
import urllib
from pathlib import Path

import numpy as np
import pytest
import qonnx.core.onnx_exec as oxe
import qonnx.util.cleanup
import qonnx.util.to_channels_last

# To conveniently run QONNX inference
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.gemm_to_matmul import GemmToMatMul

import hls4ml

test_root_path = Path(__file__).parent
example_model_path = (test_root_path / '../../example-models').resolve()

# The models


@pytest.fixture(scope='module')
def tfc_2w2a_model():
    '''
    Load the tiny fully-connected model
    '''
    dl_dir = test_root_path
    dl_file = str(dl_dir / "qonnx-tfc-2w2a.onnx")
    tfc_w2a2_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning/"
        "QONNX_model_zoo/main/models/MNIST/Brevitas_FINN_TFC/TFC/TFC_2W2A.onnx"
    )
    urllib.request.urlretrieve(tfc_w2a2_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    out_file = str(dl_dir / "qonnx-tfc-2w2a-clean.onnx")

    # cleanup
    qonnx.util.cleanup.cleanup(dl_file, out_file=out_file)
    model = ModelWrapper(out_file)
    return model


@pytest.fixture(scope='module')
def cnv_2w2a_model():
    '''
    Load the small convolution model
    '''
    dl_dir = test_root_path
    dl_file = str(dl_dir / "qonnx-cnv-2w2a.onnx")
    cnv_w2a2_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning/"
        "QONNX_model_zoo/main/models/CIFAR10/Brevitas_FINN_CNV/CNV_2W2A.onnx"
    )
    urllib.request.urlretrieve(cnv_w2a2_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    out_clean = str(dl_dir / "qonnx-cnv-2w2a-clean.onnx")
    out_chanlast = str(dl_dir / "qonnx-cnv-2w2a-clean-channels-last.onnx")
    out_file = str(dl_dir / "qonnx-cnv-2w2a-clean-channels-last-clean.onnx")

    # cleanup
    qonnx.util.cleanup.cleanup(dl_file, out_file=out_clean)
    qonnx.util.to_channels_last.to_channels_last(out_clean, make_input_channels_last=True, out_file=out_chanlast)
    qonnx.util.cleanup.cleanup(out_chanlast, out_file=out_file)
    model = ModelWrapper(out_file)
    return model


@pytest.fixture(scope='module')
def jettagging_model():
    '''
    Load the 3 hidden layer QKeras example model trained on the jet tagging dataset
    '''
    dl_dir = test_root_path
    dl_file = str(dl_dir / "qkeras_jettagging.onnx")
    jet_tagging_qonnx_url = (
        "https://raw.githubusercontent.com/fastmachinelearning/"
        "QONNX_model_zoo/main/models/JetTagging/QKeras_hls4ml_3layer/qkeras_jettagging.onnx"
    )
    urllib.request.urlretrieve(jet_tagging_qonnx_url, dl_file)
    assert os.path.isfile(dl_file)
    out_file = str(dl_dir / "qkeras_jettagging-clean.onnx")

    # cleanup
    qonnx.util.cleanup.cleanup(dl_file, out_file=out_file)
    model = ModelWrapper(out_file)
    return model


@pytest.fixture(scope='module')
def sep_conv_model():
    """
    Load separabale conv model, already channels-last and cleaned
    """
    dl_file = str(example_model_path / "onnx/separable_conv_model_ch_last.onnx")
    assert os.path.isfile(dl_file)

    model = ModelWrapper(dl_file)

    return model


@pytest.fixture(scope='module')
def branched_model():
    """
    Load branched model using separable convs, already channels-last and cleaned
    """
    dl_file = str(example_model_path / "onnx/branched_model_ch_last.onnx")
    assert os.path.isfile(dl_file)

    model = ModelWrapper(dl_file)

    return model


@pytest.fixture(scope='module')
def tiny_unet_model():
    """
    Load tiny unet model, already channels-last and cleaned
    """
    dl_file = str(example_model_path / "onnx/tiny_unet_ch_last.onnx")
    assert os.path.isfile(dl_file)

    model = ModelWrapper(dl_file)

    return model


@pytest.fixture(scope='module')
def two_layer_keras_model():
    """
    Load a simple, two-layer, originally keras, unquantized model
    """
    dl_file = str(example_model_path / "onnx/two_layer_keras.onnx")
    assert os.path.isfile(dl_file)

    model = ModelWrapper(dl_file)
    model = qonnx.util.cleanup.cleanup_model(model)
    return model


@pytest.fixture(scope='module')
def three_layer_keras_model():
    """
    Load a simple, three-layer, originally keras, unquantized model
    """
    dl_file = str(example_model_path / "onnx/three_layer_keras.onnx")
    assert os.path.isfile(dl_file)

    model = ModelWrapper(dl_file)
    model = qonnx.util.cleanup.cleanup_model(model)
    return model


@pytest.fixture(scope='module')
def two_layer_pytorch_model():
    """
    Load a simple, two-layer, originally pytorch, unquantized model
    """
    dl_file = str(example_model_path / "onnx/two_layer_keras.onnx")
    assert os.path.isfile(dl_file)

    model = ModelWrapper(dl_file)
    model = qonnx.util.cleanup.cleanup_model(model)
    model = model.transform(GemmToMatMul())
    model = qonnx.util.cleanup.cleanup_model(model)
    return model


@pytest.fixture(scope='module')
def three_layer_pytorch_model():
    """
    Load a simple, three-layer, originally pytorch, unquantized model
    """
    dl_file = str(example_model_path / "onnx/three_layer_pytorch.onnx")
    assert os.path.isfile(dl_file)

    model = ModelWrapper(dl_file)
    model = qonnx.util.cleanup.cleanup_model(model)
    model = model.transform(GemmToMatMul())
    model = qonnx.util.cleanup.cleanup_model(model)
    return model


@pytest.fixture(scope='module')
def conv1d_small_keras_model():
    """
    Load a simple conv1d, originally keras, unquantized model
    """
    dl_file = str(example_model_path / "onnx/conv1d_small_keras.onnx")
    assert os.path.isfile(dl_file)

    model = ModelWrapper(dl_file)
    model = qonnx.util.cleanup.cleanup_model(model)
    model = model.transform(ConvertToChannelsLastAndClean())
    model = model.transform(GemmToMatMul())
    model = qonnx.util.cleanup.cleanup_model(model)
    return model


@pytest.fixture(scope='module')
def conv2d_small_keras_model():
    """
    Load a simple conv2d, originally keras, unquantized model
    """
    dl_file = str(example_model_path / "onnx/conv2d_small_keras.onnx")
    assert os.path.isfile(dl_file)

    model = ModelWrapper(dl_file)
    model = qonnx.util.cleanup.cleanup_model(model)
    model = model.transform(ConvertToChannelsLastAndClean())
    model = model.transform(GemmToMatMul())
    model = qonnx.util.cleanup.cleanup_model(model)
    return model


@pytest.fixture(scope='module')
def conv2d_small_mp_keras_model():
    """
    Load a conv2d model with max pooling, originally keras, unquantized model
    """
    dl_file = str(example_model_path / "onnx/conv2d_small_mp_keras.onnx")
    assert os.path.isfile(dl_file)

    model = ModelWrapper(dl_file)
    model = qonnx.util.cleanup.cleanup_model(model)
    model = model.transform(ConvertToChannelsLastAndClean())
    model = model.transform(GemmToMatMul())
    model = qonnx.util.cleanup.cleanup_model(model)
    return model


# The actual tests


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_tfc_2w2a(tfc_2w2a_model, backend):
    model = tfc_2w2a_model

    ishape = (1, 1, 28, 28)
    X = np.random.uniform(low=-1, high=+1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**16) * 2**-16).astype(np.float32)

    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    # Convert QONNX model, compile, and run inference
    config = hls4ml.utils.config_from_onnx_model(model, backend=backend, default_precision='fixed<32,16>')
    hls_model = hls4ml.converters.convert_from_onnx_model(
        model, output_dir=str(test_root_path / f'hls4mlprj_qonnx_tfc-2w2a_{backend}'), backend=backend, hls_config=config
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)


@pytest.mark.parametrize('backend', ['Vitis'])
def test_cnv_2w2a(cnv_2w2a_model, backend):
    """
    This tests a convolution model. Note:  the batch normalizations weights not quantized, so it is
    difficult to make this match perfectly. It is also a slow test, which is why only Vitis is tested.
    """
    model = cnv_2w2a_model

    ishape = (1, 32, 32, 3)
    X = np.random.uniform(low=-1, high=+1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**6) * 2**-6).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    # Convert QONNX model, compile, and run inference
    config = hls4ml.utils.config_from_onnx_model(model, backend=backend, default_precision='fixed<32,6>')
    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=str(test_root_path / f'hls4mlprj_qonnx_cnv-2w2a_{backend}'),
        io_type='io_stream',
        backend=backend,
        hls_config=config,
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus'])
def test_jet_tagging(jettagging_model, backend):
    model = jettagging_model

    # Execute QONNX model inference
    # TODO make the test bigger
    ishape = (1, 16)
    X = np.random.uniform(low=-1, high=+1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**16) * 2**-16).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    # Convert QONNX model, compile, and run inference
    config = hls4ml.utils.config_from_onnx_model(model, backend=backend, default_precision='fixed<32,16>')

    hls_model = hls4ml.converters.convert_from_onnx_model(
        model, output_dir=str(test_root_path / f'hls4mlprj_qonnx_jettag_{backend}'), backend=backend, hls_config=config
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)


@pytest.mark.parametrize('backend', ['Vitis'])
def test_sep_conv(sep_conv_model, backend):
    model = sep_conv_model
    ishape = tuple(model.get_tensor_shape(model.graph.input[0].name))
    X = np.random.uniform(low=0, high=1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**16) * 2**-16).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    config = hls4ml.utils.config.config_from_onnx_model(
        model, granularity='name', backend=backend, default_precision='fixed<32,16>'
    )

    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=str(test_root_path / f'hls4mlprj_qonnx_sep_conv_{backend}'),
        io_type='io_stream',
        backend=backend,
        hls_config=config,
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(np.ascontiguousarray(X))

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)


@pytest.mark.parametrize('backend', ['Vitis'])
def test_branched_model(branched_model, backend):
    model = branched_model
    ishape = tuple(model.get_tensor_shape(model.graph.input[0].name))
    X = np.random.uniform(low=0, high=1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**16) * 2**-16).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    config = hls4ml.utils.config.config_from_onnx_model(
        model, granularity='name', backend=backend, default_precision='fixed<32,16>'
    )
    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=str(test_root_path / f'hls4mlprj_qonnx_branched_model_{backend}'),
        io_type='io_stream',
        backend=backend,
        hls_config=config,
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(np.ascontiguousarray(X))

    np.testing.assert_array_equal(y_qonnx.ravel(), y_hls4ml.ravel())


@pytest.mark.parametrize('backend', ['Vitis'])
def test_tiny_unet_model(tiny_unet_model, backend):

    model = tiny_unet_model
    ishape = tuple(model.get_tensor_shape(model.graph.input[0].name))
    X = np.random.uniform(low=0, high=1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**16) * 2**-16).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    config = hls4ml.utils.config.config_from_onnx_model(
        model, granularity='name', backend=backend, default_precision='fixed<32,16>'
    )

    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=str(test_root_path / f'hls4mlprj_qonnx_tiny_unet_model_{backend}'),
        io_type='io_stream',
        backend=backend,
        hls_config=config,
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(np.ascontiguousarray(X))

    np.testing.assert_array_equal(y_qonnx.ravel(), y_hls4ml.ravel())


@pytest.mark.parametrize(
    'model_name',
    [
        'two_layer_keras_model',
        'three_layer_keras_model',
        'two_layer_pytorch_model',
        'three_layer_pytorch_model',
        'conv1d_small_keras_model',
        'conv2d_small_keras_model',
        'conv2d_small_mp_keras_model',
    ],
)
@pytest.mark.parametrize('backend', ['Vitis'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_simple_model(model_name, io_type, backend, request):
    model = request.getfixturevalue(model_name)
    ishape = tuple(model.get_tensor_shape(model.graph.input[0].name))
    X = np.random.uniform(low=0, high=1, size=np.prod(ishape)).reshape(ishape)
    X = (np.round(X * 2**10) * 2**-10).astype(np.float32)
    idict = {model.graph.input[0].name: X}
    y_qonnx = oxe.execute_onnx(model, idict)[model.graph.output[0].name]

    config = hls4ml.utils.config.config_from_onnx_model(
        model, granularity='name', backend=backend, default_precision='fixed<16,6>'
    )

    for layer in config['LayerName']:
        if layer.startswith('Softmax'):
            config['LayerName'][layer]['Implementation'] = 'legacy'

    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=str(test_root_path / f'hls4mlprj_onnx_{model_name}_{io_type}_{backend}'),
        io_type=io_type,
        backend=backend,
        hls_config=config,
    )
    hls_model.compile()
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qonnx.ravel(), y_hls4ml.ravel(), atol=1e-2, rtol=1)
