import math
from pathlib import Path

import keras
import numpy as np
import pytest

if keras.__version__ < '3.0':
    pytest.skip('Keras API tests are only for Keras 3.0 and above', allow_module_level=True)

from keras.api.layers import (
    ELU,
    Activation,
    AveragePooling1D,
    AveragePooling2D,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv1D,
    DepthwiseConv2D,
    LeakyReLU,
    MaxPooling1D,
    MaxPooling2D,
    PReLU,
)

import hls4ml

test_root_path = Path('/tmp/tests')


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI', 'Catapult'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_dense(backend, io_type):
    model = keras.Sequential(
        [
            Dense(
                2,
                input_shape=(1,),
                name='Dense',
                use_bias=True,
                kernel_initializer=keras.initializers.RandomUniform(minval=1, maxval=10),  # type: ignore
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
            ),
            Activation(activation='elu', name='Activation'),
        ]
    )
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(1000, 1)

    keras_prediction = model.predict(X_input, verbose=0)  # type: ignore

    config = hls4ml.utils.config_from_keras_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_keras_api_dense_{backend}_{io_type}')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )

    hls_model.compile()

    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.02)

    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes["class_name"] == model.layers[0].name
    assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ELU'


# TODO: add ThresholdedReLU test when it can be made to pass
# https://github.com/fastmachinelearning/hls4ml/issues/376


@pytest.mark.parametrize(
    "activation_function",
    [
        Activation(activation='relu', name='relu'),
        LeakyReLU(negative_slope=0.5),
        ELU(alpha=1.0),
        PReLU(
            alpha_initializer="zeros",
        ),
        Activation(activation='sigmoid', name='sigmoid'),
    ],
)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_activations(activation_function, backend, io_type):
    model = keras.models.Sequential()
    model.add(Dense(64, input_shape=(1,), name='Dense', kernel_initializer='lecun_uniform', kernel_regularizer=None))
    model.add(activation_function)

    model.compile(optimizer='adam', loss='mse')

    model.summary()

    X_input = np.random.rand(1000, 1)
    keras_prediction = model.predict(X_input, verbose=0)  # type: ignore
    config = hls4ml.utils.config_from_keras_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_keras_api_activations_{activation_function.name}_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=0.02)

    for layer in hls_model.get_layers():
        print(layer.attributes.attributes['class_name'])
    assert len(model.layers) + 1 == len(hls_model.get_layers())

    assert list(hls_model.get_layers())[2].attributes['class_name'] == activation_function.__class__.__name__


padds_options = ['same', 'valid']


@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI', 'Catapult'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_conv1d(padds, backend, io_type):
    model = keras.models.Sequential()
    input_shape = (10, 128, 4)
    model.add(
        Conv1D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding=padds,
            activation='relu',
            input_shape=input_shape[1:],
            kernel_initializer='normal',
            use_bias=False,
            data_format='channels_last',
            name='conv',
        )
    )
    model.add(Activation(activation='relu'))
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(10, 128, 4)
    keras_prediction = model.predict(X_input, verbose=0)  # type: ignore

    config = hls4ml.utils.config_from_keras_model(model)
    output_dir = str(test_root_path / f'hls4mlprj_keras_api_conv1d_{padds}_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)  # type: ignore

    # 5e-2 might be too high
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=5e-2)

    if backend in ('Vivado', 'Vitis', 'Catapult') and io_type == 'io_stream' and padds == 'same':
        # Vivado/Vitis inserts and additional layer for 'same' padding in io_stream
        return

    conv: keras.layers.Conv1D = model.layers[0]
    ker_w, ch_in, ch_out = conv.kernel.shape
    inp_shape = model.inputs[0].shape[1:]
    out_shape = model.outputs[0].shape[1:]
    hls_attr = hls_model.graph['conv'].attributes
    _stride = conv.strides[0]

    assert len(model.layers) + 2 == len(hls_model.get_layers())

    assert hls_attr['name'] == model.layers[0].name
    assert hls_attr['class_name'] == 'Conv1D'
    assert hls_attr["in_width"] == inp_shape[0]
    assert hls_attr['filt_width'] == ker_w
    assert hls_attr['n_chan'] == ch_in
    assert hls_attr['n_filt'] == ch_out
    assert hls_attr['stride_width'] == _stride
    assert hls_attr['data_format'] == conv.data_format
    assert hls_attr["out_width"] == out_shape[0]

    w_pad = math.ceil(inp_shape[0] / ker_w) * ker_w - inp_shape[0]

    pad_left = w_pad // 2
    pad_right = w_pad - pad_left

    if model.layers[0].padding == 'same':
        assert hls_attr['pad_left'] == pad_left
        assert hls_attr['pad_right'] == pad_right
    elif model.layers[0].padding == 'valid':
        assert hls_attr['pad_left'] == 0
        assert hls_attr['pad_right'] == 0


chans_options = ['channels_last']
padds_options = ['same', 'valid']


@pytest.mark.parametrize('chans', chans_options)
@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI', 'Catapult'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_conv2d(chans, padds, backend, io_type):
    input_shape = (32, 32, 3)
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape),
            Conv2D(
                filters=32,
                kernel_size=(2, 3),
                strides=(4, 5),
                padding=padds,
                kernel_initializer='normal',
                use_bias=False,
                data_format=chans,
                name='conv',
            ),
        ]
    )
    model.compile(optimizer='adam', loss='mse')

    X_input = np.random.rand(1000, *input_shape)
    keras_prediction = model.predict(X_input)

    config = hls4ml.utils.config_from_keras_model(model)
    output_dir = str(test_root_path / f'hls4ml_project_keras_api_conv2d_{backend}_{chans}_{padds}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)  # type: ignore

    # A high tolerance, simply to verify correct functionality
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=5e-2)

    hls_conv_attr = hls_model.graph['conv'].attributes

    conv: keras.layers.Conv2D = model.get_layer('conv')

    kh, kw, ch_in, ch_out = conv.kernel.shape  # type: ignore
    _stride = conv.strides
    inp_shape = model.inputs[0].shape[1:]
    out_shape = model.outputs[0].shape[1:]

    if io_type == 'io_stream' and padds == 'same' and backend in ('Vivado', 'Vitis', 'Catapult'):
        return

    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert hls_conv_attr['name'] == conv.name
    assert hls_conv_attr['class_name'] == 'Conv2D'
    assert hls_conv_attr['filt_width'] == kw
    assert hls_conv_attr['filt_height'] == kh
    assert hls_conv_attr['n_filt'] == ch_out
    assert hls_conv_attr['stride_width'] == _stride[1]
    assert hls_conv_attr['stride_height'] == _stride[0]
    assert hls_conv_attr['data_format'] == conv.data_format

    if conv.data_format == 'channels_first':
        assert hls_conv_attr['n_chan'] == inp_shape[0]
        assert hls_conv_attr['in_height'] == inp_shape[1]
        assert hls_conv_attr['in_width'] == inp_shape[2]
        assert hls_conv_attr['out_height'] == out_shape[1]
        assert hls_conv_attr['out_width'] == out_shape[2]
    elif model.layers[0].data_format == 'channels_last':
        assert hls_conv_attr['n_chan'] == inp_shape[2]
        assert hls_conv_attr['in_height'] == inp_shape[0]
        assert hls_conv_attr['in_width'] == inp_shape[1]
        assert hls_conv_attr['out_height'] == out_shape[0]
        assert hls_conv_attr['out_width'] == out_shape[1]

    if conv.padding == 'same':
        if conv.data_format == 'channels_first':
            h_pad = math.ceil(inp_shape[1] / kh) * kh - inp_shape[1]
            w_pad = math.ceil(inp_shape[2] / kw) * kw - inp_shape[2]
        elif model.layers[0].data_format == 'channels_last':
            h_pad = math.ceil(inp_shape[0] / kh) * kh - inp_shape[0]
            w_pad = math.ceil(inp_shape[1] / kw) * kw - inp_shape[1]
        else:
            raise ValueError('Invalid data_format')
        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
        assert hls_conv_attr['pad_top'] == pad_top
        assert hls_conv_attr['pad_bottom'] == pad_bottom
        assert hls_conv_attr['pad_left'] == pad_left
        assert hls_conv_attr['pad_right'] == pad_right
    elif model.layers[0].padding == 'valid':
        assert hls_conv_attr['pad_top'] == 0
        assert hls_conv_attr['pad_bottom'] == 0
        assert hls_conv_attr['pad_left'] == 0
        assert hls_conv_attr['pad_right'] == 0


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Catapult'])
@pytest.mark.parametrize('io_type', ['io_stream', 'io_parallel'])
def test_depthwise2d(backend, io_type):
    '''
    Test proper handling of DepthwiseConv2D
    '''
    X = np.random.rand(10, 32, 32, 3)
    X = np.round(X * 2**10) * 2**-10  # make it an exact ap_fixed<16,6>
    model = keras.models.Sequential([keras.layers.Input((32, 32, 3)), DepthwiseConv2D(kernel_size=(3, 3))])
    model.compile()

    config = hls4ml.utils.config_from_keras_model(
        model, granularity='name', default_precision='fixed<32,12>', backend=backend
    )
    output_dir = str(test_root_path / f'hls4mlprj_keras_api_depthwiseconv2d_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    y_qkeras = model.predict(X)
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qkeras, y_hls4ml.reshape(y_qkeras.shape), rtol=1e-2, atol=0.01)  # type: ignore


# Currently only Vivado and Vitis is supported for io_stream.
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('io_type', ['io_stream'])
def test_depthwise1d(backend, io_type):
    '''
    Test proper handling of DepthwiseConv1D.
    '''
    X = np.random.rand(10, 32, 3)
    X = np.round(X * 2**10) * 2**-10  # make it an exact ap_fixed<16,6>
    model = keras.Sequential([DepthwiseConv1D(kernel_size=3, input_shape=(32, 3))])
    model.compile()

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    output_dir = str(test_root_path / f'hls4mlprj_keras_api_depthwiseconv1d_{backend}_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=output_dir, backend=backend, io_type=io_type
    )
    hls_model.compile()

    y_qkeras = model.predict(X)
    y_hls4ml = hls_model.predict(X)

    np.testing.assert_allclose(y_qkeras, y_hls4ml.reshape(y_qkeras.shape), rtol=1e-2, atol=0.01)  # type: ignore


pooling_layers = [MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D]


@pytest.mark.parametrize('pooling', pooling_layers)
@pytest.mark.parametrize('padds', padds_options)
@pytest.mark.parametrize('chans', chans_options)
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'oneAPI', 'Catapult'])
def test_pooling(pooling, padds, chans, backend):
    assert '1D' in pooling.__name__ or '2D' in pooling.__name__

    input_shape = (18, 15, 3) if '2D' in pooling.__name__ else (121, 3)
    pool_size = (4, 2) if '2D' in pooling.__name__ else 2

    X_input = np.random.rand(100, *input_shape)

    keras_model = keras.Sequential([pooling(pool_size, padding=padds, input_shape=input_shape)])
    keras_model.compile()

    hls_cfg = hls4ml.utils.config_from_keras_model(keras_model)
    output_dir = str(
        test_root_path / f'hls4mlprj_keras_api_pooling_{pooling.__name__}_channels_{chans}_padds_{padds}_backend_{backend}'
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model, hls_config=hls_cfg, output_dir=output_dir, backend=backend
    )
    hls_model.compile()

    # Verify accuracy
    keras_prediction = keras_model.predict(X_input)
    hls_prediction = hls_model.predict(X_input).reshape(keras_prediction.shape)  # type: ignore
    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=0, atol=3e-2)

    # # Verify correct parsing of layer
    # hls_pool = list(hls_model.get_layers())[-1]
    # ker_pool = keras_model.layers[-1]
    # if '2D' in pooling.__name__:
    #     assert hls_pool.attributes['name'] == ker_pool._name
    #     assert hls_pool.attributes['class_name'][-2] == str(2)
    #     assert hls_pool.attributes['stride_height'] == ker_pool.strides[0]
    #     assert hls_pool.attributes['stride_width'] == ker_pool.strides[1]
    #     assert hls_pool.attributes['pool_height'] == ker_pool.pool_size[1]
    #     assert hls_pool.attributes['pool_width'] == ker_pool.pool_size[0]

    #     if hls_pool.attributes['data_format'] == 'channels_last':
    #         assert hls_pool.attributes['in_height'] == ker_pool.input_shape[1]
    #         assert hls_pool.attributes['in_width'] == ker_pool.input_shape[2]
    #         assert hls_pool.attributes['n_filt'] == ker_pool.input_shape[3]
    #     elif hls_pool.attributes['data_format'] == 'channels_first':
    #         assert hls_pool.attributes['in_height'] == ker_pool.input_shape[2]
    #         assert hls_pool.attributes['in_width'] == ker_pool.input_shape[3]
    #         assert hls_pool.attributes['n_filt'] == ker_pool.input_shape[1]

    #     if ker_pool.padding == 'same':
    #         # Height
    #         in_height = ker_pool.input_shape[1]
    #         if ker_pool.data_format == 'channels_first':
    #             in_height = ker_pool.input_shape[2]
    #         out_height = int(math.ceil(float(in_height) / float(ker_pool.strides[0])))
    #         assert out_height == hls_pool.attributes['out_height']
    #         if in_height % ker_pool.strides[0] == 0:
    #             pad_along_height = max(ker_pool.pool_size[1] - ker_pool.strides[0], 0)
    #         else:
    #             pad_along_height = max(ker_pool.pool_size[1] - (in_height % ker_pool.strides[0]), 0)
    #         pad_top = pad_along_height // 2
    #         pad_bottom = pad_along_height - pad_top
    #         assert pad_bottom == hls_pool.attributes['pad_bottom']
    #         assert pad_top == hls_pool.attributes['pad_top']

    #         # Width
    #         in_width = ker_pool.input_shape[2]
    #         if ker_pool.data_format == 'channels_first':
    #             in_height = keras_model.layers[1].input_shape[-1]
    #         out_width = int(math.ceil(float(in_width) / float(ker_pool.strides[1])))
    #         assert out_width == hls_pool.attributes['out_width']
    #         if in_width % ker_pool.strides[1] == 0:
    #             pad_along_width = max(ker_pool.pool_size[0] - ker_pool.strides[1], 0)
    #         else:
    #             pad_along_width = max(ker_pool.pool_size[0] - (in_width % ker_pool.strides[1]), 0)
    #         pad_left = pad_along_width // 2
    #         pad_right = pad_along_width - pad_left
    #         assert pad_left == hls_pool.attributes['pad_left']
    #         assert pad_right == hls_pool.attributes['pad_right']

    #     elif ker_pool.padding == 'valid':
    #         if hls_pool.attributes['data_format'] == 'channels_first':
    #             in_height = ker_pool.input_shape[2]
    #             in_width = ker_pool.input_shape[3]
    #         elif hls_pool.attributes['data_format'] == 'channels_last':
    #             in_height = ker_pool.input_shape[1]
    #             in_width = ker_pool.input_shape[2]
    #         else:
    #             raise ValueError('Invalid data_format')

    #         out_width = int(math.ceil(float(in_width - ker_pool.pool_size[0] + 1) / float(ker_pool.strides[1])))
    #         out_height = int(math.ceil(float(in_height - ker_pool.pool_size[1] + 1) / float(ker_pool.strides[0])))

    #         assert hls_pool.attributes['out_height'] == out_height
    #         assert hls_pool.attributes['out_width'] == out_width
    #         assert hls_pool.attributes['pad_top'] == 0
    #         assert hls_pool.attributes['pad_bottom'] == 0
    #         assert hls_pool.attributes['pad_left'] == 0
    #         assert hls_pool.attributes['pad_right'] == 0

    # elif '1D' in pooling.__name__:
    #     assert hls_pool.attributes['name'] == ker_pool._name
    #     assert hls_pool.attributes['class_name'][-2] == str(1)
    #     assert hls_pool.attributes['n_in'] == ker_pool.input_shape[1]
    #     assert hls_pool.attributes['n_filt'] == ker_pool.input_shape[2]
    #     assert hls_pool.attributes['pool_width'] == ker_pool.pool_size[0]
    #     assert hls_pool.attributes['stride_width'] == ker_pool.strides[0]

    #     out_same = math.ceil(float(ker_pool.input_shape[1]) / float(ker_pool.strides[0]))
    #     out_valid = math.ceil(float(ker_pool.input_shape[1] - ker_pool.pool_size[0] + 1) / ker_pool.strides[0])

    #     if ker_pool.padding == 'same':
    #         assert hls_pool.attributes['n_out'] == out_same
    #         if ker_pool.input_shape[1] % ker_pool.strides[0] == 0:
    #             pad_along_width = max(ker_pool.pool_size[0] - ker_pool.strides[0], 0)
    #         else:
    #             pad_along_width = max(ker_pool.pool_size[0] - (ker_pool.input_shape[1] % ker_pool.strides[0]), 0)
    #         assert hls_pool.attributes['pad_left'] == pad_along_width // 2
    #         assert hls_pool.attributes['pad_right'] == pad_along_width - pad_along_width // 2

    #     elif ker_pool.padding == 'valid':
    #         assert hls_pool.attributes['n_out'] == out_valid
    #         assert hls_pool.attributes['pad_left'] == 0
    #         assert hls_pool.attributes['pad_right'] == 0


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Quartus', 'Catapult', 'oneAPI'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_reused_layer(backend, io_type):

    inp1 = keras.layers.Input(shape=(10, 10))
    inp2 = keras.layers.Input(shape=(10, 10))

    conv = keras.layers.Conv1D(2, 3, activation='relu')

    o1 = conv(inp1)
    o2 = conv(inp2)
    o3 = keras.layers.Add()([o1, o2])
    o4 = keras.layers.Dense(5)(o3)

    _ = keras.layers.Dense(5)(o3)

    model = keras.models.Model(inputs=[inp1, inp2], outputs=[o1, o2, o3, o4])

    _ = model([inp1, inp1])

    hls_config = {'Model': {'Precision': 'ap_fixed<32,8>', 'ReuseFactor': 1}}
    output_dir = str(test_root_path / f'hls4mlprj_keras_api_conv1d_{backend}_{io_type}')

    model_hls = hls4ml.converters.convert_from_keras_model(
        model, backend=backend, io_type=io_type, hls_config=hls_config, output_dir=output_dir
    )

    model_hls.compile()

    data = [np.random.rand(1000, 10, 10).astype(np.float32), np.random.rand(1000, 10, 10).astype(np.float32)]
    keras_pred = model.predict(data)
    hls_pred = model_hls.predict(data)

    np.testing.assert_allclose(keras_pred[0].reshape(hls_pred[0].shape), hls_pred[0], rtol=0, atol=1e-5)
    np.testing.assert_allclose(keras_pred[1].reshape(hls_pred[1].shape), hls_pred[1], rtol=0, atol=1e-5)
    np.testing.assert_allclose(keras_pred[2].reshape(hls_pred[2].shape), hls_pred[2], rtol=0, atol=1e-5)
    np.testing.assert_allclose(keras_pred[3].reshape(hls_pred[3].shape), hls_pred[3], rtol=0, atol=1e-2)
