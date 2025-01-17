from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import hls4ml
from hls4ml.contrib.garnet import GarNet, GarNetStack

test_root_path = Path(__file__).parent

vmax = 16
feat = 3


@pytest.fixture(scope='module')
def garnet_models():
    x = Input(shape=(vmax, feat))
    n = Input(shape=(1,), dtype='uint16')
    inputs = [x, n]
    outputs = GarNet(
        8,
        8,
        16,
        simplified=True,
        collapse='mean',
        input_format='xn',
        output_activation=None,
        name='gar_1',
        quantize_transforms=False,
    )(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vivado')
    config['Model'] = {}
    config['Model']['ReuseFactor'] = 1
    config['Model']['Strategy'] = 'Latency'
    config['Model']['Precision'] = 'ap_fixed<32,6>'
    config['LayerName']['gar_1']['Precision'] = {'default': 'ap_fixed<32, 6, AP_RND, AP_SAT>', 'result': 'ap_fixed<32, 6>'}

    cfg = hls4ml.converters.create_config(output_dir=str(test_root_path / 'hls4mlprj_garnet'), part='xc7z020clg400-1')
    cfg['HLSConfig'] = config
    cfg['KerasModel'] = model

    hls_model = hls4ml.converters.keras_to_hls(cfg)
    hls_model.compile()
    return model, hls_model


@pytest.fixture(scope='module')
def garnet_stack_models():
    x = Input(shape=(vmax, feat))
    n = Input(shape=(1,), dtype='uint16')
    inputs = [x, n]
    outputs = GarNetStack(
        ([4, 4, 8]),
        ([4, 4, 8]),
        ([8, 8, 16]),
        simplified=True,
        collapse='mean',
        input_format='xn',
        output_activation=None,  # added output_activation_transform back in contrib.garnet.py
        name='gar_1',
        quantize_transforms=None,  # this should be false, not None...fix in contrib.garnet.py
    )(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vivado')
    config['Model'] = {}
    config['Model']['ReuseFactor'] = 1
    config['Model']['Strategy'] = 'Latency'
    config['Model']['Precision'] = 'ap_fixed<32,6>'
    # config should now have precisions specified for ['LayerName']['gar_1']['Precision']['norm', 'aggr', etc.]
    cfg = hls4ml.converters.create_config(output_dir=str(test_root_path / 'hls4mlprj_garnet'), part='xc7z020clg400-1')
    cfg['HLSConfig'] = config
    cfg['KerasModel'] = model

    hls_model = hls4ml.converters.keras_to_hls(cfg)
    hls_model.compile()
    return model, hls_model


@pytest.mark.parametrize('batch', [1, 3])
def test_accuracy(garnet_models, batch):
    model, hls_model = garnet_models
    x = [np.random.rand(batch, vmax, feat), np.random.randint(0, vmax, size=(batch, 1))]
    y = model.predict(x)
    x_hls = [x[0], x[1].astype(np.float64)]
    y_hls = hls_model.predict(x_hls).reshape(y.shape)

    np.testing.assert_allclose(y_hls, y, rtol=0, atol=0.1)


@pytest.mark.parametrize('batch', [1, 3])
def test_accuracy_stack(garnet_stack_models, batch):
    model, hls_model = garnet_stack_models
    x = [np.random.rand(batch, vmax, feat), np.random.randint(0, vmax, size=(batch, 1))]
    y = model.predict(x)
    x_hls = [x[0], x[1].astype(np.float64)]
    y_hls = hls_model.predict(x_hls).reshape(y.shape)

    np.testing.assert_allclose(y_hls, y, rtol=0, atol=0.1)
