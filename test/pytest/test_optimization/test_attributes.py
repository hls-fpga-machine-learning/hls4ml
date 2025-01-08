from tensorflow.keras.layers import Conv2D, Dense, Flatten, ReLU
from tensorflow.keras.models import Sequential

from hls4ml.optimization.dsp_aware_pruning import get_attributes_from_keras_model_and_hls4ml_config
from hls4ml.utils.config import config_from_keras_model


def test_attributes():
    dense_units = 16
    conv_filters = 6
    conv_channels = 3
    conv_shape = (3, 3)
    input_shape = (8, 8)
    io_type = 'io_parallel'
    strategy = 'Resource'

    model = Sequential()
    model.add(
        Conv2D(
            conv_filters,
            input_shape=(*input_shape, conv_channels),
            kernel_size=conv_shape,
            name='conv2d',
            padding='same',
            kernel_initializer='ones',
        )
    )
    model.add(Flatten(name='flatten'))
    model.add(Dense(dense_units, name='dense', kernel_initializer='ones'))
    model.add(ReLU(name='relu'))

    default_reuse_factor = 2
    default_precision = 'ap_fixed<8, 0>'
    cfg = config_from_keras_model(
        model, granularity='name', default_precision=default_precision, default_reuse_factor=default_reuse_factor
    )
    cfg['IOType'] = io_type
    cfg['Model']['Strategy'] = strategy
    cfg['LayerName']['dense']['ReuseFactor'] = 1

    # optimization doesn't yet support auto precision
    for layer in cfg['LayerName'].values():
        for key, prec in layer['Precision'].items():
            if prec == 'auto':
                layer['Precision'][key] = default_precision

    # Verify correct information for every layer
    model_attributes = get_attributes_from_keras_model_and_hls4ml_config(model, cfg)
    assert len(model_attributes) == 4

    # conv2d
    assert model_attributes['conv2d'].name == 'conv2d'
    assert model_attributes['conv2d'].layer_type.__name__ == 'Conv2D'
    assert model_attributes['conv2d'].inbound_layers == []
    assert model_attributes['conv2d'].weight_shape == (3, 3, 3, 6)
    assert model_attributes['conv2d'].input_shape == (8, 8, 3)
    assert model_attributes['conv2d'].output_shape == (8, 8, 6)
    assert not model_attributes['conv2d'].optimizable
    assert model_attributes['conv2d'].args['hls4ml_attributes'].n_in == 9
    assert model_attributes['conv2d'].args['hls4ml_attributes'].n_out == 6
    assert model_attributes['conv2d'].args['hls4ml_attributes'].io_type == io_type
    assert model_attributes['conv2d'].args['hls4ml_attributes'].strategy == strategy
    assert model_attributes['conv2d'].args['hls4ml_attributes'].reuse_factor == default_reuse_factor
    assert model_attributes['conv2d'].args['hls4ml_attributes'].weight_precision.width == 8
    assert model_attributes['conv2d'].args['hls4ml_attributes'].parallelization_factor == 1

    # flatten
    assert model_attributes['flatten'].name == 'flatten'
    assert model_attributes['flatten'].layer_type.__name__ == 'Flatten'
    assert model_attributes['flatten'].weight_shape == ()
    assert model_attributes['flatten'].input_shape == (8, 8, 6)
    assert model_attributes['flatten'].output_shape == (384,)
    assert not model_attributes['flatten'].optimizable

    # Flatten is not optimizable so hls4mlAttributes (n_in, n_out, reuse factor etc.) will not be stored for it
    assert 'hls4ml_attributes' not in model_attributes['flatten'].args

    # dense
    assert model_attributes['dense'].name == 'dense'
    assert model_attributes['dense'].layer_type.__name__ == 'Dense'
    assert model_attributes['dense'].weight_shape == (384, 16)
    assert model_attributes['dense'].input_shape == (384,)
    assert model_attributes['dense'].output_shape == (16,)
    assert not model_attributes['dense'].optimizable
    assert model_attributes['dense'].args['hls4ml_attributes'].n_in == 384
    assert model_attributes['dense'].args['hls4ml_attributes'].n_out == 16
    assert model_attributes['dense'].args['hls4ml_attributes'].io_type == io_type
    assert model_attributes['dense'].args['hls4ml_attributes'].strategy == strategy
    assert model_attributes['dense'].args['hls4ml_attributes'].reuse_factor == 1
    assert model_attributes['dense'].args['hls4ml_attributes'].output_precision.width == 8
    assert model_attributes['dense'].args['hls4ml_attributes'].parallelization_factor == 1

    # relu
    assert model_attributes['relu'].name == 'relu'
    assert model_attributes['relu'].layer_type.__name__ == 'ReLU'
    assert model_attributes['relu'].weight_shape == ()
    assert model_attributes['relu'].input_shape == (16,)
    assert model_attributes['relu'].output_shape == (16,)
    assert not model_attributes['relu'].optimizable

    # ReLU is not optimizable so hls4mlAttributes (n_in, n_out, reuse factor etc.) will not be stored for it
    assert 'hls4ml_attributes' not in model_attributes['relu'].args
