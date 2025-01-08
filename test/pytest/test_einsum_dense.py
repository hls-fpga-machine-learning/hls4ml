from pathlib import Path

import keras
import numpy as np
import pytest

from hls4ml.converters import convert_from_keras_model

if keras.__version__ < '3.0.0':
    pytest.skip('Only keras v3 is supported for now', allow_module_level=True)

from keras.api.layers import EinsumDense, Input

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('strategy', ['latency'])
@pytest.mark.parametrize('io_type', ['io_parallel'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize(
    'operation',
    [
        # eq, inp, out
        ('bi,j->bij', (8,), (8, 7), None),
        ('bi,j->bij', (8,), (8, 7), 'i'),
        ('bi,j->bij', (8,), (8, 7), 'j'),
        ('bi,io->bo', (8,), 7, None),
        ('...i,oi->...o', (4, 3), (5,), None),
        ('...abcd,bcde->...aeb', (5, 4, 3, 2), (5, 6, 4), None),
        ('...abcd,bcde->...aeb', (5, 4, 3, 2), (5, 6, 4), 'aeb'),
        ('...abcd,bcde->...aeb', (5, 4, 3, 2), (5, 6, 4), 'ab'),
        ('...abcd,bcde->...aeb', (5, 4, 3, 2), (5, 6, 4), 'a'),
    ],
)
def test_einsum_dense(backend, io_type, strategy, operation):
    eq, inp_shape, out_shape, bias_axes = operation
    model = keras.Sequential(
        [Input(inp_shape), EinsumDense(eq, output_shape=out_shape, bias_axes=bias_axes, name='einsum_dense')]
    )

    if bias_axes is not None:
        layer = model.get_layer('einsum_dense')
        layer.bias.assign(keras.ops.convert_to_tensor(np.random.rand(*layer.bias.shape)))

    data = np.random.rand(1000, *inp_shape)
    eq_name = eq.replace(',', '_').replace('->', '_') + ('' if bias_axes is None else f'_{bias_axes}')
    output_dir = str(test_root_path / f'hls4mlprj_einsum_dense_{eq_name}_{backend}_{io_type}_{strategy}')
    hls_config = {'Model': {'Precision': 'ap_fixed<32,8>', 'ReuseFactor': 1}, 'Strategy': strategy}
    model_hls = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )

    model_hls.compile()
    r_keras = model.predict(data, verbose=0, batch_size=1000)  # type: ignore
    r_hls = model_hls.predict(data).reshape(r_keras.shape)  # type: ignore

    np.testing.assert_allclose(r_hls, r_keras, atol=2e-6, rtol=0)
