from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from keras.api import Layer

from hls4ml.converters.keras_v3._base import KerasV3LayerHandler, register
from hls4ml.converters.keras_v3.conv import KV3ConvHandler
from hls4ml.converters.keras_v3.core import KV3ActivationHandler, KV3DenseHandler
from hls4ml.converters.keras_v3.einsum_dense import KV3EinsumDenseHandler

if TYPE_CHECKING:
    import squark
    from keras import KerasTensor


def extract_fixed_quantizer_config(q, tensor: 'KerasTensor', is_input: bool) -> dict[str, Any]:
    from keras.api.ops import convert_to_numpy
    from squark.quantizer.internal.fixed_point_quantizer import FixedPointQuantizerKBI, FixedPointQuantizerKIF

    internal_q: FixedPointQuantizerKIF | FixedPointQuantizerKBI = q.quantizer

    shape: tuple[int, ...] = tensor.shape[1:]  # type: ignore
    if any([s is None for s in shape]):
        raise ValueError(f"Tensor {tensor.name} has at least one dimension with no fixed size")
    k, i, f = internal_q.kif
    k, B, I = k, k + i + f, k + i  # type: ignore
    k, B, I = convert_to_numpy(k), convert_to_numpy(B), convert_to_numpy(I)

    k = np.broadcast_to(k.astype(np.int8), (1,) + shape)
    B = np.broadcast_to(B.astype(np.int8), (1,) + shape)
    I = np.broadcast_to(I.astype(np.int8), (1,) + shape)

    overflow_mode = internal_q.overflow_mode
    round_mode = internal_q.round_mode
    fusible = np.unique(k).size == 1 and np.unique(B).size == 1 and np.unique(I).size == 1

    input_keras_tensor_names = tensor.name if is_input else f'{tensor.name}_q'
    output_keras_tensor_names = f'{tensor.name}_q' if is_input else tensor.name
    return {
        'name': q.name,
        'class_name': 'FixedPointQuantizer',
        'mask_kbi': (k, B, I),
        'SAT': overflow_mode,
        'RND': round_mode,
        'fusible': fusible,
        'input_keras_tensor_names': [input_keras_tensor_names],
        'output_keras_tensor_names': [output_keras_tensor_names],
        'overrides': {},
    }


def override_io_tensor_confs(confs: tuple[dict[str, Any], ...], overrides: dict[str, str]):
    for conf in confs:
        inp_tensor_names = conf['input_keras_tensor_names']
        out_tensor_names = conf['output_keras_tensor_names']
        conf['input_keras_tensor_names'] = [overrides.get(name, name) for name in inp_tensor_names]
        conf['output_keras_tensor_names'] = [overrides.get(name, name) for name in out_tensor_names]


@register
class SQLayerHandler(KerasV3LayerHandler):
    def __call__(
        self,
        layer: 'squark.layers.QLayerBase',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        ret = super().__call__(layer, in_tensors, out_tensors)

        if layer._enable_iq:
            if len(in_tensors) > 1:
                iq_confs = [extract_fixed_quantizer_config(q, tensor, True) for q, tensor in zip(layer._iqs, in_tensors)]
            else:
                iq_confs = [extract_fixed_quantizer_config(layer._iq, in_tensors[0], True)]
        else:
            iq_confs = ()

        if layer._enable_oq:
            if len(out_tensors) > 1:
                oq_confs = [extract_fixed_quantizer_config(q, tensor, False) for q, tensor in zip(layer._oqs, out_tensors)]
            else:
                oq_confs = [extract_fixed_quantizer_config(layer._oq, out_tensors[0], False)]
        else:
            oq_confs = ()

        if iq_confs:
            _froms = [t.name for t in in_tensors]
            _tos = [f'{t.name}_q' for t in in_tensors]
            overrides = dict(zip(_froms, _tos))
            override_io_tensor_confs(ret, overrides)

        if oq_confs:
            _froms = [t.name for t in out_tensors]
            _tos = [f'{t.name}_q' for t in out_tensors]
            overrides = dict(zip(_froms, _tos))
            override_io_tensor_confs(ret, overrides)

        return *iq_confs, *ret, *oq_confs

    def load_weight(self, layer: Layer, key: str):
        from keras.api.ops import convert_to_numpy

        if hasattr(layer, f'q{key}'):
            return convert_to_numpy(getattr(layer, f'q{key}'))
        return super().load_weight(layer, key)


@register
class SQEinsumDenseHandler(SQLayerHandler, KV3EinsumDenseHandler):
    handles = (
        'squark.layers.core.einsum_dense.QEinsumDense',
        'squark.layers.einsum_dense_batchnorm.QEinsumDenseBatchnorm',
    )


@register
class SQStandaloneQuantizerHandler(KerasV3LayerHandler):
    handles = ('squark.quantizer.quantizer.Quantizer',)

    def handle(
        self,
        layer: 'squark.quantizer.Quantizer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        conf = extract_fixed_quantizer_config(layer, in_tensors[0], True)
        del conf['output_keras_tensor_names']
        return conf


@register
class SQConvHandler(SQLayerHandler, KV3ConvHandler):
    handles = (
        'squark.layers.conv.QConv1D',
        'squark.layers.conv.QConv2D',
        # 'squark.layers.conv.QConv3D',
    )


@register
class SQDenseHandler(SQLayerHandler, KV3DenseHandler):
    handles = ('squark.layers.core.dense.QDense',)


@register
class SQActivationHandler(SQLayerHandler, KV3ActivationHandler):
    handles = ('squark.layers.activation.QActivation',)
