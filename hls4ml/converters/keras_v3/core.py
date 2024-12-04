import inspect
import typing
from typing import Any, Sequence

import numpy as np

from ._base import KerasV3LayerHandler, register

if typing.TYPE_CHECKING:
    import keras
    from keras.api import KerasTensor
    from keras.src.layers.merging.base_merge import Merge


@register
class KV3DenseHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.core.dense.Dense',)

    def handle(
        self,
        layer: 'keras.layers.Dense',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):

        kernel = self.load_weight(layer, 'kernel')
        bias = self.load_weight(layer, 'bias') if layer.use_bias else None
        n_in, n_out = kernel.shape

        config = {
            'data_format': 'channels_last',
            'weight_data': kernel,
            'bias_data': bias,
            'n_out': n_out,
            'n_in': n_in,
        }
        return config


@register
class KV3InputHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.core.input_layer.InputLayer',)

    def handle(
        self,
        layer: 'keras.layers.InputLayer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {'input_shape': list(layer._batch_shape[1:])}
        return config


@register
class KV3MergeHandler(KerasV3LayerHandler):
    handles = (
        'keras.src.layers.merging.add.Add',
        'keras.src.layers.merging.multiply.Multiply',
        'keras.src.layers.merging.average.Average',
        'keras.src.layers.merging.maximum.Maximum',
        'keras.src.layers.merging.minimum.Minimum',
        'keras.src.layers.merging.concatenate.Concatenate',
        'keras.src.layers.merging.subtract.Subtract',
        'keras.src.layers.merging.dot.Dot',
    )

    def handle(
        self,
        layer: 'Merge',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        assert len(out_tensors) == 1, f"Merge layer {layer.name} has more than one output"
        output_shape = list(out_tensors[0].shape[1:])

        config: dict[str, Any] = {
            'output_shape': output_shape,
            'op': layer.__class__.__name__.lower(),
        }

        match layer.__class__.__name__:
            case 'Concatenate':
                rank = len(output_shape)
                class_name = f'Concatenate{rank}d'
                config['axis'] = layer.axis
            case 'Dot':
                class_name = f'Dot{len(output_shape)}d'
                rank = len(output_shape)
                assert rank == 1, f"Dot product only supported for 1D tensors, got {rank}D on layer {layer.name}"
            case _:
                class_name = 'Merge'

        config['class_name'] = class_name
        return config


@register
class KV3ActivationHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.activations.activation.Activation',)

    def handle(
        self,
        layer: 'keras.layers.Activation',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        import keras

        config = {}
        config.update(self.default_config)

        activation = getattr(layer, 'activation', keras.activations.linear)
        match activation:
            case keras.activations.softmax:
                class_name = 'Softmax'
                config['axis'] = -1
            case keras.activations.hard_sigmoid:
                class_name = 'HardActivation'
            case keras.activations.leaky_relu:
                class_name = 'LeakyReLU'
                signature = inspect.signature(keras.activations.leaky_relu)
                config['activ_param'] = signature.parameters['negative_slope'].default
            case keras.activations.elu:
                class_name = 'ELU'
                signature = inspect.signature(keras.activations.elu)
                config['activ_param'] = signature.parameters['alpha'].default
            case _:
                class_name = 'Activation'

        config['activation'] = activation.__name__
        config['class_name'] = class_name
        return (config,)


@register
class KV3ReLUHandler(KerasV3LayerHandler):
    handles = (
        'keras.src.layers.activations.leaky_relu.LeakyReLU',
        'keras.src.layers.activations.prelu.PReLU',
        'keras.src.layers.activations.relu.ReLU',
    )

    def handle(
        self,
        layer: 'keras.layers.ReLU',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {}
        config.update(self.default_config)

        if layer.__class__.__name__ == 'ReLU':
            config['class_name'] = 'Activation'
            config['activation'] = 'relu'
            return config

        if layer.__class__.__name__ == 'PReLU':
            config['class_name'] = 'PReLU'
            config['param_data'] = np.array(layer.alpha)
            config['activation'] = 'prelu'
        else:
            config['class_name'] = 'LeakyReLU'
            config['activ_param'] = float(layer.negative_slope)
            config['activation'] = 'leaky_relu'

        return (config,)


@register
class KV3SoftmaxHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.activations.softmax.Softmax',)

    def handle(
        self,
        layer: 'keras.layers.Softmax',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {}
        config.update(self.default_config)

        config['class_name'] = 'Softmax'
        config['axis'] = layer.axis
        config['activation'] = 'softmax'

        return (config,)


@register
class KV3HardActivationHandler(KerasV3LayerHandler):
    handles = ('keras.src.layers.activations.elu.ELU',)

    def handle(
        self,
        layer: 'keras.layers.ELU',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        config = {}
        config.update(self.default_config)

        config['class_name'] = 'ELU'
        config['activ_param'] = float(layer.alpha)
        config['activation'] = 'elu'

        return (config,)
