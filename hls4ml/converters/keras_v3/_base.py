import typing
from types import FunctionType
from typing import Any, Callable, Sequence, TypedDict, overload


class DefaultConfig(TypedDict, total=False):
    name: str
    class_name: str
    module: str
    input_keras_tensor_names: list[str]
    input_shape: list[list[int]]
    output_keras_tensor_names: list[str]
    epsilon: float
    use_bias: bool
    data_format: str


if typing.TYPE_CHECKING:
    import keras
    from keras.api import KerasTensor

T_kv3_handler = Callable[
    ['keras.Layer', Sequence['keras.KerasTensor'], Sequence['keras.KerasTensor']], tuple[dict[str, Any], ...]
]

registry: dict[str, T_kv3_handler] = {}


@overload
def register(cls: type) -> type: ...


@overload
def register(cls: str) -> Callable[[T_kv3_handler], T_kv3_handler]: ...


def register(cls: str | type):
    """Decorator to register a handler for a specific layer class. Suggested to decorate the `KerasV3LayerHandler` class.

    Parameters
    ----------
    cls : str|type
        If str, the key to register the handler under. If type, the class to register the handler for.

    Examples
    --------
    ```python
    @keras_dispatcher.register
    class MyLayerHandler(KerasV3LayerHandler):
        handles = ('my_package.src.submodule.MyLayer', 'MyLayer2')

        def handle(self, layer, inp_tensors, out_tensors):
            # handler code


    @keras_dispatcher.register('MyLayer3')
    def my_layer_handler(layer, inp_tensors, out_tensors):
        # handler code
    ```
    """

    def deco(func):
        if isinstance(cls, str):
            registry[cls] = func
        for k in getattr(func, 'handles', ()):
            registry[k] = func
        if isinstance(cls, type):
            return cls
        return func

    if isinstance(cls, type):
        return deco(cls())
    return deco


def maybe_add_attrs(config: dict[str, Any] | DefaultConfig, obj: Any, *attrs: str):
    for attr in attrs:
        if attr not in config and hasattr(obj, attr):
            config[attr] = getattr(obj, attr)


class KerasV3LayerHandler:
    """Base class for keras v3 layer handlers. Subclass this class to create a handler for a specific layer type."""

    handles = ()

    def __call__(
        self,
        layer: 'keras.Layer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ) -> tuple[dict[str, Any], ...]:
        """Handle a keras layer. Return a tuple of dictionaries, each
        dictionary representing a layer (module) in the HLS model. One
        layer may correspond one or more dictionaries (e.g., layers with
        activation functions will be split into two layers).

        Some common attributes are automatically added to the dictionary
        if the handler returns a single dictionary. If the handler
        returns multiple dictionaries, the attributes must be added
        manually. Anything returned by the handler will override the
        automatic attributes.

        Automatic attributes: - name - class_name - module -
        input_keras_tensor_names - input_shape -
        output_keras_tensor_names

        If the layer has an activation function, an additional
        dictionary will be added to the return value representing the
        activation function.


        Parameters
        ----------
        layer : keras.Layer
            The layer to be converted to HLS configuration(s).
        in_tensors : Sequence[KerasTensor]
            The list of input tensors to the layer.
        out_tensors : Sequence[KerasTensor]
            The list of output tensors from the layer.

        Returns
        -------
        dict[str, Any] | tuple[dict[str, Any], ...]
            layer configuration(s) for the HLS model to be consumed by
            the ModelGraph constructor
        """

        name = layer.name
        class_name = layer.__class__.__name__
        module = layer.__module__

        default_config: DefaultConfig = {
            'name': name,
            'class_name': class_name,
            'module': module,
            'input_keras_tensor_names': [t.name for t in in_tensors],
            'input_shape': [list(t.shape[1:]) for t in in_tensors],  # type: ignore
            'output_keras_tensor_names': [t.name for t in out_tensors],
        }

        maybe_add_attrs(default_config, layer, 'epsilon', 'use_bias', 'data_format')

        mandatory_keys = ['name', 'class_name', 'output_keras_tensor_names', 'input_keras_tensor_names']

        self.default_config = default_config
        config0 = self.handle(layer, in_tensors, out_tensors)
        del self.default_config

        if isinstance(config0, tuple):
            for conf in config0:
                for key in mandatory_keys:
                    assert key in conf, f"Key {key} missing from layer {name} handled by {self.__class__.__name__}"
            return config0

        config = {}
        config.update(default_config)
        config.update(config0)
        ret = (config,)

        # If activation exists, append it

        act_config, intermediate_tensor_name = self.maybe_get_activation_config(layer, out_tensors)
        if act_config is not None:
            ret[0]['output_keras_tensor_names'] = [intermediate_tensor_name]
            ret = *ret, act_config

        return ret

    def maybe_get_activation_config(self, layer, out_tensors):
        import keras

        activation = getattr(layer, 'activation', None)
        name = layer.name
        if activation not in (keras.activations.linear, None):
            assert len(out_tensors) == 1, f"Layer {name} has more than one output, but has an activation function"
            assert isinstance(activation, FunctionType), f"Activation function for layer {name} is not a function"
            intermediate_tensor_name = f'{out_tensors[0].name}_activation'
            act_cls_name = activation.__name__
            act_config = {
                'class_name': 'Activation',
                'activation': act_cls_name,
                'name': f'{name}_{act_cls_name}',
                'input_keras_tensor_names': [intermediate_tensor_name],
                'output_keras_tensor_names': [out_tensors[0].name],
            }
            return act_config, intermediate_tensor_name
        return None, None

    def handle(
        self,
        layer: 'keras.Layer',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ) -> dict[str, Any] | tuple[dict[str, Any], ...]:
        return {}

    def load_weight(self, layer: 'keras.Layer', key: str):
        """Load a weight from a layer.

        Parameters
        ----------
        layer : keras.Layer
            The layer to load the weight from.
        key : str
            The key of the weight to load.

        Returns
        -------
        np.ndarray
            The weight.
        """
        import keras

        return keras.ops.convert_to_numpy(getattr(layer, key))
