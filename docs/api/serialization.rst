============================
Saving/Loading hls4ml models
============================

``hls4ml`` model objects (instances of ``ModelGraph`` class) can be serialized to disk and loaded at a later stage. The saved model doesn't require original Keras/PyTorch/ONNX model for loading.

To save/load a model use the following API:

.. code-block:: python

    from hls4ml.converters import convert_from_keras_model
    from hls4ml.utils.serialization import serialize_model, deserialize_model

    model = convert_from_keras_model(keras_model, ...)

    # Save a model to some path
    serialize_model(model, 'some/path/my_hls4ml_model.fml')

    # Load a model from a file
    loaded_model = deserialize_model('some/path/my_hls4ml_model.fml')


Saved model will have a ``.fml`` extension, but is in fact a gzipped tar archive. Loaded model can be used in the same way as the original one. This includes modification of certain config parameters, for example output directory, layer reuse factor etc.
