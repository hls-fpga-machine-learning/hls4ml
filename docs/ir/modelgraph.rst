================
ModelGraph Class
================

This page documents our ``ModelGraph`` class usage. You can generate generate an instance of this class through ``hls4ml``'s API, for example by converting a Keras model:

.. code-block:: python

   import hls4ml

   # Generate a simple configuration from keras model
   config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')

   # Convert to a ModelGraph instance (hls_model)
   hls_model = hls4ml.converters.convert_from_keras_model(keras_model, hls_config=config, output_dir='test_prj')

This object can be used to perform common simulation and firmware-generation tasks. Here is a list of important user-facing methods:


* :ref:`write <write-method>`
* :ref:`compile <compile-method>`
* :ref:`predict <predict-method>`
* :ref:`build <build-method>`
* :ref:`trace <trace-method>`

----

.. _write-method:

``write`` method
====================

Write the ``ModelGraph`` to the output directory specified in the config:

.. code-block:: python

   hls_model.write()

----

.. _compile-method:

``compile`` method
======================

Compiles the written C++/HLS code and links it into the Python runtime. Compiled model can be used to evaluate performance (accuracy) through ``predict()`` method.

.. code-block:: python

   hls_model.compile()

----

.. _predict-method:

``predict`` method
======================

Similar to ``keras``\ 's predict API, you can get the predictions just by supplying an input ``numpy`` array:

.. code-block:: python

   # Suppose that you already have input array X
   # Note that you have to do hls_model.compile() before using predict

   y = hls_model.predict(X)

This is similar to doing ``csim`` simulation, without creating the testbench and supplying data. It's very helpful when you want to quickly prototype different configurations for your model.

----

.. _build-method:

``build`` method
====================

This method "builds" the generated HLS project. The parameters of build are backend-specific and usually include simulation and synthesis. Refer to each backend for a complete list of supported parameters to ``build()``.

.. code-block:: python

   report = hls_model.build()

   #You can also read the report of the build
   hls4ml.report.read_vivado_report('hls4ml_prj')

The returned ``report`` object will contain the result of build step, which may include C-simulation results, HLS synthesis estimates, co-simulation latency etc, depending on the backend used.

----

.. _trace-method:

``trace`` method
====================

The trace method is an advanced version of the ``predict`` method. It's used to trace individual outputs from each layer of the hls_model. This is useful for debugging and setting the appropriate configuration.

**Return:** A dictionary where the keys are the names of the layers, and its values are the layers's outputs.

.. code-block:: python

   predict_ouputs, trace_outputs =  hls_model.trace(X)

   #We also support a similar function for keras
   keras_trace = hls4ml.model.profiling.get_ymodel_keras(keras_model, X)

----

.. _hardware_predict-method:

``hardware_predict`` method
===========================

A specialized version of the ``predict`` method, for the VitisAccelerator backend after a successful build. Runs the project on the FPGA and obtains prediction for the supplied numpy array.

**Note:** The host code being run under the hood is an example written for generic benchmarking purposes, helpful for validating projects and gauging maximum throughput. It should be further adapted for more specific applications. Currently, the maximum number of input samples that can be processed is ``batchsize * num_cu * num_buffer``. If the input array exceeds that size, the additional samples will be ignored.

An optional ``target`` argument can be used to specify the target emulation mode (``hw``, ``sw_emu``, ``hw_emu``) to run the project on. The default is ``hw``.

.. code-block:: python

   # Suppose that you already have input array X
   # Note that you have to do both hls_model.compile() and hls_model.build(), ensuring the
   # .xclbin file is successfully created, before using hardware_predict

   y = hls_model.hardware_predict(X)
