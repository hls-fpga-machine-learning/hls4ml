from __future__ import print_function
import six
import os
import sys
import platform
import ctypes
import re
import numpy as np
import numpy.ctypeslib as npc
from collections import OrderedDict

from hls4ml.model.hls_layers import *
from hls4ml.templates import get_backend
from hls4ml.writer import get_writer
from hls4ml.model.optimizer import optimize_model, get_available_passes
from hls4ml.report.vivado_report import parse_vivado_report

class HLSConfig(object):
    def __init__(self, config):
        self.config = config

        self.backend = get_backend(self.config.get('Backend', 'Vivado'))
        self.writer = get_writer(self.config.get('Backend', 'Vivado'))

        self.model_precision = {}
        self.layer_type_precision = {}
        self.layer_name_precision = {}

        self.model_rf = None
        self.layer_type_rf = {}
        self.layer_name_rf = {}

        self.model_strategy = 'Latency'
        self.layer_type_strategy = {}
        self.layer_name_strategy = {}

        self.model_compression = False
        self.layer_type_compression = {}
        self.layer_name_compression = {}

        self.trace_output = self.get_config_value('TraceOutput', False)

        self._parse_hls_config()
        self._validate_hls_config()

    def get_config_value(self, key, default=None):
        return self.config.get(key, default)

    def get_project_name(self):
        return self.get_config_value('ProjectName')

    def get_output_dir(self):
        return self.get_config_value('OutputDir')

    def get_layer_config_value(self, layer, key, default=None):
        hls_config = self.config['HLSConfig']

        name_config = hls_config.get('LayerName', {}).get(layer.name.lower(), None)
        if name_config is not None:
            return name_config.get(key, default)

        type_config = hls_config.get('LayerType', {}).get(layer.__class__.__name__, None)
        if type_config is not None:
            return type_config.get(key, default)

        model_config = hls_config.get('Model', None)
        if model_config is not None:
            return model_config.get(key, default)

        return default

    def get_layer_config(self, layer):
        hls_config = self.config['HLSConfig']
        layer_config = {}

        type_config = hls_config.get('LayerType', {}).get(layer.__class__.__name__, None)
        if type_config is not None:
            layer_config.update(type_config)

        name_config = hls_config.get('LayerName', {}).get(layer.name.lower(), None)
        if name_config is not None:
            layer_config.update(name_config)

        return layer_config

    def get_precision(self, layer, var='default'):
        precision = self.layer_name_precision.get(layer.name.lower() + '_' + var)
        type_name = layer.name.lower() + '_' + var + '_t'
        if precision is None:
            precision = self.layer_name_precision.get(layer.name.lower() + '_default')
            type_name = layer.name.lower() + '_default_t'

        if precision is None:
            precision = self.layer_type_precision.get(layer.__class__.__name__.lower() + '_' + var)
            type_name = layer.__class__.__name__ + '_' + var + '_t'
        if precision is None:
            precision = self.layer_type_precision.get(layer.__class__.__name__.lower() + '_default')
            type_name = layer.__class__.__name__ + '_default_t'

        if precision is None:
            precision = self.model_precision.get(var)
            type_name = var + '_default_t'
        if precision is None:
            precision = self.model_precision.get('default')
            type_name = 'model_default_t'

        if precision is None:
            raise Exception('No precision for {}->{} found and no default specified.'.format(layer.name, var))

        precision = self.backend.convert_precision_string(precision)

        return (precision, type_name)

    def get_reuse_factor(self, layer):
        rf = self.layer_name_rf.get(layer.name.lower())
        if rf is None:
            rf = self.layer_type_rf.get(layer.__class__.__name__.lower())
        if rf is None:
            rf = self.model_rf

        if rf is None:
            raise Exception('No reuse factor for {} found and no default specified.'.format(layer.name))

        return rf

    def get_strategy(self, layer):
        strategy = self.layer_name_strategy.get(layer.name.lower())
        if strategy is None:
            strategy = self.layer_type_strategy.get(layer.__class__.__name__.lower())
        if strategy is None:
            strategy = self.model_strategy

        return strategy

    def is_resource_strategy(self, layer):
        return self.get_strategy(layer).lower() == 'resource'

    def get_compression(self, layer):
        compression = self.layer_name_compression.get(layer.name.lower())
        if compression is None:
            compression = self.layer_type_compression.get(layer.__class__.__name__.lower())
        if compression is None:
            compression = self.model_compression

        return compression

    def _parse_hls_config(self):
        hls_config = self.config['HLSConfig']
        
        self.optimizers = hls_config.get('Optimizers')
        if 'SkipOptimizers' in hls_config:
            if self.optimizers is not None:
                raise Exception('Invalid optimizer configuration, please use either "Optimizers" or "SkipOptimizers".')
            skip_optimizers = hls_config.get('SkipOptimizers')
            selected_optimizers = get_available_passes()
            for opt in skip_optimizers:
                try:
                    selected_optimizers.remove(opt)
                except ValueError:
                    pass                
            self.optimizers = selected_optimizers
        
        model_cfg = hls_config.get('Model')
        if model_cfg is not None:
            precision_cfg = model_cfg.get('Precision')
            if precision_cfg is not None:
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.model_precision[var] = precision
                else:
                    self.model_precision['default'] = precision_cfg # Default precision for everything

            self.model_rf = model_cfg.get('ReuseFactor')
            self.model_strategy = model_cfg.get('Strategy', 'Latency')
            self.model_compression = bool(model_cfg.get('Compression', 0))

        layer_type_cfg = hls_config.get('LayerType')
        if layer_type_cfg is not None:
            for layer_type, layer_cfg in layer_type_cfg.items():
                precision_cfg = layer_cfg.get('Precision')
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.layer_type_precision[layer_type.lower() + '_' + var] = precision
                else:
                    self.layer_type_precision[layer_type.lower() + '_default'] = precision_cfg

                rf = layer_cfg.get('ReuseFactor')
                if rf is not None:
                    self.layer_type_rf[layer_type.lower()] = rf

                strategy = layer_cfg.get('Strategy')
                if strategy is not None:
                    self.layer_type_strategy[layer_type.lower()] = strategy

                compression = layer_cfg.get('Compression')
                if compression is not None:
                    self.layer_type_compression[layer_type.lower()] = bool(compression)

        layer_name_cfg = hls_config.get('LayerName')
        if layer_name_cfg is not None:
            for layer_name, layer_cfg in layer_name_cfg.items():
                precision_cfg = layer_cfg.get('Precision')
                if isinstance(precision_cfg, dict):
                    for var, precision in precision_cfg.items():
                        self.layer_name_precision[layer_name.lower() + '_' + var] = precision
                else:
                    self.layer_name_precision[layer_name.lower() + '_default'] = precision_cfg

                rf = layer_cfg.get('ReuseFactor')
                if rf is not None:
                    self.layer_name_rf[layer_name.lower()] = rf

                strategy = layer_cfg.get('Strategy')
                if strategy is not None:
                    self.layer_name_strategy[layer_name.lower()] = strategy

                compression = layer_cfg.get('Compression')
                if compression is not None:
                    self.layer_name_compression[layer_name.lower()] = bool(compression)

    def _validate_hls_config(self):
        use_resource = False
        if self.model_strategy.lower() == 'latency' and self.model_compression:
            print('WARNING: Compression enabled while model strategy set to "Latency".')
            use_resource = True
        for layer_type, strategy in self.layer_type_strategy.items():
            if strategy.lower() == 'resource' and self.model_strategy.lower() == 'latency':
                print('WARNING: Strategy for layer type {} set to "Resource", while model strategy set to "Latency".'.format(layer_type))
                use_resource = True

        for layer_name, strategy in self.layer_name_strategy.items():
            if strategy.lower() == 'resource' and self.model_strategy.lower() == 'latency':
                print('WARNING: Strategy for layer {} set to "Resource", while model strategy set to "Latency".'.format(layer_name))
                use_resource = True

        for layer_type, compression in self.layer_type_compression.items():
            if compression and self.model_strategy.lower() == 'latency':
                print('WARNING: Compression enabled for layer type {}, while model strategy set to "Latency".'.format(layer_type))
                use_resource = True

        for layer_name, compression in self.layer_name_compression.items():
            if compression and self.model_strategy.lower() == 'latency':
                print('WARNING: Compression enabled for layer {}, while model strategy set to "Latency".'.format(layer_name))
                use_resource = True

        if use_resource:
            print('WARNING: Changing model strategy to "Resource"')
            self.model_strategy = 'Resource'

class HLSModel(object):
    def __init__(self, config, data_reader, layer_list, inputs=None, outputs=None):
        self.config = HLSConfig(config)
        self.reader = data_reader

        # If not provided, assumes layer_list[0] is input, and layer_list[-1] is output
        self.inputs = inputs if inputs is not None else [layer_list[0]['name']]
        self.outputs = outputs if outputs is not None else [layer_list[-1]['name']]

        self.index = 0
        self.graph = OrderedDict()
        self.output_vars = {}

        self._top_function_lib = None

        self._make_graph(layer_list)

        self._optimize_model(self.config.optimizers)

    def _make_graph(self, layer_list):
        for layer in layer_list:
            kind = layer['class_name']
            name = layer['name']
            inputs = layer.get('inputs', [])
            outputs = layer.get('outputs', [])
            if len(inputs) == 0:
                inputs = [next(reversed(self.graph), 'input')]
            if len(outputs) == 0:
                outputs = [name]

            self.graph[name] = self.make_node(kind, name, layer, inputs, outputs)

    def _optimize_model(self, optimizers):
        optimize_model(self, optimizers)

    def make_node(self, kind, name, attributes, inputs, outputs=None):
        """ Make a new node not connected to the model graph.

        The 'kind' should be a valid layer registered with `register_layer`. If no outputs
        are specified, a default output named the same as the node will be created. The 
        returned node should be added to the graph with `insert_node` or `replace_node`
        functions.

        Args:
            kind (str): Type of node to add
            name (str): Name of the node
            attributes (dict): Initial set of attributes required to construct the node (Layer)
            inputs (list): List of inputs to the layer
            outputs (list, optional): The optional list of named outputs of the node

        Raises:
            Exception: If an attempt to insert a node with multiple inputs is made or if
                `before` does not specify a correct node in sequence.

        Returns:
            Layer: The node created.
        """

        if kind not in layer_map:
            raise Exception('Layer {} not found in registry.'.format(kind))

        node = layer_map[kind](self, name, attributes, inputs, outputs)
        for o in node.outputs:
            out_var = node.get_output_variable(output_name=o)
            if o in self.outputs:
                out_var.type.name = 'result_t'
            self.output_vars[o] = out_var

        return node

    def insert_node(self, node, before=None):
        """ Insert a new node into the model graph.

        The node to be inserted should be created with `make_node()` function. The optional 
        parameter `before` can be used to specify the node that follows in case of ambiguities.

        Args:
            node (Layer): Node to insert
            before (Layer, optional): The next node in sequence before which a
                new node should be inserted. 
        Raises:
            Exception: If an attempt to insert a node with multiple inputs is made or if
                `before` does not specify a correct node in sequence.

        """
        if len(node.inputs) > 1:
            raise Exception('Cannot insert a node with more than one input (for now).')

        prev_node = self.graph.get(node.inputs[0])
        next_nodes = [x for x in self.graph.values() if x.inputs[0] == prev_node.outputs[0]]
        if before is None:
            next_node = next((x for x in self.graph.values() if x.inputs[0] == prev_node.outputs[0]), None)
        else:
            if before not in next_nodes:
                raise Exception('Cannot insert a node {} before {} (candidates: {}).'.format(node.name, before.name, ','.join([n.name for n in next_nodes])))
            next_node = before

        if next_node is not None:
            next_node.inputs[0] = node.outputs[0]
        else:
            self.outputs[0] = node.outputs[0]

        new_graph = OrderedDict()
        for k, v in self.graph.items():
            new_graph[k] = v
            if k == prev_node.name:
                new_graph[node.name] = node

        self.graph = new_graph

    def remove_node(self, node, rewire=True):
        """ Remove a node from a graph.

        By default, this function can connect the outputs of previous node to the input of next one.
        Note that when removing a leaf node `rewire` should be set to `False`.

        Args:
            node (Layer): The node to remove
            rewire (bool, optional): If `True`, connects the outputs of the previous node
                to the inputs of the next node

        Raises:
            Exception: If an attempt is made to rewire a leaf node or a node with multiple
                inputs/outpus.

        """
        if rewire:
            if len(node.inputs) > 1 or len(node.outputs) > 1:
                raise Exception('Cannot rewire a node with multiple inputs/outputs')
            prev_node = self.graph.get(node.inputs[0])
            next_node = next((x for x in self.graph.values() if node.outputs[0] in x.inputs), None)
            if prev_node is not None:
                if next_node is not None:
                    for i,_ in enumerate(next_node.inputs):
                        if node.outputs[0] == next_node.inputs[i]:
                            next_node.inputs[i] = prev_node.outputs[0]
                            break
                else:
                    if node.outputs[0] in self.outputs:
                        self.outputs = [prev_node.outputs[0] if x == node.outputs[0] else x for x in self.outputs]
                    else:
                        raise Exception('Cannot rewire a node without child')
            else:
                raise Exception('Cannot rewire a node without a parent')

        del self.output_vars[node.outputs[0]]
        del self.graph[node.name]

    def replace_node(self, old_node, new_node):
        """ Replace an existing node in the graph with a new one.

        Args:
            old_node (Layer): The node to replace
            new_node (Layer): The new node

        """
        prev_node = self.graph.get(old_node.inputs[0])
        next_node = next((x for x in self.graph.values() if x.inputs[0] == old_node.outputs[0]), None)
        if next_node is not None:
            next_node.inputs[0] = new_node.outputs[0]
        if prev_node is not None:
            if new_node.inputs is None or len(new_node.inputs) == 0: # Check if already rewired
                new_node.inputs = [prev_node.outputs[0]]

        self.graph = OrderedDict((new_node.name, new_node) if k == old_node.name else (k, v) for k, v in self.graph.items())

    def get_weights_data(self, layer_name, var_name):
        return self.reader.get_weights_data(layer_name, var_name)

    def next_layer(self):
        self.index += 1
        return self.index

    def get_layers(self):
        return self.graph.values()

    def get_input_variables(self):
        variables = []
        for inp in self.inputs:
            variables.append(self.graph[inp].get_output_variable())
        return variables

    def register_output_variable(self, out_name, variable):
        if out_name in self.outputs:
            variable.type.name = 'result_t'
        self.output_vars[out_name] = variable

    def get_output_variables(self):
        variables = []
        for out in self.outputs:
            variables.append(self.output_vars[out])
        return variables

    def get_layer_output_variable(self, output_name):
        return self.output_vars.get(output_name, None)

    def write(self):
        def make_stamp():
            from string import hexdigits
            from random import choice
            length = 8
            return ''.join(choice(hexdigits) for m in range(length))
        
        self.config.config['Stamp'] = make_stamp()

        self.config.writer.write_hls(self)

    def compile(self):
        self.write()

        curr_dir = os.getcwd()
        os.chdir(self.config.get_output_dir())

        try:
            ret_val = os.system('bash build_lib.sh')
            if ret_val != 0:
                raise Exception('Failed to compile project "{}"'.format(self.config.get_project_name()))
            lib_name = 'firmware/{}-{}.so'.format(self.config.get_project_name(), self.config.get_config_value('Stamp'))
            if self._top_function_lib is not None:

                if platform.system() == "Linux":
                    dlclose_func = ctypes.CDLL('libdl.so').dlclose
                elif platform.system() == "Darwin":
                    dlclose_func = ctypes.CDLL('libc.dylib').dlclose

                dlclose_func.argtypes = [ctypes.c_void_p]
                dlclose_func.restype = ctypes.c_int
                dlclose_func(self._top_function_lib._handle)
            self._top_function_lib = ctypes.cdll.LoadLibrary(lib_name)
        finally:
            os.chdir(curr_dir)

    def _get_top_function(self, x):
        if self._top_function_lib is None:
            raise Exception('Model not compiled')
        if len(self.get_input_variables()) > 1 or len(self.get_input_variables()) > 1:
            raise Exception('Calling "predict" on models with multiple inputs or outputs is not supported (yet)')

        if not isinstance(x, np.ndarray):
            raise Exception('Expected numpy.ndarray, but got {}'.format(type(x)))
        if not x.flags['C_CONTIGUOUS']:
            raise Exception('Array must be c_contiguous, try using numpy.ascontiguousarray(x)')

        if x.dtype in [np.single, np.float32]:
            top_function = getattr(self._top_function_lib, self.config.get_project_name() + '_float')
            ctype = ctypes.c_float
        elif x.dtype in [np.double, np.float64, np.float_]:
            top_function = getattr(self._top_function_lib, self.config.get_project_name() + '_double')
            ctype = ctypes.c_double
        else:
            raise Exception('Invalid type ({}) of numpy array. Supported types are: single, float32, double, float64, float_.'.format(x.dtype))

        top_function.restype = None
        top_function.argtypes = [npc.ndpointer(ctype, flags="C_CONTIGUOUS"), npc.ndpointer(ctype, flags="C_CONTIGUOUS"),
            ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]

        return top_function, ctype

    def _compute_n_samples(self, x):
        expected_size = self.get_input_variables()[0].size()
        x_size = np.prod(x.shape)
        n_samples, rem = divmod(x_size, expected_size)
        if rem != 0:
            raise Exception('Input size mismatch, got {}, expected {}'.format(x_size.shape, self.get_input_variables()[0].shape))

        return n_samples

    def predict(self, x):
        top_function, ctype = self._get_top_function(x)
        n_samples = self._compute_n_samples(x)

        curr_dir = os.getcwd()
        os.chdir(self.config.get_output_dir() + '/firmware')

        output = []
        if n_samples == 1:
            x = [x]

        try:
            for i in range(n_samples):
                predictions = np.zeros(self.get_output_variables()[0].size(), dtype=ctype)
                top_function(x[i], predictions, ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()))
                output.append(predictions)

            #Convert to numpy array
            output = np.asarray(output)
        finally:
            os.chdir(curr_dir)

        if n_samples == 1:
            return output[0]
        else:
            return output

    def trace(self, x):
        print('Recompiling {} with tracing'.format(self.config.get_project_name()))
        self.config.trace_output = True
        self.compile()

        top_function, ctype = self._get_top_function(x)
        n_samples = self._compute_n_samples(x)

        class TraceData(ctypes.Structure):
            _fields_ = [('name', ctypes.c_char_p),
                        ('data', ctypes.c_void_p)]

        trace_output = {}
        layer_sizes = {}
        n_traced = 0
        for layer in self.get_layers():
            if layer.function_cpp() and layer.get_attr('Trace', False):
                n_traced += len(layer.get_variables())
                trace_output[layer.name] = []
                layer_sizes[layer.name] = layer.get_output_variable().shape

        collect_func = self._top_function_lib.collect_trace_output
        collect_func.argtypes = [ctypes.POINTER(TraceData)]
        collect_func.restype = None
        trace_data = (TraceData * n_traced)()

        alloc_func = self._top_function_lib.allocate_trace_storage
        alloc_func.argtypes = [ctypes.c_size_t]
        alloc_func.restype = None

        free_func = self._top_function_lib.free_trace_storage
        free_func.argtypes = None
        free_func.restype = None

        curr_dir = os.getcwd()
        os.chdir(self.config.get_output_dir() + '/firmware')

        output = []
        if n_samples == 1:
            x = [x]

        try:
            alloc_func(ctypes.sizeof(ctype))

            for i in range(n_samples):
                predictions = np.zeros(self.get_output_variables()[0].size(), dtype=ctype)
                top_function(x[i], predictions, ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()))
                output.append(predictions)
                collect_func(trace_data)
                for trace in trace_data:
                    layer_name = str(trace.name, 'utf-8')
                    layer_data = ctypes.cast(trace.data, ctypes.POINTER(ctype))
                    np_array = np.ctypeslib.as_array(layer_data, shape=layer_sizes[layer_name])
                    trace_output[layer_name].append(np.copy(np_array))

            for key in trace_output.keys():
                trace_output[key] = np.asarray(trace_output[key])

            #Convert to numpy array
            output = np.asarray(output)

            free_func()
        finally:
            os.chdir(curr_dir)

        if n_samples == 1:
            return output[0], trace_output
        else:
            return output, trace_output

    def build(self, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, vsynth=False):
        if 'linux' in sys.platform:
            backend = self.config.get_config_value('Backend', 'Vivado')
            if backend == 'Vivado':
                found = os.system('command -v vivado_hls > /dev/null')
                if found != 0:
                    raise Exception('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')

            elif backend == 'Intel':
                raise NotImplementedError
            elif backend == 'Mentor':
                raise NotImplementedError
            else:
                raise Exception('Backend values can be [Vivado, Intel, Mentor]')

        if not os.path.exists(self.config.get_output_dir()):
            # Assume the project wasn't written before
            self.write()

        curr_dir = os.getcwd()
        os.chdir(self.config.get_output_dir())
        os.system('vivado_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} validation={validation} export={export} vsynth={vsynth}"'
            .format(reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, vsynth=vsynth))
        os.chdir(curr_dir)

        return parse_vivado_report(self.config.get_output_dir())

