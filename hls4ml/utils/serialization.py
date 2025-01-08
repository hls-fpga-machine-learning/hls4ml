import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from hls4ml.model.graph import HLSConfig, ModelGraph
from hls4ml.model.optimizer import optimize_model

from .._version import version


def serialize_model(model, file_path):
    arch_dict = {}
    for name, layer in model.graph.items():
        arch_dict[name] = layer.serialize()

    config_dict = model.config.serialize()
    graph_state_dict = model.serialize()

    with tempfile.TemporaryDirectory(prefix='hls4ml_model_') as tmpdir:
        dest_path = Path(tmpdir)
        for layer_name, layer_dict in arch_dict.items():
            _serialize_array_attrs(layer_dict, layer_name, dest_path)

        # Save the model architecture (ModelGraph.graph)
        arch_path = dest_path / 'model_arch.json'
        with open(arch_path, 'w') as arch_file:
            json.dump(arch_dict, arch_file, indent=4)

        # Save the model config (ModelGraph.config)
        config_path = dest_path / 'config.json'
        with open(config_path, 'w') as config_file:
            json.dump(config_dict, config_file, indent=4)

        if config_dict.get('InputData', None) is not None:
            tb_data_src_path = Path(config_dict['InputData'])
            tb_data_dst_path = dest_path / ('input_data_tb' + tb_data_src_path.suffix)
            tb_data_dst_path.write_bytes(tb_data_src_path.read_bytes())
        if config_dict.get('OutputPredictions', None) is not None:
            tb_data_src_path = Path(config_dict['OutputPredictions'])
            tb_data_dst_path = dest_path / ('output_data_tb' + tb_data_src_path.suffix)
            tb_data_dst_path.write_bytes(tb_data_src_path.read_bytes())

        # Save internal state (ModelGraph.inputs, .outputs, ._applied_flows)
        state_path = dest_path / 'graph_state.json'
        with open(state_path, 'w') as state_file:
            json.dump(graph_state_dict, state_file, indent=4)

        # Save version (hls4ml.version)
        version_path = dest_path / 'version.json'
        with open(version_path, 'w') as version_file:
            version_dict = {
                'hls4ml': version,
                'model_graph': '1',
                # Leave space for versioning other things in the future (like layers)
            }
            json.dump(version_dict, version_file, indent=4)

        # Pack it all in a tar.gz but with a .fml extension
        if isinstance(file_path, str):
            if not file_path.endswith('.fml'):
                file_path += '.fml'
            tar_path = Path(file_path)
        elif isinstance(file_path, Path):
            tar_path = file_path.with_suffix('.fml')

        if tar_path.exists():
            os.remove(tar_path)
        with tarfile.open(tar_path, mode='w:gz') as archive:
            archive.add(dest_path, recursive=True, arcname='')


def deserialize_model(file_path, output_dir=None):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if output_dir is None:
        output_dir = file_path.parent
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    with tempfile.TemporaryDirectory(prefix='hls4ml_model_') as tmpdir:
        with tarfile.open(file_path, mode='r:gz') as archive:
            archive.extractall(tmpdir)  # TODO For safety, we should only extract relevant files

        src_path = Path(tmpdir)

        # Load the model config (ModelGraph.config)
        config_path = src_path / 'config.json'
        with open(config_path) as config_file:
            config_state = json.load(config_file)

        config_dict = config_state['config']
        if config_dict.get('InputData', None) is not None:
            tb_data_src_path = src_path / ('input_data_tb' + Path(config_dict['InputData']).suffix)
            tb_data_dst_path = output_dir / ('input_data_tb' + Path(config_dict['InputData']).suffix)
            tb_data_dst_path.write_bytes(tb_data_src_path.read_bytes())
            config_dict['InputData'] = str(tb_data_dst_path)
        if config_dict.get('OutputPredictions', None) is not None:
            tb_data_src_path = src_path / ('output_data_tb' + Path(config_dict['OutputPredictions']).suffix)
            tb_data_dst_path = output_dir / ('output_data_tb' + Path(config_dict['OutputPredictions']).suffix)
            tb_data_dst_path.write_bytes(tb_data_src_path.read_bytes())
            config_dict['OutputPredictions'] = str(tb_data_dst_path)

        config = HLSConfig.deserialize(config_state)

        # Load internal state (ModelGraph.inputs, .outputs, ._applied_flows)
        state_path = src_path / 'graph_state.json'
        with open(state_path) as state_file:
            graph_state_dict = json.load(state_file)

        model = ModelGraph.from_saved_state(config, graph_state_dict)

        # Load the model architecture (ModelGraph.graph)
        arch_path = src_path / 'model_arch.json'
        with open(arch_path) as arch_file:
            arch_dict = json.load(arch_file)
            for layer_name, layer_state in arch_dict.items():
                _deserialize_array_attrs(src_path, layer_state)
                kind = _deserialize_class_name(layer_state['class_name'])
                attributes = _deserialize_layer_attrs(layer_state['state']['attributes'])
                inputs = layer_state['state']['inputs']
                outputs = layer_state['state']['outputs']
                node = model.make_node(kind, layer_name, attributes, inputs, outputs, initialize=False)
                model.graph[layer_name] = node

    # This is a temporary hack until we restructure so we can apply the type transformation flow more intuintively
    _reapply_type_conversion_flow(model)

    return model


def _serialize_array_attrs(attr_dict, layer_name, dest_dir):
    for attr_name, attr_val in attr_dict.items():
        if isinstance(attr_val, dict):
            _serialize_array_attrs(attr_val, layer_name, dest_dir)
        if isinstance(attr_val, np.ndarray):
            # arr_name ensures a nicer name for the data of weight variables and avoids name-clashing
            arr_name = layer_name
            if 'name' in attr_dict and attr_dict['name'] != layer_name:
                arr_name += '_' + attr_dict['name']
            arr_name += '_' + attr_name
            serialized_name = _serialize_ndarray(attr_val, arr_name, dest_dir)
            attr_dict[attr_name] = serialized_name
        if isinstance(attr_val, np.integer):
            attr_dict[attr_name] = int(attr_val)
        if isinstance(attr_val, np.floating):
            attr_dict[attr_name] = float(attr_val)


def _serialize_ndarray(arr, name_prefix, dest_dir):
    arr_path = dest_dir / (name_prefix + '.npy')
    np.save(arr_path, arr, allow_pickle=False)
    return '@ndarray:' + str(arr_path.name)


def _deserialize_layer_attrs(layer_attrs):
    deserialized_attrs = {}
    for attr_name, attr_val in layer_attrs.items():
        if isinstance(attr_val, dict) and 'class_name' in attr_val and 'state' in attr_val:
            deserialized_attrs[attr_name] = _deserialize_type(attr_val['class_name'], attr_val['state'])
        else:
            deserialized_attrs[attr_name] = attr_val

    return deserialized_attrs


def _deserialize_array_attrs(src_dir, attr_dict):
    for attr_name, attr_val in attr_dict.items():
        if isinstance(attr_val, dict):
            _deserialize_array_attrs(src_dir, attr_val)
        if isinstance(attr_val, str) and attr_val.startswith('@ndarray:'):
            arr = _deserialize_ndarray(src_dir, attr_val)
            attr_dict[attr_name] = arr


def _deserialize_ndarray(src_dir, arr_name):
    arr_name = arr_name.replace('@ndarray:', '')
    arr_path = src_dir / arr_name
    arr = np.load(arr_path, allow_pickle=False)
    return arr


def _deserialize_class_name(full_class_name):
    module_name, class_name = full_class_name.rsplit('.', 1)
    module = sys.modules[module_name]

    cls = getattr(module, class_name, None)
    if cls is None:
        raise Exception(f'Cannot deserialize class: {full_class_name}')

    return cls


def _deserialize_type(full_class_name, state):
    cls = _deserialize_class_name(full_class_name)

    nested_types = [
        type_name
        for type_name, type_val in state.items()
        if isinstance(type_val, dict) and 'class_name' in type_val and 'state' in type_val
    ]
    for atype in nested_types:
        state[atype] = _deserialize_type(state[atype]['class_name'], state[atype]['state'])

    return cls.deserialize(state)


def _reapply_type_conversion_flow(model):
    def _find_transform_types_opt(model):
        transform_type_opt = None
        for flow_run in model._applied_flows:
            for applied_opts in flow_run.values():
                for opt in applied_opts:
                    if 'transform_types' in opt:
                        transform_type_opt = opt
                        return transform_type_opt

        return transform_type_opt

    transform_type_opt = _find_transform_types_opt(model)
    if transform_type_opt is not None:
        optimize_model(model, [transform_type_opt])
