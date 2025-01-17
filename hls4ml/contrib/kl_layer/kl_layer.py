"""
    Usage example for a custom KL loss layer
    Takes as an input two arrays: z_mean and z_log_var
    and computes KL "distance" between normal distribution
    and Gaussian with mu=z_mean and sigma=z_log_var

    The HLS part is in contrib/kl_layer/kl_layer.h
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from keras.layers.merge import _Merge as Merge
except Exception:
    from keras.layers.merging.base_merge import _Merge as Merge

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops

import hls4ml
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.model.attributes import ConfigurableAttribute, TypeAttribute
from hls4ml.model.types import FixedPrecisionType, RoundingMode, SaturationMode


# Keras implementation of a KL layer
class KLLoss(Merge):
    '''Keras implementation of a KL loss custom layer'''

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super().build(input_shape)

    def _merge_function(self, inputs):
        mean = inputs[0]
        log_var = inputs[1]

        kl = 1.0 + log_var - math_ops.square(mean) - math_ops.exp(log_var)
        kl = -0.5 * math_ops.reduce_mean(kl, axis=-1, keepdims=True)

        return kl


# hls4ml implementations
class HKLLoss(hls4ml.model.layers.Layer):
    '''hls4ml implementation of a KL loss custom layer'''

    _expected_attributes = [
        ConfigurableAttribute('table_size', default=1024),
        ConfigurableAttribute('exp_range', default=8),
        TypeAttribute('accum'),
        TypeAttribute(
            'sum',
            default=FixedPrecisionType(18, 8, rounding_mode=RoundingMode.RND, saturation_mode=SaturationMode.SAT),
        ),
        TypeAttribute(
            'exp_table',
            default=FixedPrecisionType(18, 8, rounding_mode=RoundingMode.RND, saturation_mode=SaturationMode.SAT),
        ),
    ]

    def initialize(self):
        self.add_output_variable(shape=[1], dim_names=[f'KL_LOSS_{self.index}'])


# Templates
distance_config_template = """struct config{index} : nnet::distance_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = 1;
    typedef {accum_t.name} accum_t;
    typedef {sum_t.name} sum_t;
    typedef {exp_table_t.name} exp_table_t;
    static const unsigned table_size = {table_size};
    static constexpr float exp_range = {exp_range};
}};\n"""
distance_function_template = 'nnet::klloss<{input1_t}, {input2_t}, {output_t}, {config}>({input1}, {input2}, {output});'
distance_include_list = ['nnet_utils/kl_layer.h']


class HKLLossConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HKLLoss)
        self.template = distance_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_in'] = node.get_input_variable(node.inputs[0]).shape[0]
        params['n_out'] = 1
        return self.template.format(**params)


class HKLLossFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(HKLLoss, include_header=distance_include_list)
        self.template = distance_function_template

    def format(self, node):
        params = {}
        params['config'] = f'config{node.index}'
        params['input1_t'] = node.get_input_variable(node.inputs[0]).type.name
        params['input2_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['output_t'] = node.get_output_variable().type.name
        params['input1'] = node.get_input_variable(node.inputs[0]).name
        params['input2'] = node.get_input_variable(node.inputs[1]).name
        params['output'] = node.get_output_variable().name

        return self.template.format(**params)


# Parser for converter
def parse_klloss_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'KLLoss' in keras_layer['class_name']

    layer = parse_default_keras_layer(keras_layer, input_names)

    output_shape = [input_shapes[0][0], 1]

    return layer, output_shape


def main():
    # Register the converter for custom Keras layer
    hls4ml.converters.register_keras_layer_handler('KLLoss', parse_klloss_layer)

    # Register the hls4ml's IR layer
    hls4ml.model.layers.register_layer('KLLoss', HKLLoss)

    # Register the optimization passes (if any)
    backend = hls4ml.backends.get_backend('Vivado')

    # Register template passes for the given backend
    backend.register_template(HKLLossConfigTemplate)
    backend.register_template(HKLLossFunctionTemplate)

    # Register HLS implementation
    p = Path(__file__).parent / 'kl_layer.h'
    backend.register_source(p)

    # Test if it works
    # Create a dummy Keras model with KL loss layer
    inp = tf.keras.layers.Input(shape=(19, 3, 1))
    z_mean = tf.keras.layers.Dense(10)(inp)
    z_log_var = tf.keras.layers.Dense(10)(inp)
    custom_output = KLLoss()([z_mean, z_log_var])
    # create new model
    kmodel = tf.keras.models.Model(inputs=inp, outputs=custom_output)
    kmodel.summary()

    # test on random inputs
    x = np.random.randint(-5, 5, (1, 19, 3, 1), dtype='int32')
    kres = kmodel(x)

    # Create dummy config
    config = {}
    config['Model'] = {
        'Precision': 'ap_fixed<16,6>',
        'ReuseFactor': 1,
        'ParallelizationFactor': 1,
        'Strategy': 'Resource',
    }
    hmodel = hls4ml.converters.convert_from_keras_model(
        kmodel,
        output_dir='hls4mlprj_kl_layer',
        backend='Vivado',
        io_type='io_parallel',
        part='xcvu9p-flga2577-2-e',
        hls_config=config,
    )

    hmodel.compile()
    hres = hmodel.predict(x.astype('float32'))

    print('Compare prediction by hls4ml model to Keras one')
    print(kres - hres)

    print('Building model')
    report = hmodel.build(reset=True, csim=False, cosim=True, synth=True, vsynth=True)
    print(report)


if __name__ == '__main__':
    main()
