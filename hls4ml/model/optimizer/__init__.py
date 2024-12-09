import os

from hls4ml.model.flow.flow import register_flow
from hls4ml.model.optimizer.optimizer import (  # noqa: F401
    ConfigurableOptimizerPass,
    GlobalOptimizerPass,
    LayerOptimizerPass,
    ModelOptimizerPass,
    OptimizerPass,
    extract_optimizers_from_object,
    extract_optimizers_from_path,
    get_available_passes,
    get_backend_passes,
    get_optimizer,
    layer_optimizer,
    model_optimizer,
    optimize_model,
    optimizer_pass,
    register_pass,
)

opt_path = os.path.dirname(__file__) + '/passes'
module_path = __name__ + '.passes'

optimizers = extract_optimizers_from_path(opt_path, module_path)
for opt_name, opt in optimizers.items():
    register_pass(opt_name, opt)

del opt_path
del module_path
del optimizers

register_flow(
    'parse_qonnx',
    [
        'reshape_constant',
        'resize_remove_constants',
        'quant_constant_parameters',
        'quant_to_activation',
        'fuse_quant_with_constant',
        'const_quant_to_const_alpha',
        'quant_to_alpha_activation_alpha',
        'batch_norm_onnx_constant_parameters',
        'constant_batch_norm_fusion',
        'merge_two_constants',
        'scale_down_add',
        'bias_down_add',
        'scale_down_mat_mul',
        'scale_down_conv',
        'merge_to_apply_alpha',
        'merge_to_apply_alpha_div',
        'matmul_const_to_dense',
        'conv_to_conv_x_d',
        'conv_to_depthwise_conv_x_d',
    ],
)

register_flow(
    'convert',
    [
        'channels_last_converter',
        'seperable_to_depthwise_and_conv',
        'remove_transpose_before_flatten',
        'remove_nop_transpose',
        'remove_single_channel_transpose',
        'fuse_bias_add',
        'expand_layer_group',
        'output_rounding_saturation_mode',
        'qkeras_factorize_alpha',
        'extract_ternary_threshold',
        'fuse_consecutive_batch_normalization',
        'fuse_batch_normalization',
        'replace_multidimensional_dense_with_conv',
        # 'enforce_proxy_model_embedded_config',
        'bit_exact',
        'fuse_fixed_point_quantizer',
        'fix_input_precision',
        'eliminate_linear_activation',
        'merge_linear_activation',
        # many of the above optimzers need to be done before this
        'infer_precision_types',
    ],
    requires=['parse_qonnx'],
)  # TODO Maybe not all QKeras optmizers belong here?

register_flow(
    'optimize',
    [
        'remove_nop_batch_normalization',
    ],
    requires=['convert'],
)
