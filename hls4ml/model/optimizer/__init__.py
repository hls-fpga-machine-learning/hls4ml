from __future__ import absolute_import

from hls4ml.model.optimizer.optimizer import OptimizerPass, register_pass, get_optimizer, optimize_model, get_available_passes


from hls4ml.model.optimizer.passes.nop import EliminateLinearActivation
from hls4ml.model.optimizer.passes.bn_quant import MergeBatchNormAndQuantizedTanh
from hls4ml.model.optimizer.passes.bn_quant import QuantizeDenseOutput
from hls4ml.model.optimizer.passes.bn_fuse import FuseBatchNormalization
from hls4ml.model.optimizer.passes.fuse_biasadd import FuseBiasAdd
from hls4ml.model.optimizer.passes.conv_same_pad import InsertZeroPaddingBeforeConv1D
from hls4ml.model.optimizer.passes.conv_same_pad import InsertZeroPaddingBeforeConv2D
from hls4ml.model.optimizer.passes.pointwise import OptimizePointwiseConv
from hls4ml.model.optimizer.passes.clone import CloneOutput
from hls4ml.model.optimizer.passes.repack_stream import ReshapeStream

try:
    from hls4ml.model.optimizer.passes.qkeras import OutputRoundingSaturationMode
    from hls4ml.model.optimizer.passes.qkeras import QKerasFactorizeAlpha
    from hls4ml.model.optimizer.passes.qkeras import FuseConsecutiveBatchNormalization
    from hls4ml.model.optimizer.passes.qkeras import ExtractTernaryThreshold
    __qkeras_optimizers__ = True
except ImportError:
    __qkeras_optimizers__ = False

if __qkeras_optimizers__:
    register_pass('output_rounding_saturation_mode', OutputRoundingSaturationMode)
    register_pass('qkeras_factorize_alpha', QKerasFactorizeAlpha)
    register_pass('extract_ternary_threshold', ExtractTernaryThreshold)
    register_pass('fuse_consecutive_batch_normalization', FuseConsecutiveBatchNormalization) 

register_pass('eliminate_linear_activation', EliminateLinearActivation)
register_pass('merge_batch_norm_quantized_tanh', MergeBatchNormAndQuantizedTanh)
register_pass('quantize_dense_output', QuantizeDenseOutput)
register_pass('fuse_batch_norm', FuseBatchNormalization)
register_pass('fuse_biasadd', FuseBiasAdd)
register_pass('conv1d_same_pad', InsertZeroPaddingBeforeConv1D)
register_pass('conv2d_same_pad', InsertZeroPaddingBeforeConv2D)
register_pass('optimize_pointwise_conv', OptimizePointwiseConv)
register_pass('clone_output', CloneOutput)
register_pass('reshape_stream', ReshapeStream)

