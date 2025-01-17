from hls4ml.writer.catapult_writer import CatapultWriter
from hls4ml.writer.oneapi_writer import OneAPIWriter
from hls4ml.writer.quartus_writer import QuartusWriter
from hls4ml.writer.symbolic_writer import SymbolicExpressionWriter
from hls4ml.writer.vitis_accelerator_ip_flow_writer import VitisAcceleratorIPFlowWriter
from hls4ml.writer.vitis_writer import VitisWriter
from hls4ml.writer.vivado_accelerator_writer import VivadoAcceleratorWriter
from hls4ml.writer.vivado_writer import VivadoWriter
from hls4ml.writer.writers import Writer, get_writer, register_writer  # noqa: F401

register_writer('Vivado', VivadoWriter)
register_writer('VivadoAccelerator', VivadoAcceleratorWriter)
register_writer('Vitis', VitisWriter)
register_writer('VitisAcceleratorIPFlow', VitisAcceleratorIPFlowWriter)
register_writer('Quartus', QuartusWriter)
register_writer('oneAPI', OneAPIWriter)
register_writer('Catapult', CatapultWriter)
register_writer('SymbolicExpression', SymbolicExpressionWriter)
