import os
import sys

from hls4ml.backends import VivadoBackend
from hls4ml.model.flow import get_flow, register_flow
from hls4ml.model.layers import MultiHeadAttention
from hls4ml.model.optimizer import layer_optimizer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, NamedType
from hls4ml.report import parse_vivado_report


class VitisBackend(VivadoBackend):
    def __init__(self):
        super(VivadoBackend, self).__init__(name='Vitis')
        self._register_layer_attributes()
        self._register_flows()

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        validation_passes = [
            'vitis:validate_conv_implementation',
            'vitis:validate_resource_strategy',
            'vitis:validate_resource_unrolled_strategy',
        ]
        validation_flow = register_flow('validation', validation_passes, requires=['vivado:init_layers'], backend=self.name)

        # Any potential templates registered specifically for Vitis backend
        template_flow = register_flow(
            'apply_templates', self._get_layer_templates, requires=['vivado:init_layers'], backend=self.name
        )

        writer_passes = ['make_stamp', 'vitis:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['vitis:ip'], backend=self.name)

        ip_flow_requirements = get_flow('vivado:ip').requires.copy()
        ip_flow_requirements.insert(ip_flow_requirements.index('vivado:init_layers'), validation_flow)
        ip_flow_requirements.insert(ip_flow_requirements.index('vivado:streaming'), init_flow)
        ip_flow_requirements.insert(ip_flow_requirements.index('vivado:apply_templates'), template_flow)

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def create_initial_config(
        self,
        part='xcvu13p-flga2577-2-e',
        clock_period=5,
        clock_uncertainty='27%',
        io_type='io_parallel',
        namespace=None,
        write_weights_txt=True,
        write_tar=False,
        **_,
    ):
        """Create initial configuration of the Vitis backend.

        Args:
            part (str, optional): The FPGA part to be used. Defaults to 'xcvu13p-flga2577-2-e'.
            clock_period (int, optional): The clock period. Defaults to 5.
            clock_uncertainty (str, optional): The clock uncertainty. Defaults to 27%.
            io_type (str, optional): Type of implementation used. One of
                'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
            namespace (str, optional): If defined, place all generated code within a namespace. Defaults to None.
            write_weights_txt (bool, optional): If True, writes weights to .txt files which speeds up compilation.
                Defaults to True.
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file. Defaults to False.

        Returns:
            dict: initial configuration.
        """
        config = {}

        config['Part'] = part if part is not None else 'xcvu13p-flga2577-2-e'
        config['ClockPeriod'] = clock_period if clock_period is not None else 5
        config['ClockUncertainty'] = clock_uncertainty if clock_uncertainty is not None else '27%'
        config['IOType'] = io_type if io_type is not None else 'io_parallel'
        config['HLSConfig'] = {}
        config['WriterConfig'] = {
            'Namespace': namespace,
            'WriteWeightsTxt': write_weights_txt,
            'WriteTar': write_tar,
        }

        return config

    def build(self, model, reset=False, csim=True, synth=True, cosim=False, validation=False, export=False, vsynth=False):
        if 'linux' in sys.platform:
            found = os.system('command -v vitis_hls > /dev/null')
            if found != 0:
                raise Exception('Vitis HLS installation not found. Make sure "vitis_hls" is on PATH.')

        curr_dir = os.getcwd()
        os.chdir(model.config.get_output_dir())
        os.system(
            (
                'vitis_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} '
                'validation={validation} export={export} vsynth={vsynth}"'
            ).format(reset=reset, csim=csim, synth=synth, cosim=cosim, validation=validation, export=export, vsynth=vsynth)
        )
        os.chdir(curr_dir)

        return parse_vivado_report(model.config.get_output_dir())

    @layer_optimizer(MultiHeadAttention)
    def init_mha(self, layer):
        # TODO Allow getting recurrent reuse factor from the config
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('reuse_factor', reuse_factor)
        index_t = IntegerPrecisionType(width=1, signed=False)
        layer.set_attr('index_t', index_t)
        if 'table_t' not in layer.attributes:
            layer.set_attr(
                'table_t', NamedType(name=layer.name + '_table_t', precision=FixedPrecisionType(width=24, integer=8))
            )
        if 'table_size' not in layer.attributes:
            layer.set_attr('table_size', 2048)
        if 'accum_t' not in layer.attributes:
            layer.set_attr('accum_t', FixedPrecisionType(width=24, integer=8))
        if 'inv_range' not in layer.attributes:
            layer.set_attr('inv_range', 128)
        if 'exp_range' not in layer.attributes:
            layer.set_attr('exp_range', 8)
        layer.set_attr('strategy', 'resource')  # latency
