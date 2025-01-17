import glob
import os
from shutil import copy

from hls4ml.writer.vivado_writer import VivadoWriter


class VitisWriter(VivadoWriter):
    def __init__(self):
        super().__init__()

    def write_nnet_utils_overrides(self, model):
        ###################
        # nnet_utils
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/vitis/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copy(srcpath + h, dstpath + h)

    def write_board_script(self, model):
        '''
        Write the tcl scripts and kernel sources to create a Vitis IPI
        '''

        ###################
        # project.tcl
        ###################

        f = open(f'{model.config.get_output_dir()}/project.tcl', 'w')
        f.write('variable project_name\n')
        f.write(f'set project_name "{model.config.get_project_name()}"\n')
        f.write('variable backend\n')
        f.write('set backend "vitis"\n')
        f.write('variable part\n')
        f.write('set part "{}"\n'.format(model.config.get_config_value('Part')))
        f.write('variable clock_period\n')
        f.write('set clock_period {}\n'.format(model.config.get_config_value('ClockPeriod')))
        f.write('variable clock_uncertainty\n')
        f.write('set clock_uncertainty {}\n'.format(model.config.get_config_value('ClockUncertainty', '12.5%')))
        f.write('variable version\n')
        f.write('set version "{}"\n'.format(model.config.get_config_value('Version', '1.0.0')))
        f.close()
        return

    def write_hls(self, model):
        """
        Write the HLS project. Calls the steps from VivadoWriter, adapted for Vitis
        """
        super().write_hls(model)
        self.write_nnet_utils_overrides(model)
        self.write_tar(model)
        self.write_board_script(model)
