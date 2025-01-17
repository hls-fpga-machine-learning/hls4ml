import os
from shutil import copy, copytree

from hls4ml.writer.vitis_writer import VitisWriter


class VitisAcceleratorWriter(VitisWriter):
    def __init__(self):

        super().__init__()

    def create_accelerator_config(self, model):
        from hls4ml.backends import VitisAcceleratorConfig

        self.vitis_accelerator_config = VitisAcceleratorConfig(model.config)

    def write_parameters_overrides(self, model):
        """Write the C++ layer config file (parameters.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, "../templates/vivado/firmware/parameters.h"))
        fout = open(f"{model.config.get_output_dir()}/firmware/parameters.h", "w")

        for line in f.readlines():
            if "// hls-fpga-machine-learning insert includes" in line:
                newline = line
                for include in sorted(
                    set(
                        sum(
                            (layer.get_attr("include_header", []) for layer in model.get_layers()),
                            [],
                        )
                    )
                ):
                    newline += '#include "%s"\n' % include
                newline += '#include "defines.h"'

            elif "// hls-fpga-machine-learning insert weights" in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.storage.lower() != "bram":
                            newline += f'#include "weights/{w.name}.h"\n'

            elif "// hls-fpga-machine-learning insert layer-config" in line:
                newline = line
                for layer in model.get_layers():
                    config = layer.get_attr("config_cpp", None)
                    if config:
                        newline += "// " + layer.name + "\n"
                        newline += config + "\n"
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_build_script_backend_override(self, model):
        # project.tcl
        f = open(f"{model.config.get_output_dir()}/project.tcl", "w")
        f.write("variable project_name\n")
        f.write(f'set project_name "{model.config.get_project_name()}"\n')
        f.write("variable backend\n")
        f.write('set backend "vitisaccelerator"\n')
        f.write("variable part\n")
        f.write('set part "{}"\n'.format(model.config.get_config_value("Part")))
        f.write("variable clock_period\n")
        f.write("set clock_period {}\n".format(model.config.get_config_value("ClockPeriod")))
        f.write("variable clock_uncertainty\n")
        f.write("set clock_uncertainty {}\n".format(model.config.get_config_value("ClockUncertainty", "12.5%")))
        f.close()

    def write_kernel(self, model):
        """Write the Python-C++ kernel (kernel_wrapper.cpp & kernel_wrapper.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        io_type = model.config.get_config_value("IOType")

        # Writing header file
        f_header = open(os.path.join(filedir, "../templates/vitis_accelerator/kernel_wrapper.h"))
        fout_header = open(f"{model.config.get_output_dir()}/kernel_wrapper.h", "w")
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        if len(model_inputs) != 1 or len(model_outputs) != 1:
            raise Exception("Accelerator currently only supports projects with a single input and a single output variable")
        inp = model_inputs[0]
        out = model_outputs[0]
        for line in f_header.readlines():
            if "// hls-fpga-machine-learning accelerator parameters" in line:
                newline = ""
                newline += "#define NUM_CU " + format(self.vitis_accelerator_config.get_num_kernel()) + "\n"
                newline += "#define NUM_WORKER " + format(self.vitis_accelerator_config.get_num_worker()) + "\n"
                newline += "#define NUM_CHANNEL "
                if self.vitis_accelerator_config.get_memory_type() == "hbm":
                    newline += (
                        format(
                            self.vitis_accelerator_config.get_memory_channel_count()
                            // (2 * self.vitis_accelerator_config.get_num_kernel())
                        )
                        + "\n"
                    )
                elif self.vitis_accelerator_config.get_memory_type() == "ddr":
                    newline += "1\n"
                newline += "#define BATCHSIZE " + format(self.vitis_accelerator_config.get_batchsize()) + "\n"
            elif "// hls-fpga-machine-learning accelerator io" in line:
                newline = ""
                if io_type == "io_parallel":
                    newline += "#define DATA_SIZE_IN " + format(inp.size_cpp()) + "\n"
                    newline += "#define INSTREAMSIZE DATA_SIZE_IN" + "\n\n"
                    newline += "#define DATA_SIZE_OUT " + format(out.size_cpp()) + "\n"
                    newline += "#define OUTSTREAMSIZE DATA_SIZE_OUT" + "\n\n"
                    newline += "typedef " + format(inp.type.name) + " in_buffer_t;\n"
                    newline += "typedef " + format(out.type.name) + " out_buffer_t;\n"
                elif io_type == "io_stream":
                    dims, _ = zip(*inp.get_shape())
                    dims = list(dims)
                    nnet_array_depth = dims.pop()
                    dims.append("1")
                    newline += "#define DATA_SIZE_IN " + " * ".join(dims) + "\n"
                    newline += "#define NNET_ARRAY_DEPTH " + format(nnet_array_depth) + "\n"
                    newline += "#define INSTREAMSIZE (DATA_SIZE_IN * NNET_ARRAY_DEPTH)" + "\n\n"
                    newline += "#define DATA_SIZE_OUT " + format(out.size_cpp()) + "\n"
                    newline += "#define OUTSTREAMSIZE DATA_SIZE_OUT" + "\n\n"
                    newline += "typedef " + inp.type.precision.definition_cpp() + " in_buffer_t;\n"
                    newline += "typedef " + out.type.precision.definition_cpp() + " out_buffer_t;\n"
            else:
                newline = line
            fout_header.write(newline)
        f_header.close()
        fout_header.close()

        # Writing source file
        f_source = open(
            os.path.join(
                filedir,
                "../templates/vitis_accelerator/kernel_wrapper_" + io_type + ".cpp",
            )
        )
        fout_source = open(f"{model.config.get_output_dir()}/kernel_wrapper.cpp", "w")
        isHwQuant = self.vitis_accelerator_config.get_hw_quant()
        for line in f_source.readlines():
            if "myproject" in line:
                newline = line.replace("myproject", format(model.config.get_project_name()))
            elif "/*IN_HW_QUANT*/ " in line:
                newline = line.replace("/*IN_HW_QUANT*/ ", "(in_buffer_t)" if isHwQuant else "")
            elif "/*OUT_HW_QUANT*/ " in line:
                newline = line.replace("/*OUT_HW_QUANT*/ ", "(float)" if isHwQuant else "")
            else:
                newline = line

            if "/*IN_INTERFACE_TYPE*/" in newline:
                newline = newline.replace("/*IN_INTERFACE_TYPE*/", ("float" if isHwQuant else "in_buffer_t"))
            if "/*OUT_INTERFACE_TYPE*/" in newline:
                newline = newline.replace("/*OUT_INTERFACE_TYPE*/", ("float" if isHwQuant else "out_buffer_t"))

            fout_source.write(newline)
        f_source.close()
        fout_source.close()

    def write_host(self, model):
        """Write the OpenCL-based host code (myproject_host_cl.cpp) and associated libraries

        Args:
            model (ModelGraph): the hls4ml model.
        """
        # Write host code
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, "../templates/vitis_accelerator/myproject_host_cl.cpp"))
        fout = open(
            f"{model.config.get_output_dir()}/{model.config.get_project_name()}_host_cl.cpp",
            "w",
        )
        memoryType = self.vitis_accelerator_config.get_memory_type()
        isHwQuant = self.vitis_accelerator_config.get_hw_quant()
        for line in f.readlines():
            if "/*FPGA_Type*/" in line:
                newline = line.replace("/*FPGA_Type*/", memoryType.upper())
            elif "/*INTERFACE_TYPES*/" in line:
                dataTypes = "float, float" if isHwQuant else "in_buffer_t, out_buffer_t"
                newline = line.replace("/*INTERFACE_TYPES*/", dataTypes)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

        # Write libraries
        src = os.path.join(filedir, "../templates/vitis_accelerator/libs")
        dst = f"{model.config.get_output_dir()}/libs"
        copytree(src, dst, copy_function=copy, dirs_exist_ok=True)

    def write_makefile(self, model):
        """Write the Python-C++ Makefile (Makefile)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, "../templates/vitis_accelerator/Makefile"))
        fout = open(f"{model.config.get_output_dir()}/Makefile", "w")

        board_type = self.vitis_accelerator_config.get_board_type()
        project_name = format(model.config.get_project_name())
        for line in f.readlines():
            if "#PRJNAME" in line:
                newline = line.replace("#PRJNAME", project_name)
            elif "#BOARDTYPE" in line:
                newline = line.replace("#BOARDTYPE", board_type)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_accelerator_card_cfg(self, model):
        """Write the configuration file passed to Vivado/Vitis (accelerator_card.cfg)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        # Write accelerator_card.cfg
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, "../templates/vitis_accelerator/accelerator_card.cfg"))
        fout = open(f"{model.config.get_output_dir()}/accelerator_card.cfg", "w")

        memory_type = self.vitis_accelerator_config.get_memory_type()
        num_kernels = self.vitis_accelerator_config.get_num_kernel()
        num_channels = self.vitis_accelerator_config.get_memory_channel_count()
        if memory_type == "hbm":
            if num_kernels > 4:
                print(
                    "WARNING: You are trying to instantiate too many kernels on the FPGA."
                    "Synthesis is likely to fail due to resource shortage"
                )
            num_channels_per_cu = num_channels // (num_kernels * 2)
        elif memory_type == "ddr":
            if num_kernels > self.vitis_accelerator_config.get_memory_channel_count():
                raise Exception(
                    format(self.vitis_accelerator_config.get_platform())
                    + " has only "
                    + format(num_channels)
                    + " memory banks."
                )

        directives = self.vitis_accelerator_config.get_vivado_directives()

        for line in f.readlines():
            if "MYPLATFORM" in line:
                newline = line.replace("MYPLATFORM", format(self.vitis_accelerator_config.get_platform()))
            elif "# hls-fpga-machine-learning clock control" in line:
                freq = round(1e9 / model.config.get_config_value("ClockPeriod"))
                newline = f"clock={freq}:kernel_wrapper\n"
            elif "# hls-fpga-machine-learning kernel control" in line:
                newline = "[connectivity]\n"
                newline += "nk=kernel_wrapper:" + format(num_kernels) + "\n\n"
                if self.vitis_accelerator_config.get_board_type() == "alveo":
                    if memory_type == "hbm":
                        for i in range(0, num_kernels):
                            newline += "sp=kernel_wrapper_{}.in:HBM[{}:{}]\n".format(
                                i + 1,
                                (i * 2) * num_channels_per_cu,
                                ((i * 2 + 1) * num_channels_per_cu) - 1,
                            )
                            newline += "sp=kernel_wrapper_{}.out:HBM[{}:{}]\n".format(
                                i + 1,
                                (i * 2 + 1) * num_channels_per_cu,
                                ((i + 1) * 2) * num_channels_per_cu - 1,
                            )
                    elif memory_type == "ddr":
                        for i in range(0, num_kernels):
                            newline += f"sp=kernel_wrapper_{i + 1}.in:DDR[{i}]\n"
                            newline += f"sp=kernel_wrapper_{i + 1}.out:DDR[{i}]\n"
                            newline += "\n"
                        for i in range(0, num_kernels):
                            newline += f"slr=kernel_wrapper_{i + 1}:SLR{i}\n"
            elif "# hls-fpga-machine-learning vivado directives" in line:
                newline = ""
                if directives:
                    newline += "[vivado]\n"
                    for x in directives:
                        newline += x + "\n"
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

        # Write hls_config.tcl
        tcl_f = open(os.path.join(filedir, "../templates/vitis_accelerator/hls_config.tcl"))
        tcl_fout = open(f"{model.config.get_output_dir()}/hls_config.tcl", "w")
        for line in tcl_f.readlines():
            newline = line
            tcl_fout.write(newline)
        tcl_fout.write("\nset_clock_uncertainty {}\n".format(model.config.get_config_value("ClockUncertainty", "12.5%")))
        tcl_f.close()
        tcl_fout.close()

    def write_nnet_utils_overrides(self, model):
        """Override nnet_types.h pointer comparison

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        srcpath = os.path.join(filedir, "../templates/vitis_accelerator/nnet_utils/")
        dstpath = f"{model.config.get_output_dir()}/firmware/nnet_utils/"
        copy(srcpath + "nnet_types.h", dstpath + "nnet_types.h")

    def write_hls(self, model):
        """
        Write the HLS project. Calls the steps from VivadoWriter, adapted for Vitis
        """
        super().write_hls(model)
        print("\n\nWriting Accelerator code")
        self.create_accelerator_config(model)
        self.write_nnet_utils_overrides(model)
        self.write_build_script_backend_override(model)
        self.write_parameters_overrides(model)
        self.write_kernel(model)
        self.write_host(model)
        self.write_makefile(model)
        self.write_accelerator_card_cfg(model)
        print("Done")
