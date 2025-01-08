import glob
import os
import stat
import tarfile
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile, copytree, rmtree

import numpy as np
import yaml

from hls4ml.backends import get_backend
from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'


class CatapultWriter(Writer):
    def print_array_to_cpp(self, var, odir, write_txt_file=True):
        """Write a weights array to C++ header files.

        Args:
            var (WeightVariable): Weight to write
            odir (str): Output directory
            write_txt_file (bool, optional): Write txt files in addition to .h files. Defaults to True.
        """

        h_file = open(f"{odir}/firmware/weights/{var.name}.h", "w")
        if write_txt_file:
            txt_file = open(f"{odir}/firmware/weights/{var.name}.txt", "w")

        # meta data
        h_file.write(f"//Numpy array shape {var.shape}\n")
        h_file.write(f"//Min {np.min(var.min):.12f}\n")
        h_file.write(f"//Max {np.max(var.max):.12f}\n")
        h_file.write(f"//Number of zeros {var.nzeros}\n")
        h_file.write("\n")

        h_file.write(f"#ifndef {var.name.upper()}_H_\n")
        h_file.write(f"#define {var.name.upper()}_H_\n")
        h_file.write("\n")

        if write_txt_file:
            h_file.write("#ifndef __SYNTHESIS__\n")
            h_file.write("// global extern pointer only - actual array allocated in myproject_test.cpp\n")
            h_file.write("extern " + var.definition_cpp() + ";\n")
            h_file.write("#else\n")

        h_file.write(var.definition_cpp() + " = {")

        # fill c++ array.
        # not including internal brackets for multidimensional case
        sep = ''
        for x in var:
            h_file.write(sep + x)
            if write_txt_file:
                txt_file.write(sep + x)
            sep = ", "
        h_file.write("};\n")
        if write_txt_file:
            h_file.write("#endif\n")
            txt_file.close()
        h_file.write("\n#endif\n")
        h_file.close()

    def write_output_dir(self, model):
        """Write the base output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f"{model.config.get_output_dir()}/firmware/weights"):
            os.makedirs(f"{model.config.get_output_dir()}/firmware/weights")

    @staticmethod
    def _make_array_pragma(variable, model):
        """
        Layers in hls_model.py can specify output array partitioning through the `pragma` attribute.
        If `pragma` is a string: options are 'partition', 'reshape', or 'stream'.
        If `pragma` is a tuple: (mode, type, factor) where mode is 'partition' or 'reshape', type is
        'complete', 'cyclic', or 'block', and factor is an integer only used when the type is not 'complete'.
        """

        config = variable.pragma
        if type(config) is tuple:
            mode = config[0]
            if mode in ['partition', 'reshape']:
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        if mode in ['partition', 'reshape']:
            if typ == 'complete':
                template = '// #pragma HLS ARRAY_{mode} variable={name} {type} dim={dim}'
            else:
                template = '// #pragma HLS ARRAY_{mode} variable={name} {type} factor={factor} dim={dim}'

            return template.format(mode=mode.upper(), name=variable.name, type=typ, factor=factor, dim=0)

        elif mode == 'stream':
            fifo = model.config.get_config_value("FIFO")
            if fifo is not None:
                retstr = f'#pragma hls_resource {variable.name}:cns variables="{variable.name}"'
                retstr += f' map_to_module="{fifo}" // depth="{depth}"'
                return retstr
            else:
                return ''
        else:
            return ''

    @staticmethod
    def _make_array_fifo_pragma(variable, model):
        config = variable.pragma
        factor = ''
        if type(config) is tuple:
            mode = config[0]
            if mode in ['partition', 'reshape']:
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        if mode == 'stream':
            fifo = model.config.get_config_value("FIFO")
            if fifo is not None:
                return f'// #pragma hls_fifo_depth {depth} {factor}'
            else:
                return ''
        else:
            return ''

    def write_project_cpp(self, model):
        """Write the main architecture source file (myproject.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        fout = open(f'{model.config.get_output_dir()}/firmware/layer_summary.txt', 'w')
        outstr = ""
        outstr = outstr + "{}".format("Layer Name").ljust(25)
        outstr = outstr + "  {}".format("Layer Class").ljust(20)
        outstr = outstr + "  {}".format("Input Type").ljust(40)
        outstr = outstr + "  {}".format("Input Shape").ljust(15)
        outstr = outstr + "  {}".format("Output Type").ljust(40)
        outstr = outstr + "  {}".format("Output Shape").ljust(15)
        # outstr = outstr + "  {}".format("Weight Type").ljust(24)
        # outstr = outstr + "  {}".format("Bias Type").ljust(24)
        outstr = outstr + "  {}".format("Filter Shape").ljust(15)
        outstr = outstr + "  {}".format("Stride").ljust(10)
        outstr = outstr + "  {}".format("IOType").ljust(15)
        outstr = outstr + "  {}".format("Reuse").ljust(10)

        fout.write(outstr + "\n")
        input_shape = ""
        input_datatype = ""
        for layer in model.get_layers():
            datatype = layer.get_output_variable().type.precision.definition_cpp() + " "
            shape = ""
            # layer.get_output_variable().type.precision.width
            # layer.get_output_variable().type.precision.integer
            # layer.get_output_variable().type.precision.sign
            for _k, v in layer.get_output_variable().get_shape():
                shape = shape + "[" + str(v) + "]"

            if layer.attributes.layer.class_name != 'Input':
                my_class_name = layer.class_name
                if layer.attributes.layer.class_name == 'Activation':
                    my_class_name = layer.get_attr('activation')

                # filter_datatype = ""
                # print(layer.weights.__dir__())
                # layer_precision = layer.get_layer_precision()
                # for wname, weights in layer.weights.items():
                #    print(wname)
                #    print(weights.type.name)
                #    print(weights.type.precision.definition_cpp())
                #    #print(weights.type.precision.__dir__())
                #    print(weights.type.precision.width)
                #    if 'ACFixed' in weights.type.precision.__class__:
                #        print(weights.type.precision.integer)
                #        print(weights.type.precision.signed)
                #    print(weights.data_length)

                filter = ""
                filt_width = layer.get_attr('filt_width')
                filt_height = layer.get_attr('filt_height')
                if filt_width is not None:
                    filter = "[" + str(filt_width) + "]"
                if filt_height is not None:
                    filter = filter + "[" + str(filt_height) + "]"

                stride = ""
                stride_width = layer.get_attr('stride_width')
                if stride_width is not None:
                    stride = str(stride_width)

                outstr = ""
                outstr = outstr + f"{layer.name}".ljust(25)
                outstr = outstr + f"  {my_class_name}".ljust(20)
                outstr = outstr + f"  {input_datatype}".ljust(40)
                outstr = outstr + f"  {input_shape}".ljust(15)
                outstr = outstr + f"  {datatype}".ljust(40)
                outstr = outstr + f"  {shape}".ljust(15)
                # outstr = outstr + "  {}".format("weight type").ljust(24)
                # outstr = outstr + "  {}".format("bias type").ljust(24)
                outstr = outstr + f"  {filter}".ljust(15)
                outstr = outstr + f"  {stride}".ljust(10)
                outstr = outstr + "  {}".format(layer.model.config.get_config_value('IOType')).ljust(15)
                outstr = outstr + f"  {str(layer.model.config.get_reuse_factor(layer))}".ljust(10)
                fout.write(outstr + "\n")

            input_shape = shape
            input_datatype = datatype

        fout.close()

        f = open(os.path.join(filedir, '../templates/catapult/firmware/myproject.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([i.definition_cpp(as_reference=True) for i in model_inputs])
                outputs_str = ', '.join([o.definition_cpp(as_reference=True) for o in model_outputs])
                brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str
                if len(model_brams) > 0:
                    newline += ',\n' + brams_str
                newline += '\n'

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.weight_class == 'CompressedWeightVariable':
                            newline += indent + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.nonzeros, w.name, w.name
                            )
                        elif w.weight_class == 'ExponentWeightVariable':
                            newline += indent + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.data_length, w.name, w.name
                            )
                        else:
                            newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.data_length, w.name, w.name
                            )

            # Add Interface Synthesis resource pragmas
            elif '// hls-fpga-machine-learning insert IFSynPragmas' in line:
                newline = line
                all_inputs = [i.name for i in model_inputs]
                all_outputs = [o.name for o in model_outputs]
                all_brams = [b.name for b in model_brams]
                io_type = model.config.get_config_value("IOType")

                if io_type == 'io_serial' or io_type == 'io_stream':
                    # Eventually this will be amba.ccs_axi4stream_in and amba.ccs_axi4stream_out
                    for dut_input in all_inputs:
                        newline += f'#pragma hls_resource {dut_input}:rsc variables="{dut_input}"'
                        newline += ' map_to_module="ccs_ioport.ccs_in_wait"\n'
                    for dut_output in all_outputs:
                        newline += f'#pragma hls_resource {dut_output}:rsc variables="{dut_output}"'
                        newline += ' map_to_module="ccs_ioport.ccs_out_wait"\n'

            # Add input/output type
            elif '// hls-fpga-machine-learning insert IO' in line:
                newline = line
                all_inputs = [i.name for i in model_inputs]
                all_outputs = [o.name for o in model_outputs]
                all_brams = [b.name for b in model_brams]
                io_type = model.config.get_config_value("IOType")

                if io_type == 'io_parallel':
                    for i in model_inputs:
                        newline += indent + self._make_array_pragma(i, model) + '\n'
                    for o in model_outputs:
                        newline += indent + self._make_array_pragma(o, model) + '\n'
                    # TODO discussed adding a handle for setting the interface mode for individual input and output arrays
                    # Probably the handle doesn't need to be exposed to the user but should be just set in hls_model.py
                    newline += indent + '// #pragma HLS INTERFACE ap_vld port={},{} \n'.format(
                        ','.join(all_inputs), ','.join(all_outputs)
                    )
                    if model.config.model_strategy.lower() == 'dataflow':
                        newline += indent + '// #pragma HLS DATAFLOW \n'
                    else:
                        newline += indent + '// #pragma HLS PIPELINE \n'
                if io_type == 'io_stream':
                    newline += indent + '// #pragma HLS INTERFACE axis port={},{} \n'.format(
                        ','.join(all_inputs), ','.join(all_outputs)
                    )
                    if all_brams:
                        newline += indent + '// #pragma HLS INTERFACE bram port={} \n'.format(','.join(all_brams))
                    newline += indent + '// #pragma HLS DATAFLOW \n'

            elif '// hls-fpga-machine-learning insert layers' in line:
                io_type = model.config.get_config_value("IOType")
                newline = line + '\n'
                for layer in model.get_layers():
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in model_inputs and var not in model_outputs:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None:
                                if var.pragma:
                                    newline += '    ' + self._make_array_fifo_pragma(var, model) + '\n'
                                if io_type == 'io_serial' or io_type == 'io_stream':
                                    newline += '    static ' + def_cpp + '; \n'
                                else:
                                    newline += '    ' + def_cpp + '; \n'
                                if var.pragma:
                                    newline += '    ' + self._make_array_pragma(var, model) + '\n'
                    func = layer.get_attr('function_cpp', None)
                    if func:
                        if not isinstance(func, (list, set)):
                            func = [func]
                        if len(func) == 1:
                            newline += '    ' + func[0] + ' // ' + layer.name + '\n'
                        else:
                            newline += '    // ' + layer.name + '\n'
                            for line in func:
                                newline += '    ' + line + '\n'
                        if model.config.trace_output and layer.get_attr('trace', False):
                            newline += '#ifndef __SYNTHESIS__\n'
                            for var in vars:
                                newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(
                                    var.type.name, var.name, layer.name, var.size_cpp()
                                )
                            newline += '#endif\n'
                        newline += '\n'

            # Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_project_header(self, model):
        """Write the main architecture header file (myproject.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/firmware/myproject.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.h', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([i.definition_cpp(as_reference=True) for i in model_inputs])
                outputs_str = ', '.join([o.definition_cpp(as_reference=True) for o in model_outputs])
                brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str
                if len(model_brams) > 0:
                    newline += ',\n' + brams_str
                newline += '\n'
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_defines(self, model):
        """Write the C++ type definitions file (defines.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/firmware/defines.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/defines.h', 'w')

        for line in f.readlines():
            # Insert numbers
            if '// hls-fpga-machine-learning insert numbers' in line:
                newline = line

                defines_list = []
                for layer in model.get_layers():
                    defines = ''
                    for k, v in layer.get_output_variable().get_shape():
                        defines += f'#define {k} {v}\n'

                    defines_list.append(defines)

                newline += ''.join(defines_list)

            elif '// hls-fpga-machine-learning insert layer-precision' in line:
                newline = line
                all_precision = OrderedDict()
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    for type_name, type_var in layer_precision.items():
                        # Ensure that layer's types doesn't override existing types
                        # This can happen in case of InplaceVariable types
                        if type_name not in all_precision:
                            all_precision[type_name] = type_var
                for used_type in all_precision.values():
                    newline += used_type.definition_cpp()

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_parameters(self, model):
        """Write the C++ layer config file (parameters.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/firmware/parameters.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/parameters.h', 'w')

        for line in f.readlines():
            if '// hls-fpga-machine-learning insert includes' in line:
                newline = line
                for include in sorted(set(sum((layer.get_attr('include_header', []) for layer in model.get_layers()), []))):
                    newline += '#include "%s"\n' % include

            elif '// hls-fpga-machine-learning insert weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.storage.lower() != 'bram':
                            newline += f'#include "weights/{w.name}.h"\n'

            elif "// hls-fpga-machine-learning insert layer-config" in line:
                newline = line
                for layer in model.get_layers():
                    config = layer.get_attr('config_cpp', None)
                    if config:
                        newline += '// ' + layer.name + '\n'
                        newline += config + '\n'
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_weights(self, model):
        """Write the weights into header files

        Args:
            model (ModelGraph): the hls4ml model.
        """
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(weights, model.config.get_output_dir())

    def __make_dat_file(self, original_path, project_path):
        """
        Convert other input/output data types into a dat file, which is
        a text file with the falttened matrix printed out. Note that ' ' is
        assumed to be the delimiter.
        """

        # Take in data from current supported data files
        if original_path[-3:] == "npy":
            data = np.load(original_path)
        else:
            raise Exception("Unsupported input/output data files.")

        # Faltten data, just keep first dimension
        data = data.reshape(data.shape[0], -1)

        def print_data(f):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write(str(data[i][j]) + " ")
                f.write("\n")

        # Print out in dat file
        with open(project_path, "w") as f:
            print_data(f)

    def write_test_bench(self, model):
        """Write the testbench files (myproject_test.cpp and input/output .dat files)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists(f'{model.config.get_output_dir()}/tb_data/'):
            os.mkdir(f'{model.config.get_output_dir()}/tb_data/')

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == "dat":
                copyfile(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')
            else:
                self.__make_dat_file(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')

        if output_predictions:
            if output_predictions[-3:] == "dat":
                copyfile(output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat')
            else:
                self.__make_dat_file(
                    output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat'
                )

        f = open(os.path.join(filedir, '../templates/catapult/myproject_test.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            # Insert numbers
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

            elif '// hls-fpga-machine-learning insert declare weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += w.definition_cpp() + ";\n"

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.weight_class == 'CompressedWeightVariable':
                            newline += indent + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.nonzeros, w.name, w.name
                            )
                        elif w.weight_class == 'ExponentWeightVariable':
                            newline += indent + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.data_length, w.name, w.name
                            )
                        else:
                            newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.data_length, w.name, w.name
                            )

            elif '// hls-fpga-machine-learning insert data' in line:
                newline = line
                offset = 0
                for inp in model_inputs:
                    newline += '      ' + inp.definition_cpp() + ';\n'
                    newline += '      nnet::copy_data<float, {}, {}, {}>(in, {});\n'.format(
                        inp.type.name, offset, inp.size_cpp(), inp.name
                    )
                    offset += inp.size()
                for out in model_outputs:
                    newline += '      ' + out.definition_cpp() + ';\n'
            elif '// hls-fpga-machine-learning insert random' in line:
                newline = line
                for inp in model_inputs:
                    newline += '    ' + inp.definition_cpp() + ';\n'
                    newline += f'    nnet::fill_random<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n'
                for out in model_outputs:
                    newline += '    ' + out.definition_cpp() + ';\n'
            elif '// hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inp in model_inputs:
                    newline += '    ' + inp.definition_cpp() + ';\n'
                    newline += f'    nnet::fill_zero<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n'
                for out in model_outputs:
                    newline += '    ' + out.definition_cpp() + ';\n'
            elif '// hls-fpga-machine-learning insert top-level-function' in line:
                newline = line

                input_vars = ','.join([i.name for i in model_inputs])
                output_vars = ','.join([o.name for o in model_outputs])
                bram_vars = ','.join([b.name for b in model_brams])

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))

                top_level = indent + f'{model.config.get_project_name()}({all_vars});\n'

                newline += top_level
            elif '// hls-fpga-machine-learning insert predictions' in line:
                newline = line
                for out in model_outputs:
                    newline += indent + f'for(int i = 0; i < {out.size_cpp()}; i++) {{\n'
                    newline += indent + '  std::cout << pr[i] << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'
            elif '// hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                for out in model_outputs:
                    newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(
                        out.type.name, out.size_cpp(), out.name
                    )  # TODO enable this
            elif (
                '// hls-fpga-machine-learning insert output' in line
                or '// hls-fpga-machine-learning insert quantized' in line
            ):
                newline = line
                for out in model_outputs:
                    newline += indent + 'nnet::print_result<{}, {}>({}, std::cout, true);\n'.format(
                        out.type.name, out.size_cpp(), out.name
                    )
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_bridge(self, model):
        """Write the Python-C++ bridge (myproject_bridge.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/myproject_bridge.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))
            elif '// hls-fpga-machine-learning insert weights dir' in line:
                weights_dir = (Path(fout.name).parent / 'firmware/weights').resolve()
                newline = f'static std::string s_weights_dir = "{weights_dir}";\n'
            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'
            elif '// hls-fpga-machine-learning insert declare weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += w.definition_cpp() + ";\n"
            elif '// hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
                outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + '\n'
            elif '// hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                newline = ''
                for i in model_inputs:
                    newline += indent + '{var};\n'.format(var=i.definition_cpp(name_suffix='_ap'))
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(
                        dtype, i.type.name, i.size_cpp(), i.name, i.name
                    )
                newline += '\n'

                for o in model_outputs:
                    newline += indent + '{var};\n'.format(var=o.definition_cpp(name_suffix='_ap'))

                newline += '\n'

                input_vars = ','.join([i.name + '_ap' for i in model_inputs])
                bram_vars = ','.join([b.name for b in model_brams])
                output_vars = ','.join([o.name + '_ap' for o in model_outputs])

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))

                top_level = indent + f'{model.config.get_project_name()}({all_vars});\n'
                newline += top_level

                newline += '\n'

                for o in model_outputs:
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                        o.type.name, dtype, o.size_cpp(), o.name, o.name
                    )
            elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func and model.config.trace_output and layer.get_attr('trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += (
                                indent
                                + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                            )

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_build_script(self, model):
        """Write the TCL/Shell build scripts (build_prj.tcl, build_lib.sh)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = Path(__file__).parent

        # build_prj.tcl
        srcpath = (filedir / '../templates/catapult/build_prj.tcl').resolve()
        dstpath = Path(f'{model.config.get_output_dir()}/build_prj.tcl').resolve()
        with open(srcpath) as src, open(dstpath, 'w') as dst:
            for line in src.readlines():
                indent = line[: len(line) - len(line.lstrip())]
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('CATAPULT_DIR', model.config.get_project_dir())
                if '#hls-fpga-machine-learning insert techlibs' in line:
                    if model.config.get_config_value('Technology') is None:
                        if model.config.get_config_value('Part') is not None:
                            line = indent + 'setup_xilinx_part {{{}}}\n'.format(model.config.get_config_value('Part'))
                        elif model.config.get_config_value('ASICLibs') is not None:
                            line = indent + 'setup_asic_libs {{{}}}\n'.format(model.config.get_config_value('ASICLibs'))
                    else:
                        if model.config.get_config_value('Technology') == 'asic':
                            line = indent + 'setup_asic_libs {{{}}}\n'.format(model.config.get_config_value('ASICLibs'))
                        else:
                            line = indent + 'setup_xilinx_part {{{}}}\n'.format(model.config.get_config_value('Part'))
                elif '#hls-fpga-machine-learning insert invoke_args' in line:
                    tb_in_file = model.config.get_config_value('InputData')
                    tb_out_file = model.config.get_config_value('OutputPredictions')
                    invoke_args = '$sfd/firmware/weights'
                    if tb_in_file is not None:
                        invoke_args = invoke_args + f' $sfd/tb_data/{tb_in_file}'
                    if tb_out_file is not None:
                        invoke_args = invoke_args + f' $sfd/tb_data/{tb_out_file}'
                    line = indent + f'flow package option set /SCVerify/INVOKE_ARGS "{invoke_args}"\n'
                elif 'set hls_clock_period 5' in line:
                    line = indent + 'set hls_clock_period {}\n'.format(model.config.get_config_value('ClockPeriod'))
                dst.write(line)

        # build_lib.sh
        build_lib_src = (filedir / '../templates/catapult/build_lib.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))

                dst.write(line)
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/catapult/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        if model.config.get_config_value('DontCopyNNET') is not None:
            h = 'nnet_code_gen.h'
            copyfile(srcpath + h, dstpath + h)
            return

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        print("Copying NNET files to local firmware directory")

        filedir = os.path.dirname(os.path.abspath(__file__))
        for pkg in ('ac_types', 'ac_math', 'ac_simutils'):
            dstpath = f'{model.config.get_output_dir()}/firmware/{pkg}/'

            # backward compatibility, look in root dir
            srcpath = os.path.join(filedir, '../../' + pkg + '/')
            if not os.path.exists(srcpath):
                # look next in Catapult-specific templates
                srcpath = os.path.join(filedir, '../templates/catapult/' + pkg + '/')

            if os.path.exists(srcpath):
                if os.path.exists(dstpath):
                    rmtree(dstpath)
                print("... copying AC " + pkg + " headers from " + srcpath)
                copytree(srcpath, dstpath)
            else:
                print("... skipping copy of " + pkg + " headers - assumed to located in Catapult install tree")

        # custom source
        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = get_backend('Catapult').get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = f'{model.config.get_output_dir()}/firmware/{dst}'
            copyfile(srcpath, dstpath)

    def write_generated_code(self, model):
        """Write the generated code (nnet_code_gen.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        path = f'{model.config.get_output_dir()}/firmware/nnet_utils/nnet_code_gen.h'
        f = open(path)
        contents = f.readlines()
        f.close()
        f = open(path, 'w')

        for line in contents:
            if '// hls4ml insert code' in line:
                newline = line
                for layer in model.get_layers():
                    for generated_code in layer.code.values():
                        newline += str(generated_code)
            else:
                newline = line
            f.write(newline)
        f.close()

    def write_yml(self, model):
        """Write the config to the YAML file

        Args:
            model (ModelGraph): the hls4ml model.
        """

        def keras_model_representer(dumper, keras_model):
            model_path = model.config.get_output_dir() + '/keras_model.keras'
            keras_model.save(model_path)
            return dumper.represent_scalar('!keras_model', model_path)

        try:
            import keras

            KerasModel = keras.models.Model

            yaml.add_multi_representer(KerasModel, keras_model_representer)
        except Exception:
            pass

        with open(model.config.get_output_dir() + '/' + config_filename, 'w') as file:
            yaml.dump(model.config.config, file)

    def write_tar(self, model):
        """Write the generated project as a .tar.gz archive

        Args:
            model (ModelGraph): the hls4ml model.
        """

        if not os.path.exists(model.config.get_output_dir() + '.tar.gz'):
            with tarfile.open(model.config.get_output_dir() + '.tar.gz', mode='w:gz') as archive:
                archive.add(model.config.get_output_dir(), recursive=True)
        else:
            print("Project .tar.gz archive already exists")

    def write_hls(self, model):
        print('Writing HLS project')
        self.write_output_dir(model)
        self.write_project_cpp(model)
        self.write_project_header(model)
        self.write_weights(model)
        self.write_defines(model)
        self.write_parameters(model)
        self.write_test_bench(model)
        self.write_bridge(model)
        self.write_build_script(model)
        self.write_nnet_utils(model)
        self.write_generated_code(model)
        self.write_yml(model)
        self.write_tar(model)
        print('Done')
