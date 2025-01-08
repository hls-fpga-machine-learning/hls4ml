import glob
import os
import tarfile
from collections import OrderedDict
from shutil import copyfile

import numpy as np
import yaml

from hls4ml.backends import get_backend
from hls4ml.utils.fixed_point_utils import FixedPointEmulator, ceil_log2, uint_to_binary
from hls4ml.utils.string_utils import convert_to_pascal_case
from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'


class OneAPIWriter(Writer):

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

    def get_max_reuse_factor(self, model):
        max_rf = 0
        for layer in model.get_layers():
            rf = int(layer.get_attr('reuse_factor'))
            if rf > max_rf:
                max_rf = rf
        return max_rf

    def print_array_to_cpp(self, var, layer, odir):
        """Write a weights array to C++ header files.

        Args:
            var (WeightVariable): Weight to write
            layer (Layer): Instance of the layer to which the weights belong
            odir (str): Output directory
        """
        with open(f"{odir}/src/firmware/weights/{var.name}.h", "w") as h_file:
            # meta data
            h_file.write(f"//Numpy array shape {var.shape}\n")
            h_file.write(f"//Min {np.min(var.min):.12f}\n")
            h_file.write(f"//Max {np.max(var.max):.12f}\n")
            h_file.write(f"//Number of zeros {var.nzeros}\n")
            h_file.write("\n")

            h_file.write(f"#ifndef {var.name.upper()}_H_\n")
            h_file.write(f"#define {var.name.upper()}_H_\n")
            h_file.write("\n")

            rf = int(layer.get_attr('reuse_factor', 1))

            h_file.write(var.definition_cpp(rf) + " = {{")

            # fill c++ array.
            # not including internal brackets for multidimensional case
            sep = ''
            for x in var:
                h_file.write(sep + x)
                sep = ", "
            h_file.write("}};\n")
            h_file.write("\n#endif\n")

    def write_project_dir(self, model):
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f"{model.config.get_output_dir()}/src/firmware/weights"):
            os.makedirs(f"{model.config.get_output_dir()}/src/firmware/weights")

    def write_project_cpp(self, model):
        """Write the main architecture source file (myproject.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        project_name = model.config.get_project_name()

        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/oneapi/firmware/myproject.cpp')) as f,
            open(f'{model.config.get_output_dir()}/src/firmware/{project_name}.cpp', 'w') as fout,
        ):
            model_inputs = model.get_input_variables()
            model_outputs = model.get_output_variables()
            model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

            if len(model_brams) != 0:
                raise NotImplementedError("Weights on the interface is currently not supported")

            io_type = model.config.get_config_value('IOType')
            indent = '    '

            for line in f.readlines():
                # Add headers to weights and biases
                if 'myproject' in line:
                    newline = line.replace('myproject', project_name)
                elif 'MyProject' in line:
                    newline = line.replace('MyProject', convert_to_pascal_case(project_name))

                # oneAPI pipes need to be declared and passed as template parameters
                elif '// hls-fpga-machine-learning insert inter-task pipes' in line:
                    newline = line
                    if io_type == 'io_stream':
                        for layer in model.get_layers():
                            vars = layer.get_variables()
                            for var in vars:
                                if var not in model_inputs and var not in model_outputs:
                                    newline += var.declare_cpp()

                # Read in inputs
                elif '// hls-fpga-machine-learning read in' in line:
                    newline = line
                    if io_type == 'io_parallel':
                        for inp in model_inputs:
                            newline += indent + f'auto {inp.name} = {inp.pipe_name}::read();\n'
                    # for streaming we don't need to read it in

                # Insert weights
                elif '// hls-fpga-machine-learning insert weights' in line:
                    newline = line
                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            if w not in model_brams:
                                newline += f'#include "weights/{w.name}.h"\n'

                # Insert task sequences
                elif '// hls-fpga-machine-learning declare task sequences' in line:
                    newline = line
                    if io_type == 'io_stream':  # only need this for io_stream
                        for layer in model.get_layers():
                            ts = layer.get_attr('tast_sequence_cpp')
                            if ts:
                                newline += '    ' + ts + '\n'

                # Neural net instantiation
                elif '// hls-fpga-machine-learning insert layers' in line:
                    newline = line + '\n'
                    for layer in model.get_layers():
                        if io_type != 'io_stream':
                            vars = layer.get_variables()
                            for var in vars:
                                if var not in model_inputs:
                                    def_cpp = var.definition_cpp()
                                    if def_cpp is not None:
                                        newline += '    ' + def_cpp + ';\n'
                        func = (
                            layer.get_attr('function_cpp')
                            if io_type == 'io_parallel'
                            else layer.get_attr('stream_function_cpp')
                        )
                        if func:
                            newline += '    ' + func + '\n'
                            if model.config.trace_output and layer.get_attr('trace', False):
                                newline += '#ifndef HLS_SYNTHESIS\n'
                                for var in vars:
                                    newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(
                                        var.type.name, var.name, layer.name, var.size_cpp()
                                    )
                                newline += '#endif\n'

                # Write the output
                elif '// hls-fpga-machine-learning return' in line:
                    newline = line
                    if io_type == 'io_parallel':
                        for out in model_outputs:
                            newline += indent + f'{out.pipe_name}::write({out.name});\n'
                    # don't need to add anything in io_stream

                # Just copy line
                else:
                    newline = line

                fout.write(newline)

    def write_project_header(self, model):
        """Write the main architecture header file (myproject.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        project_name = model.config.get_project_name()

        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/oneapi/firmware/myproject.h')) as f,
            open(f'{model.config.get_output_dir()}/src/firmware/{project_name}.h', 'w') as fout,
        ):
            model_inputs = model.get_input_variables()
            model_outputs = model.get_output_variables()
            # model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

            # io_parallel and io_stream instantiate the top-level function differently (io_stream not yet supported)
            # io_type = model.config.get_config_value('IOType')
            # indent = '    '
            # brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

            for line in f.readlines():
                if 'MYPROJECT' in line:
                    newline = line.replace('MYPROJECT', format(project_name.upper()))

                elif 'myproject' in line:
                    newline = line.replace('myproject', project_name)

                elif 'MyProject' in line:
                    newline = line.replace('MyProject', convert_to_pascal_case(project_name))

                # Declarations for the inputs. May need modification when io_stream is supported
                elif '// hls-fpga-machine-learning insert inputs' in line:
                    newline = line
                    for inp in model_inputs:
                        newline += inp.declare_cpp()

                # and declareations for the outputs
                elif '// hls-fpga-machine-learning insert outputs' in line:
                    newline = line
                    for out in model_outputs:
                        newline += out.declare_cpp()

                # Simply copy line, if no inserts are required
                else:
                    newline = line

                fout.write(newline)

    def write_defines(self, model):
        """Write the C++ type definitions file (defines.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/oneapi/firmware/defines.h')) as f,
            open(f'{model.config.get_output_dir()}/src/firmware/defines.h', 'w') as fout,
        ):
            for line in f.readlines():
                # Insert numbers
                if '// hls-fpga-machine-learning insert numbers' in line:
                    newline = line

                    defines_list = []
                    for layer in model.get_layers():
                        defines = ''
                        # Note: this assumes all the layers have one ouput
                        # (or in clones, one type of output)
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

    def write_parameters(self, model):
        """Write the C++ layer config file (parameters.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/oneapi/firmware/parameters.h')) as f,
            open(f'{model.config.get_output_dir()}/src/firmware/parameters.h', 'w') as fout,
        ):
            for line in f.readlines():
                if '// hls-fpga-machine-learning insert includes' in line:
                    newline = line
                    for include in sorted(
                        set(sum((layer.get_attr('include_header', []) for layer in model.get_layers()), []))
                    ):
                        newline += '#include "%s"\n' % include

                elif "// hls-fpga-machine-learning insert layer-config" in line:
                    newline = line
                    for layer in model.get_layers():
                        config = layer.get_attr('config_cpp', None)
                        if config:
                            newline += config + '\n'
                else:
                    newline = line
                fout.write(newline)

    def write_weights(self, model):
        """Write the weights into header files

        Args:
            model (ModelGraph): the hls4ml model.
        """
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(weights, layer, model.config.get_output_dir())

    def write_test_bench(self, model):
        """Write the testbench

        Args:
            model (ModelGraph): the hls4ml model.
        """
        # TODO - This function only works with one model input
        # (NOT one data point - it works as expected with multiple data points)

        # copy the exception handler
        filedir = os.path.dirname(os.path.abspath(__file__))
        srcpath = os.path.join(filedir, '../templates/oneapi/exception_handler.hpp')
        dstpath = f'{model.config.get_output_dir()}/src/exception_handler.hpp'
        copyfile(srcpath, dstpath)

        project_name = model.config.get_project_name()
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        if len(model_brams) != 0:
            raise NotImplementedError("Weights on the interface is currently not supported")

        if len(model_inputs) != 1 or len(model_outputs) != 1:
            print("The testbench supports only single input arrays and single output arrays.")
            print("Please modify it before using it.")

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

        with (
            open(os.path.join(filedir, '../templates/oneapi/myproject_test.cpp')) as f,
            open(f'{model.config.get_output_dir()}/src/{project_name}_test.cpp', 'w') as fout,
        ):
            for line in f.readlines():
                indent = ' ' * (len(line) - len(line.lstrip(' ')))

                if 'myproject' in line:
                    newline = line.replace('myproject', project_name)
                elif 'MyProject' in line:
                    newline = line.replace('MyProject', convert_to_pascal_case(project_name))

                elif '// hls-fpga-machine-learning insert bram' in line:
                    newline = line
                    for bram in model_brams:
                        newline += f'#include \"firmware/weights/{bram.name}.h\"\n'
                elif '// hls-fpga-machine-learning insert zero' in line:
                    newline = line
                    inp = model_inputs[0]
                    newline += indent + f'float vals[{inp.size_cpp()}]; \n'
                    newline += indent + f'for (int j = 0 ; j < {inp.size_cpp()} ; j++) {{\n'
                    newline += indent + '    vals[j] = 0.0; \n'
                    newline += indent + '}\n'
                    newline += indent + f'nnet::convert_data<float, {inp.pipe_name}, {inp.size_cpp()}>(q, vals);\n'
                elif '// hls-fpga-machine-learning insert data' in line:
                    newline = line
                    inp = model_inputs[0]
                    newline += indent + f'float vals[{inp.size_cpp()}]; \n'
                    newline += indent + f'for (int j = 0 ; j < {inp.size_cpp()} ; j++) {{\n'
                    newline += indent + '    vals[j] = in[j]; \n'
                    newline += indent + '}\n'
                    newline += indent + f'nnet::convert_data<float, {inp.pipe_name}, {inp.size_cpp()}>(q, vals);\n'
                elif '// hls-fpga-machine-learning convert output' in line:
                    newline = line
                    out = model_outputs[0]
                    newline += indent + f'float outputs[{out.size_cpp()}];\n'
                    newline += indent + f'nnet::convert_data_back<{out.pipe_name}, float, {out.size_cpp()}>(q, outputs);\n'
                else:
                    newline = line

                fout.write(newline)

    def write_bridge(self, model):
        """Write the Python-C++ bridge (myproject_bridge.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        project_name = model.config.get_project_name()
        stamp = model.config.get_config_value('Stamp')
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']
        # model brambs aren't actually supported yet

        # io_type = model.config.get_config_value('IOType')
        indent = '    '

        filedir = os.path.dirname(os.path.abspath(__file__))
        with (
            open(os.path.join(filedir, '../templates/oneapi/myproject_bridge.cpp')) as f,
            open(f'{model.config.get_output_dir()}/src/{project_name}_bridge.cpp', 'w') as fout,
        ):
            for line in f.readlines():
                if 'MYPROJECT' in line:
                    newline = line.replace('MYPROJECT', format(project_name.upper()))

                elif 'myproject' in line:
                    newline = line.replace('myproject', format(project_name))

                elif 'MyProject' in line:
                    newline = line.replace('MyProject', convert_to_pascal_case(project_name))

                elif '// hls-fpga-machine-learning insert bram' in line:
                    newline = line
                    for bram in model_brams:
                        newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

                elif '// hls-fpga-machine-learning insert class def' in line:
                    dtype = line.split('#', 1)[1].strip()
                    newline = f'class {convert_to_pascal_case(project_name)}Class{dtype.capitalize()}_{stamp};\n'

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
                        newline += indent + f'nnet::convert_data<{dtype}, {i.pipe_name}, {i.size_cpp()}>(q, {i.name});\n'

                    newline += (
                        indent
                        + f'q.single_task<{convert_to_pascal_case(project_name)}Class{dtype.capitalize()}_{stamp}>'
                        + f'({convert_to_pascal_case(project_name)}{{}});\n'
                    )

                    for o in model_outputs:
                        newline += (
                            indent + f'nnet::convert_data_back<{o.pipe_name}, {dtype}, {o.size_cpp()}>(q, {o.name});\n'
                        )
                    newline += '\n'
                    newline += indent + 'q.wait();\n'

                elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                    newline = ''
                    for layer in model.get_layers():
                        func = layer.get_attr('function_cpp')
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

    def write_build_script(self, model):
        """Write the build scripts (Makefile, build_lib.sh)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # Makefile
        filedir = os.path.dirname(os.path.abspath(__file__))
        device = model.config.get_config_value('Part')
        with (
            open(os.path.join(filedir, '../templates/oneapi/CMakeLists.txt')) as f,
            open(f'{model.config.get_output_dir()}/CMakeLists.txt', 'w') as fout,
        ):
            for line in f.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))

                if 'set(FPGA_DEVICE' in line:
                    line = f'    set(FPGA_DEVICE "{device}")\n'

                fout.write(line)

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/oneapi/firmware/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/src/firmware/nnet_utils/'

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        # custom source
        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = get_backend('oneAPI').get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = f'{model.config.get_output_dir()}/src/firmware/{dst}'
            copyfile(srcpath, dstpath)

    def __get_table_size(self, model, activation):
        for layer in model.get_layers():
            if (
                layer.get_attr('activation') == activation or layer.get_attr('recurrent_activation') == activation
            ) and layer.get_attr('table_size') is not None:
                return int(layer.get_attr('table_size'))
        return 1024

    def __get_table_header(self, table_name, table_size):
        table_header = f'static const typename CONFIG_T::table_t {table_name}[{table_size}] = {{'
        return table_header

    def __write_elu_table(self, model, path):
        table_name = 'elu_table'
        table_size = self.__get_table_size(model, 'elu')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = -8.0 * i / float(table_size)
            real_val = np.exp(in_val) - 1.0
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_sigmoid_table(self, model, path):
        MAX_VALUE = 8
        MIN_VALUE = 0

        table_name = 'sigmoid_table'
        table_size = self.__get_table_size(model, 'sigmoid')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(int(table_size)):
            in_val = (
                i * (MAX_VALUE - MIN_VALUE) / float(table_size)
                + (MAX_VALUE - MIN_VALUE) / (float(table_size) * 2)
                + MIN_VALUE
            )
            real_val = 1.0 / (1 + np.exp(-in_val))
            if real_val >= 0.5:
                h_file.write(sep + str(real_val))
                sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_tanh_table(self, model, path):
        MAX_VALUE = 4
        MIN_VALUE = 0

        table_name = 'tanh_table'
        table_size = self.__get_table_size(model, 'tanh')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = (
                i * (MAX_VALUE - MIN_VALUE) / float(table_size)
                + (MAX_VALUE - MIN_VALUE) / (float(table_size) * 2)
                + MIN_VALUE
            )
            real_val = np.tanh(in_val)
            if real_val >= 0:
                h_file.write(sep + str(real_val))
                sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_softplus_table(self, model, path):
        table_name = 'softplus_table'
        table_size = self.__get_table_size(model, 'softplus')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = 2 * 8.0 * (i - float(table_size) / 2.0) / float(table_size)
            real_val = np.log(np.exp(in_val) + 1.0)
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_softsign_table(self, model, path):
        MAX_VALUE = 8
        MIN_VALUE = 0
        table_name = 'softsign_table'
        table_size = self.__get_table_size(model, 'softsign')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = (
                i * (MAX_VALUE - MIN_VALUE) / float(table_size)
                + (MAX_VALUE - MIN_VALUE) / (float(table_size) * 2)
                + MIN_VALUE
            )

            real_val = in_val / (np.fabs(in_val) + 1.0)
            if real_val >= 0:
                h_file.write(sep + str(real_val))
                sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_selu_table(self, model, path):
        table_name = 'selu_table'
        table_size = self.__get_table_size(model, 'selu')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = -8.0 * i / float(table_size)
            real_val = 1.0507009873554804934193349852946 * (1.6732632423543772848170429916717 * (np.exp(in_val) - 1.0))
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_exp_table(self, model, path):
        table_name = 'exp_table'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        # Default fixed point precision
        # 6 bits for integer part, 10 bits for decimal - total, 16
        fp_bits = 16
        fp_integer = 6
        fp_signed = True

        # Exp table should use the same precision as exp_table, as seen in Vivado code
        # init_exp_table<data_T, CONFIG_T>(exp_table);
        for layer in model.get_layers():
            if layer.name == 'softmax':
                ac_type = layer.get_input_variable().type
                if ac_type is not None:
                    try:
                        fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                        fp_integer = ac_type.precision.integer
                        fp_signed = ac_type.precision.signed
                    except Exception:
                        # FixedPrecisionType wasn't correctly stored in layer attributes, use default values
                        pass
                    if fp_signed is False:
                        raise Exception('Softmax types need to be signed')

        sep = ''
        N = ceil_log2(table_size)
        for i in range(table_size):
            f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
            b = uint_to_binary(i, N)
            if i == 0:
                b.insert(0, 0)
            else:
                b.insert(0, 1)
            f.set_msb_bits(b)
            real_val = f.exp_float()
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_invert_table(self, model, path):
        table_name = 'invert_table'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        # Default fixed point precision, in case values from layer attributes cannot be extracted
        # 8 bits for integer part, 10 bits for decimal - total, 18
        fp_bits = 18
        fp_integer = 8
        fp_signed = True

        # Invert table should use the same precision as exp_table, as seen in Vivado code
        # init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        for layer in model.get_layers():
            if layer.name == 'softmax':
                ac_type = layer.get_attr('exp_table_t')
                if ac_type is not None:
                    try:
                        fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                        fp_integer = ac_type.precision.integer
                        fp_signed = ac_type.precision.signed
                    except Exception:
                        # FixedPrecisionType wasn't correctly stored in layer attributes, use default values
                        pass
                    if fp_signed is False:
                        raise Exception('Softmax types need to be signed')

        sep = ''
        N = ceil_log2(table_size)
        for i in range(table_size):
            f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
            b = uint_to_binary(i, N)
            b.insert(0, 0)
            f.set_msb_bits(b)
            real_val = f.inv_float()
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_exp_table_latency(self, model, path):
        table_name = 'exp_table_latency'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        # Default fixed point precision
        # 6 bits for integer part, 10 bits for decimal - total, 16
        fp_bits = 16
        fp_integer = 6
        fp_signed = True

        # Exp table should use the same precision as exp_table, as seen in Vivado code
        # init_exp_table<data_T, CONFIG_T>(exp_table);
        for layer in model.get_layers():
            if layer.name == 'softmax':
                ac_type = layer.get_input_variable().type
                if ac_type is not None:
                    try:
                        fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                        fp_integer = ac_type.precision.integer
                        fp_signed = ac_type.precision.signed
                    except Exception:
                        # FixedPrecisionType wasn't correctly stored in layer attributes, use default values
                        pass

        sep = ''
        N = ceil_log2(table_size)
        for i in range(table_size):
            f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
            f.set_msb_bits(uint_to_binary(i, N))
            real_val = f.exp_float()
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_invert_table_latency(self, model, path):
        table_name = 'invert_table_latency'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        # Default fixed point precision, in case values from layer attributes cannot be extracted
        # 8 bits for integer part, 10 bits for decimal - total, 18
        fp_bits = 18
        fp_integer = 8
        fp_signed = True

        # Invert table should use the same precision as exp_table, as seen in Vivado code
        # init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        for layer in model.get_layers():
            if layer.name == 'softmax':
                ac_type = layer.get_attr('exp_table_t')
                if ac_type is not None:
                    try:
                        fp_bits = ac_type.precision.integer + ac_type.precision.fractional
                        fp_integer = ac_type.precision.integer
                        fp_signed = ac_type.precision.signed
                    except Exception:
                        # FixedPrecisionType wasn't correctly stored in layer attributes, use default values
                        pass

        sep = ''
        N = ceil_log2(table_size)
        for i in range(table_size):
            f = FixedPointEmulator(fp_bits, fp_integer, signed=fp_signed)
            f.set_msb_bits(uint_to_binary(i, N))
            real_val = f.inv_float()
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_exp_table_legacy(self, model, path):
        table_name = 'exp_table_legacy'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            in_val = 2 * 8.0 * (i - float(table_size) / 2.0) / float(table_size)
            real_val = np.exp(in_val)
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def __write_invert_table_legacy(self, model, path):
        table_name = 'invert_table_legacy'
        table_size = self.__get_table_size(model, 'softmax')

        h_file = open(f'{path}/{table_name}.tb', 'w')
        h_file.write(self.__get_table_header(table_name, table_size))

        sep = ''
        for i in range(table_size):
            real_val = 0
            in_val = 64.0 * i / float(table_size)
            if in_val > 0.0:
                real_val = 1.0 / in_val
            h_file.write(sep + str(real_val))
            sep = ", "

        h_file.write('};\n')
        h_file.close()

    def write_activation_tables(self, model):
        """Write the lookup tables for activation functions

        Args:
            model (ModelGraph): the hls4ml model.
        """
        # Output path
        dstpath = f'{model.config.get_output_dir()}/src/firmware/nnet_utils/activation_tables'
        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        # Tables
        # TODO - Only write tables needed by model, not all of them
        self.__write_elu_table(model, dstpath)
        self.__write_sigmoid_table(model, dstpath)
        self.__write_tanh_table(model, dstpath)
        self.__write_softplus_table(model, dstpath)
        self.__write_softsign_table(model, dstpath)
        self.__write_selu_table(model, dstpath)
        self.__write_exp_table(model, dstpath)
        self.__write_invert_table(model, dstpath)
        self.__write_exp_table_latency(model, dstpath)
        self.__write_invert_table_latency(model, dstpath)
        self.__write_exp_table_legacy(model, dstpath)
        self.__write_invert_table_legacy(model, dstpath)

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
            from tensorflow.keras import Model as KerasModel

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

        with tarfile.open(model.config.get_output_dir() + '.tar.gz', mode='w:gz') as archive:
            archive.add(model.config.get_output_dir(), recursive=True)

    def write_hls(self, model):
        print('Writing HLS project')
        self.write_project_dir(model)
        self.write_project_cpp(model)
        self.write_project_header(model)
        self.write_weights(model)
        self.write_defines(model)
        self.write_parameters(model)
        self.write_test_bench(model)
        self.write_bridge(model)
        self.write_build_script(model)
        self.write_nnet_utils(model)
        self.write_activation_tables(model)
        self.write_yml(model)
        self.write_tar(model)
        print('Done')
