#!/usr/bin/env python

import argparse
import json
import os
import sys

import h5py
import yaml

import hls4ml

config_filename = 'hls4ml_config.yml'

hls4ml_description = """

        ╔╧╧╧╗────o
    hls ║ 4 ║ ml   - Machine learning inference in FPGAs
   o────╚╤╤╤╝
"""


def main():
    parser = argparse.ArgumentParser(description=hls4ml_description, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers()

    config_parser = subparsers.add_parser('config', help='Create a conversion configuration file')
    convert_parser = subparsers.add_parser('convert', help='Convert Keras or ONNX model to HLS')
    build_parser = subparsers.add_parser('build', help='Build generated HLS project')
    report_parser = subparsers.add_parser('report', help='Show synthesis report of an HLS project')

    config_parser.add_argument(
        '-m',
        '--model',
        help='Model file to convert (Keras .h5 or .json file, ONNX .onnx file, or TensorFlow .pb file)',
        default=None,
    )
    config_parser.add_argument(
        '-w', '--weights', help='Optional weights file (if Keras .json file is provided))', default=None
    )
    config_parser.add_argument('-p', '--project', help='Project name', default='myproject')
    config_parser.add_argument('-d', '--dir', help='Project output directory', default='my-hls-test')
    config_parser.add_argument('-f', '--fpga', help='FPGA part', default='xcvu13p-flga2577-2-e')
    config_parser.add_argument('-bo', '--board', help='Board used.', default='pynq-z2')
    config_parser.add_argument(
        '-ba', '--backend', help='Backend to use (Vivado, VivadoAccelerator, Quartus)', default='Vivado'
    )
    config_parser.add_argument('-c', '--clock', help='Clock frequency (ns)', type=int, default=5)
    config_parser.add_argument(
        '-g', '--granularity', help='Granularity of configuration. One of "model", "type" or "name"', default='model'
    )
    config_parser.add_argument('-x', '--precision', help='Default precision', default='ap_fixed<16,6>')
    config_parser.add_argument('-r', '--reuse-factor', help='Default reuse factor', type=int, default=1)
    config_parser.add_argument('-o', '--output', help='Output file name', default=None)
    config_parser.set_defaults(func=_config)

    convert_parser.add_argument('-c', '--config', help='Configuration file', default=None)
    convert_parser.set_defaults(func=_convert)

    build_parser.add_argument('-p', '--project', help='Project directory', default=None)
    build_parser.add_argument(
        '-l', '--list-options', help='List available build options for a given project', action='store_true', default=False
    )
    build_parser.set_defaults(func=_build)

    report_parser.add_argument('-p', '--project', help='Project directory', default=None)
    report_parser.add_argument(
        '-l', '--list-options', help='List available report options for a given project', action='store_true', default=False
    )
    report_parser.set_defaults(func=_report)

    parser.add_argument('--version', action='version', version=f'%(prog)s {hls4ml.__version__}')

    args, extra_args = parser.parse_known_args()
    if hasattr(args, 'func'):
        args.func(args, extra_args)
    else:
        print(hls4ml_description)
        parser.print_usage()


def _config(args, extra_args):
    if args.model is None:
        print('Model file (-m or --model) must be provided.')
        sys.exit(1)

    config = hls4ml.utils.config.create_config(
        backend=args.backend,
        output_dir=args.dir,
        project_name=args.project,
        part=args.fpga,
        board=args.board,
        clock_period=args.clock,
        write_tar=True,
    )

    if args.model.endswith('.h5'):
        config['KerasH5'] = args.model

        with h5py.File(args.model, mode='r') as h5file:
            # Load the configuration from h5 using json's decode
            model_arch = h5file.attrs.get('model_config')
            if model_arch is None:
                print('No model found in the provided h5 file.')
                sys.exit(1)
            else:
                model_arch = json.loads(model_arch.decode('utf-8'))

            config['HLSConfig'] = hls4ml.utils.config_from_keras_model(
                model_arch,
                granularity=args.granularity,
                default_precision=args.precision,
                default_reuse_factor=args.reuse_factor,
            )
    elif args.model.endswith('.json'):
        if args.weights is None:
            print('Weights file (-w or --weights) must be provided when parsing from JSON file.')
            sys.exit(1)
        config['KerasJson'] = args.model
        config['KerasH5'] = args.weights

        with open(args.model) as json_file:
            model_arch = json.load(json_file)
            config['HLSConfig'] = hls4ml.utils.config_from_keras_model(
                model_arch,
                granularity=args.granularity,
                default_precision=args.precision,
                default_reuse_factor=args.reuse_factor,
            )
    elif args.model.endswith('.onnx'):
        print('Creating configuration for ONNX mdoels is not supported yet.')
        sys.exit(1)
    elif args.model.endswith('.pb'):
        print('Creating configuration for Tensorflow mdoels is not supported yet.')
        sys.exit(1)

    if args.output is not None:
        outname = args.output
        if not outname.endswith('.yml'):
            outname += '.yml'
        print(f'Writing config to {outname}')
        with open(outname, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)
    else:
        print('Config output:')
        yaml.dump(config, sys.stdout, default_flow_style=False, sort_keys=False)


def _convert(args, extra_args):
    model = hls4ml.converters.convert_from_config(args.config)

    if model is not None:
        model.write()


def _build(args, extra_args):
    if args.project is None:
        print('Project directory (-p or --project) must be provided.')
        sys.exit(1)

    try:
        yamlConfig = hls4ml.converters.parse_yaml_config(args.project + '/' + config_filename)
    except Exception:
        print(f'Project configuration file not found in "{args.project}".')
        sys.exit(1)

    backend_map = {}
    backend_map['vivado'] = _build_vivado
    backend_map['quartus'] = _build_quartus

    backend = yamlConfig.get('Backend')

    if backend.lower() in backend_map:
        backend_map[backend.lower()](args, extra_args)
    else:
        print(f'Backend {backend} does not support building projects.')


def _build_vivado(args, extra_args):
    vivado_parser = argparse.ArgumentParser(prog=f'hls4ml build -p {args.project}', add_help=False)
    vivado_parser.add_argument('-c', '--simulation', help='Run C simulation', action='store_true', default=False)
    vivado_parser.add_argument('-s', '--synthesis', help='Run C/RTL synthesis', action='store_true', default=False)
    vivado_parser.add_argument('-r', '--co-simulation', help='Run C/RTL co-simulation', action='store_true', default=False)
    vivado_parser.add_argument('-v', '--validation', help='Run C/RTL validation', action='store_true', default=False)
    vivado_parser.add_argument('-e', '--export', help='Export IP (implies -s)', action='store_true', default=False)
    vivado_parser.add_argument(
        '-l', '--vivado-synthesis', help='Run Vivado synthesis (implies -s)', action='store_true', default=False
    )
    vivado_parser.add_argument(
        '-a',
        '--all',
        help='Run C simulation, C/RTL synthesis, C/RTL co-simulation and Vivado synthesis',
        action='store_true',
    )
    vivado_parser.add_argument('--reset', help='Remove any previous builds', action='store_true', default=False)

    if args.list_options:
        vivado_parser.print_help()
        sys.exit(0)

    vivado_args = vivado_parser.parse_args(extra_args)

    reset = int(vivado_args.reset)
    csim = int(vivado_args.simulation)
    synth = int(vivado_args.synthesis)
    cosim = int(vivado_args.co_simulation)
    validation = int(vivado_args.validation)
    export = int(vivado_args.export)
    vsynth = int(vivado_args.vivado_synthesis)
    if vivado_args.all:
        csim = synth = cosim = validation = export = vsynth = 1

    # Check if vivado_hls is available
    if 'linux' in sys.platform or 'darwin' in sys.platform:
        found = os.system('command -v vivado_hls > /dev/null')
        if found != 0:
            print('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')
            sys.exit(1)

    os.system(
        (
            'cd {dir} && vivado_hls -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} '
            'validation={validation} export={export} vsynth={vsynth}"'
        ).format(
            dir=args.project,
            reset=reset,
            csim=csim,
            synth=synth,
            cosim=cosim,
            validation=validation,
            export=export,
            vsynth=vsynth,
        )
    )


def _build_quartus(args, extra_args):
    quartus_parser = argparse.ArgumentParser(prog=f'hls4ml build -p {args.project}', add_help=False)
    quartus_parser.add_argument(
        '-s', '--synthesis', help='Compile project and run C/RTL synthesis', action='store_true', default=False
    )
    quartus_parser.add_argument(
        '-q', '--quartus-synthesis', help='Run Quartus synthesis (implies -s)', action='store_true', default=False
    )
    quartus_parser.add_argument(
        '-a', '--all', help='Run C simulation, C/RTL synthesis, Quartus synthesis', action='store_true'
    )

    if args.list_options:
        quartus_parser.print_help()
        sys.exit(0)

    quartus_args = quartus_parser.parse_args(extra_args)

    synth = int(quartus_args.synthesis)
    qsynth = int(quartus_args.quartus_synthesis)
    if quartus_args.all:
        synth = qsynth = 1

    yamlConfig = hls4ml.converters.parse_yaml_config(args.project + '/' + config_filename)
    project_name = yamlConfig['ProjectName']

    curr_dir = os.getcwd()

    os.chdir(yamlConfig['OutputDir'])
    if synth:
        os.system(f'make {project_name}-fpga')
        os.system(f'./{project_name}-fpga')

    if qsynth:
        found = os.system('command -v quartus_sh > /dev/null')
        if found != 0:
            print('Quartus installation not found. Make sure "quartus_sh" is on PATH.')
            sys.exit(1)
        os.chdir(project_name + '-fpga.prj/quartus')
        os.system('quartus_sh --flow compile quartus_compile')

    os.chdir(curr_dir)


def _report(args, extra_args):
    if args.project is None:
        print('Project directory (-p or --project) must be provided.')
        sys.exit(1)

    try:
        yamlConfig = hls4ml.converters.parse_yaml_config(args.project + '/' + config_filename)
    except Exception:
        print(f'Project configuration file not found in "{args.project}".')
        sys.exit(1)

    backend_map = {}
    backend_map['vivado'] = _report_vivado
    backend_map['quartus'] = _report_quartus

    backend = yamlConfig.get('Backend')

    if backend.lower() in backend_map:
        backend_map[backend.lower()](args, extra_args)
    else:
        print(f'Backend {backend} does not support reading reports.')


def _report_vivado(args, extra_args):
    vivado_parser = argparse.ArgumentParser(prog=f'hls4ml report -p {args.project}', add_help=False)
    vivado_parser.add_argument('-f', '--full', help='Show full report', action='store_true', default=False)

    if args.list_options:
        vivado_parser.print_help()
    else:
        vivado_args = vivado_parser.parse_args(extra_args)
        hls4ml.report.read_vivado_report(args.project, vivado_args.full)


def _report_quartus(args, extra_args):
    quartus_parser = argparse.ArgumentParser(prog=f'hls4ml report -p {args.project}', add_help=False)
    quartus_parser.add_argument(
        '-b', '--open-browser', help='Open a web browser with the report', action='store_true', default=False
    )

    if args.list_options:
        quartus_parser.print_help()
    else:
        quartus_args = quartus_parser.parse_args(extra_args)
        hls4ml.report.read_quartus_report(args.project, quartus_args.open_browser)


if __name__ == "__main__":
    main()
