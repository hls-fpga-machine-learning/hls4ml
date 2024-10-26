import os
import webbrowser
from ast import literal_eval


def parse_quartus_report(hls_dir, write_to_file=True):
    '''
    Parse a report from a given Quartus project as a dictionary.

    Args:
        hls_dir (string): The directory where the project is found
        write_to_file (bool): A flag indicating whether to write the results to a separate file

    Returns:
        results (dict): The report dictionary, containing latency, resource usage etc.

    '''
    if not os.path.exists(hls_dir):
        print(f'Path {hls_dir} does not exist. Exiting.')
        return

    prj_dir = _find_project_dir(hls_dir)

    rpt_dir = hls_dir + '/' + prj_dir + '/reports'
    if not os.path.exists(rpt_dir):
        print(f'Project {prj_dir} does not exist. Rerun "hls4ml build -p {hls_dir}".')
        return

    results = _find_reports(rpt_dir)
    print(results)
    if write_to_file:
        print("Here")
        f = open(hls_dir + '/' 'synthesis-report.txt', 'w')
        f.write('HLS Synthesis Latency & Resource Usage Report')
        for key in results:
            f.write(str(key) + ':' + str(results[key]) + '\n')
        print("There")
        print(f'Saved latency & resource usage summary to {hls_dir}/synthesis-report.txt')
    return results


def read_quartus_report(hls_dir, open_browser=False):
    '''
    Parse and print the Quartus report to print the report. Optionally open a browser.

    Args:
        hls_dir (string):  The directory where the project is found
        open_browser, optional:  whether to open a browser

    Returns:
        None
    '''
    from tabulate import tabulate

    report = parse_quartus_report(hls_dir)

    print('HLS Resource Summary\n')
    print(tabulate(list(report.items())[0:10], tablefmt='orgtbl', headers=['Resource', 'Utilization']))
    print('\n\nHLS Validation Summary\n')
    print(tabulate(list(report.items())[11:13], tablefmt='orgtbl', headers=['', '[Min, Max, Avg]']))
    if 'Clock' in report.keys():
        print('\n\nQuartus Synthesis Summary\n')
        print(tabulate(list(report.items())[13:], tablefmt='orgtbl', headers=['Resource', 'Utilization']))
    else:
        print('Quartus compile data not found! To generate data run FPGA synthesis')

    if open_browser:
        url = 'file:' + os.getcwd() + '/' + _find_project_dir(hls_dir) + '/report.html'
        webbrowser.open(url)


def _find_project_dir(hls_dir):
    '''
    Finds the synthesis folder from the HLS project directory

    Args:
        hls_dir (string): HLS project location

    Returns:
        project_dir (string): Synthesis folder within HLS project directory
    '''
    top_func_name = None

    with open(hls_dir + '/build_lib.sh') as f:
        for line in f.readlines():
            if 'PROJECT=' in line:
                top_func_name = line.split(sep='=')[-1].rstrip()

    return top_func_name + '-fpga.prj'


def read_js_object(js_script):
    '''
    Reads the JavaScript file and return a dictionary of variables definded in the script.

    Args:
        js_script (string) - path to JavaScript File

    Returns:
        Dictionary of variables defines in script
    '''
    from calmjs.parse import asttypes, es5

    def visit(node):
        if isinstance(node, asttypes.Program):
            d = {}
            for child in node:
                if not isinstance(child, asttypes.VarStatement):
                    raise ValueError("All statements should be var statements")
                key, val = visit(child)
                d[key] = val
            return d
        elif isinstance(node, asttypes.VarStatement):
            return visit(node.children()[0])
        elif isinstance(node, asttypes.VarDecl):
            return (visit(node.identifier), visit(node.initializer))
        elif isinstance(node, asttypes.Object):
            d = {}
            for property in node:
                key = visit(property.left)
                value = visit(property.right)
                d[key] = value
            return d
        elif isinstance(node, asttypes.BinOp):
            # simple constant folding
            if node.op == '+':
                if isinstance(node.left, asttypes.String) and isinstance(node.right, asttypes.String):
                    return visit(node.left) + visit(node.right)
                elif isinstance(node.left, asttypes.Number) and isinstance(node.right, asttypes.Number):
                    return visit(node.left) + visit(node.right)
                else:
                    raise ValueError("Cannot + on anything other than two literals")
            else:
                raise ValueError("Cannot do operator '%s'" % node.op)

        elif isinstance(node, asttypes.String) or isinstance(node, asttypes.Number):
            return literal_eval(node.value)
        elif isinstance(node, asttypes.Array):
            return [visit(x) for x in node]
        elif isinstance(node, asttypes.Null):
            return None
        elif isinstance(node, asttypes.Boolean):
            if str(node) == "false":
                return False
            else:
                return True
        elif isinstance(node, asttypes.Identifier):
            return node.value
        else:
            raise Exception("Unhandled node: %r" % node)

    return visit(es5(js_script))


def _read_quartus_file(filename):
    '''
    Reads results (clock frequency, resource usage) obtained through FPGA synthesis (full Quartus compilation)

    Args:
        filename (string): Location of Quartus report

    Returns:
        results (dict): Resource usage obtained through Quartus Compile
    '''

    with open(filename) as dataFile:
        quartus_data = dataFile.read()
        quartus_data = read_js_object(quartus_data)

    results = {}
    if quartus_data['quartusJSON']['quartusFitClockSummary']['nodes'][0]['clock'] != "TBD":
        results['Clock'] = quartus_data['quartusJSON']['quartusFitClockSummary']['nodes'][0]['clock']
        results['Quartus ALM'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['alm']
        results['Quartus REG'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['reg']
        results['Quartus DSP'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['dsp']
        results['Quartus RAM'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['ram']
        results['Quartus MLAB'] = quartus_data['quartusJSON']['quartusFitResourceUsageSummary']['nodes'][-1]['mlab']
    else:
        print(
            'Quartus report not found. '
            'Run Quartus Compilation using Quartus Shell or Full Compilation from Intel Quartus Prime'
        )
    return results


def _read_hls_file(filename):
    '''
    Reads HLS resource estimate obtained through HLS synthesis

    Args:
        filename (string):  Location of HLS report

    Returns:
        results (dict): Resource usage obtained through HLS Estimation
    '''
    with open(filename) as dataFile:
        report_data = dataFile.read()
        report_data = report_data[: report_data.rfind('var fileJSON')]
        report_data = read_js_object(report_data)
        results = {}
        (
            results['HLS Estimate ALUT'],
            results['HLS Estimate FF'],
            results['HLS Estimate RAM'],
            results['HLS Estimate DSP'],
            results['HLS Estimate MLAB'],
        ) = report_data['areaJSON']['total']
        (
            results['HLS Estimate ALUT (%)'],
            results['HLS Estimate FF(%)'],
            results['HLS Estimate RAM (%)'],
            results['HLS Estimate DSP (%)'],
            results['HLS Estimate MLAB (%)'],
        ) = report_data['areaJSON']['total_percent']
        return results


def _read_verification_file(filename):
    '''
    Reads verification data (latency, initiation interval) obtained through simulation

    Args:
        filename (string):  Location of verification file

    Returns:
        results (dict): Verification data obtained from simulation
    '''
    results = {}
    if os.path.isfile(filename):
        with open(filename) as dataFile:
            verification_data = dataFile.read()
            verification_data = read_js_object(verification_data)

        try:
            results['Number of Invoations'] = verification_data['verifJSON']['functions'][0]['data'][0]

            latency = verification_data['verifJSON']['functions'][0]['data'][1].split(",")
            results['Latency (MIN)'] = latency[0]
            results['Latency (MAX)'] = latency[1]
            results['Latency (AVG)'] = latency[2]

            ii = verification_data['verifJSON']['functions'][0]['data'][2].split(",")
            results['ii (MIN)'] = ii[0]
            results['ii (MAX)'] = ii[1]
            results['ii (AVG)'] = ii[2]
        except Exception:
            print('Verification data not found. Run ./[projectname]-fpga to generate.')
    else:
        print('Verification file not found. Run ./[projectname]-fpga to generate.')
    return results


def _find_reports(rpt_dir):
    results = {}
    results.update(_read_hls_file(rpt_dir + '/lib/report_data.js'))
    results.update(_read_verification_file(rpt_dir + '/lib/verification_data.js'))
    results.update(_read_quartus_file(rpt_dir + '/lib/quartus_data.js'))
    return results
