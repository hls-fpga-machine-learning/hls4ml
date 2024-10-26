import json
import os
import shutil
import uuid
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sb

from hls4ml.model.graph import ModelGraph
from hls4ml.model.layers import GRU, LSTM, SeparableConv1D, SeparableConv2D

try:
    import keras

    __keras_profiling_enabled__ = True
except ImportError:
    __keras_profiling_enabled__ = False

try:
    import torch

    __torch_profiling_enabled__ = True
except ImportError:
    __torch_profiling_enabled__ = False

try:
    import qkeras

    __qkeras_profiling_enabled__ = True
except ImportError:
    __qkeras_profiling_enabled__ = False

__keras_activations = list()
if __keras_profiling_enabled__:
    __keras_activations.append(keras.layers.Activation)
if __qkeras_profiling_enabled__:
    __keras_activations.append(qkeras.QActivation)


def get_unoptimized_hlsmodel(model):
    from hls4ml.converters import convert_from_config

    new_config = model.config.config.copy()
    new_config['HLSConfig'] = json.loads(json.dumps(new_config['HLSConfig']))

    new_output_dir = uuid.uuid4().hex

    while os.path.exists(new_output_dir):
        new_output_dir = uuid.uuid4().hex

    if 'SkipOptimizers' in new_config['HLSConfig']:
        del new_config['HLSConfig']['SkipOptimizers']

    new_config['HLSConfig']['Optimizers'] = []
    new_config['OutputDir'] = new_output_dir

    return convert_from_config(new_config), new_output_dir


def array_to_summary(x, fmt='boxplot'):
    if fmt == 'boxplot':
        y = {'med': np.median(x), 'q1': np.percentile(x, 25), 'q3': np.percentile(x, 75), 'whislo': min(x), 'whishi': max(x)}
    elif fmt == 'histogram':
        # Power of 2 bins covering data range
        high = np.ceil(np.log2(max(x))) + 1
        low = np.floor(np.log2(min(x))) - 1
        bits = np.arange(low, high, 1)
        bins = 2**bits
        h, b = np.histogram(x, bins=bins)
        h = h * 1.0 / float(sum(h))  # normalize
        y = {'h': h, 'b': np.log2(b)}
    return y


def boxplot(data, fmt='longform'):
    if fmt == 'longform':
        f = plt.figure()  # figsize=(3, 3))
        hue = 'layer' if 'layer' in data.keys() else None
        vp = sb.boxplot(x='x', y='weight', hue=hue, data=data[data['x'] > 0], showfliers=False)
        vp.set_yticklabels(vp.get_yticklabels(), rotation=45, ha='right')
        if hue is not None:
            vp.get_legend().remove()
        vp.set_xscale('log', base=2)
        return f
    elif fmt == 'summary':
        from matplotlib.patches import Rectangle

        medianprops = dict(linestyle='-', color='k')
        f, ax = plt.subplots(1, 1)
        data.reverse()
        colors = sb.color_palette("Blues", len(data))
        bp = ax.bxp(data, showfliers=False, vert=False, medianprops=medianprops)
        # add colored boxes
        for line, color in zip(bp['boxes'], colors):
            x = line.get_xdata()
            xl, xh = min(x), max(x)
            y = line.get_ydata()
            yl, yh = min(y), max(y)
            rect = Rectangle((xl, yl), (xh - xl), (yh - yl), fill=True, color=color)
            ax.add_patch(rect)
        ax.set_yticklabels([d['weight'] for d in data])
        ax.set_xscale('log', base=2)
        plt.xlabel('x')
        return f
    else:
        return None


def histogram(data, fmt='longform'):
    f = plt.figure()
    from matplotlib.ticker import MaxNLocator

    n = len(data) if fmt == 'summary' else len(data['weight'].unique())
    colors = sb.color_palette("husl", n)
    if fmt == 'longform':
        for i, weight in enumerate(data['weight'].unique()):
            y = array_to_summary(data[data['weight'] == weight]['x'], fmt='histogram')
            plt.bar(y['b'][:-1], y['h'], width=1, fill=False, label=weight, edgecolor=colors[i])
    elif fmt == 'summary':
        for i, weight in enumerate(data):
            plt.bar(weight['b'][:-1], weight['h'], width=1, fill=False, label=weight['weight'], edgecolor=colors[i])

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('log2(x)')
    plt.ylabel('frequency')
    plt.legend()
    return f


plots = {'boxplot': boxplot, 'histogram': histogram}


def types_boxplot(data, fmt='longform'):
    from matplotlib.patches import PathPatch, Rectangle

    ax = plt.gca()
    _ = plt.gcf()
    # Scale the data
    data['low'] = 2.0 ** data['low']
    data['high'] = 2.0 ** data['high']

    # Plot the custom precisions
    ticks = np.array([tick.get_text() for tick in plt.yticks()[1]])
    # Get the coordinates of the boxes to place the markers
    if fmt == 'longform':
        # seaborn adjusts the box positions slightly in groups
        boxes = [c.get_extents().inverse_transformed(ax.transData) for c in ax.get_children() if isinstance(c, PathPatch)]
        ys = [(box.y0 + box.y1) / 2 for box in boxes]
        ys = [(y, y) for y in ys]
    elif fmt == 'summary':
        ys = [(y, y) for y in plt.yticks()[0]]
    for _irow, row in data[data['layer'] != 'model'].iterrows():
        if row['layer'] in ticks:
            iy = np.argwhere(ticks == row['layer'])[0][0]  # Determine which layer in the plot
            rectangle = Rectangle(
                (row['low'], ys[iy][0] - 0.4), row['high'] - row['low'], 0.8, fill=True, color='grey', alpha=0.2
            )
            ax.add_patch(rectangle)


def types_histogram(data, fmt='longform'):
    ax = plt.gca()
    layers = np.array(ax.get_legend_handles_labels()[1])
    colors = sb.color_palette("husl", len(layers))
    ylim = ax.get_ylim()
    for _irow, row in data[data['layer'] != 'model'].iterrows():
        if row['layer'] in layers:
            col = colors[np.argwhere(layers == row['layer'])[0][0]]
            plt.plot((row['low'], row['low']), ylim, '--', color=col)
            plt.plot((row['high'], row['high']), ylim, '--', color=col)


types_plots = {'boxplot': types_boxplot, 'histogram': types_histogram}


def ap_fixed_WIFS(dtype):
    from hls4ml.backends import VivadoBackend

    dtype = VivadoBackend.convert_precision_string(dtype)
    W, I, F, S = dtype.width, dtype.integer, dtype.fractional, dtype.signed
    return W, I, F, S


def types_hlsmodel(model):
    data = {'layer': [], 'low': [], 'high': []}
    # Plot the default precision
    default_precision = model.config.model_precision['default']
    W, I, F, S = ap_fixed_WIFS(default_precision)
    data['layer'].append('model')
    data['low'].append(-F)
    data['high'].append(I - 1 if S else I)

    for layer in model.get_layers():
        if isinstance(layer, GRU) or isinstance(layer, LSTM):
            suffix = ['w', 'rw', 'b', 'rb']
        elif isinstance(layer, SeparableConv1D) or isinstance(layer, SeparableConv2D):
            suffix = ['dw', 'pw', 'db', 'pb']
        else:
            suffix = ['w', 'b']
        for iw, weight in enumerate(layer.get_weights()):
            wname = f'{layer.name}/{suffix[iw]}'
            T = weight.type
            if T.name != 'model':
                W, I, F, S = ap_fixed_WIFS(T.precision)
                data['layer'].append(wname)
                data['low'].append(-F)
                data['high'].append(I - 1 if S else I)
    data = pandas.DataFrame(data)
    return data


def activation_types_hlsmodel(model):
    data = {'layer': [], 'low': [], 'high': []}
    # Get the default precision
    default_precision = model.config.model_precision['default']
    W, I, F, S = ap_fixed_WIFS(default_precision)
    data['layer'].append('model')
    data['low'].append(-F)
    data['high'].append(I - 1 if S else I)
    for layer in model.get_layers():
        T = layer.get_output_variable().type.precision
        W, I, F, S = ap_fixed_WIFS(T)
        data['layer'].append(layer.name)
        data['low'].append(-F)
        data['high'].append(I - 1 if S else I)
    data = pandas.DataFrame(data)
    return data


def weights_hlsmodel(model, fmt='longform', plot='boxplot'):
    if fmt == 'longform':
        data = {'x': [], 'layer': [], 'weight': []}
    elif fmt == 'summary':
        data = []

    for layer in model.get_layers():
        if isinstance(layer, GRU) or isinstance(layer, LSTM):
            suffix = ['w', 'rw', 'b', 'rb']
        elif isinstance(layer, SeparableConv1D) or isinstance(layer, SeparableConv2D):
            suffix = ['dw', 'pw', 'db', 'pb']
        else:
            suffix = ['w', 'b']
        name = layer.name
        for iw, weight in enumerate(layer.get_weights()):
            label = f'{name}/{suffix[iw]}'
            w = weight.data.flatten()
            w = abs(w[w != 0])
            n = len(w)
            if n == 0:
                print(f'Weights for {name} are only zeros, ignoring.')
                break
            if fmt == 'longform':
                data['x'].extend(w.tolist())
                data['layer'].extend([name] * len(w))
                data['weight'].extend([label] * len(w))
            elif fmt == 'summary':
                data.append(array_to_summary(w, fmt=plot))
                data[-1]['layer'] = name
                data[-1]['weight'] = label

    if fmt == 'longform':
        data = pandas.DataFrame(data)
    return data


def _keras_batchnorm(layer):
    weights = layer.get_weights()
    epsilon = layer.epsilon

    gamma = weights[0]
    beta = weights[1]
    mean = weights[2]
    var = weights[3]

    scale = gamma / np.sqrt(var + epsilon)
    bias = beta - gamma * mean / np.sqrt(var + epsilon)

    return [scale, bias], ['s', 'b']


def _keras_layer(layer):
    return layer.get_weights(), ['w', 'b']


def _keras_lstm(layer):
    return layer.get_weights(), ['w', 'u', 'b']


keras_process_layer_map = defaultdict(
    lambda: _keras_layer,
    {
        'BatchNormalization': _keras_batchnorm,
        'QBatchNormalization': _keras_batchnorm,
        'LSTM': _keras_lstm,
        'QLSTM': _keras_lstm,
    },
)


def activations_hlsmodel(model, X, fmt='summary', plot='boxplot'):
    if fmt == 'longform':
        raise NotImplementedError
    elif fmt == 'summary':
        data = []

    _, trace = model.trace(np.ascontiguousarray(X))

    if len(trace) == 0:
        raise RuntimeError("ModelGraph must have tracing on for at least 1 layer (this can be set in its config)")

    for layer in trace.keys():
        print(f"   {layer}")

        if fmt == 'summary':
            y = trace[layer].flatten()
            y = abs(y[y != 0])

            if len(y) == 0:
                print(f'Activations for {layer} are only zeros, ignoring.')
                continue

            data.append(array_to_summary(y, fmt=plot))
            data[-1]['weight'] = layer

    return data


def weights_keras(model, fmt='longform', plot='boxplot'):
    if fmt == 'longform':
        data = {'x': [], 'layer': [], 'weight': []}
    elif fmt == 'summary':
        data = []
    for layer in model.layers:
        name = layer.name
        weights, suffix = keras_process_layer_map[type(layer).__name__](layer)

        for i, w in enumerate(weights):
            label = f'{name}/{suffix[i]}'
            w = w.flatten()
            w = abs(w[w != 0])
            n = len(w)
            if n == 0:
                print(f'Weights for {name} are only zeros, ignoring.')
                break
            if fmt == 'longform':
                data['x'].extend(w.tolist())
                data['layer'].extend([name] * n)
                data['weight'].extend([label] * n)
            elif fmt == 'summary':
                data.append(array_to_summary(w, fmt=plot))
                data[-1]['layer'] = name
                data[-1]['weight'] = label

    if fmt == 'longform':
        data = pandas.DataFrame(data)
    return data


def activations_keras(model, X, fmt='longform', plot='boxplot'):
    # test layer by layer on data
    if fmt == 'longform':
        # return long form pandas dataframe for
        # seaborn boxplot
        data = {'x': [], 'weight': []}
    elif fmt == 'summary':
        # return summary statistics for matplotlib.axes.Axes.bxp
        # or histogram bin edges and heights
        data = []
    outputs = _get_outputs(
        [layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)], X, model.input
    )
    outputs = dict(zip([layer.name for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)], outputs))
    for layer_name, y in outputs.items():
        print(f"   {layer_name}")
        y = y.flatten()
        y = abs(y[y != 0])
        if len(y) == 0:
            print(f'Activations for {layer_name} are only zeros, ignoring.')
            continue
        if fmt == 'longform':
            data['x'].extend(y.tolist())
            data['weight'].extend([layer_name for i in range(len(y))])
        elif fmt == 'summary':
            data.append(array_to_summary(y, fmt=plot))
            data[-1]['weight'] = layer_name

    if fmt == 'longform':
        data = pandas.DataFrame(data)
    return data


def weights_torch(model, fmt='longform', plot='boxplot'):
    wt = WeightsTorch(model, fmt, plot)
    return wt.get_weights()


def _torch_batchnorm(layer):
    weights = list(layer.parameters())
    epsilon = layer.eps

    gamma = weights[0]
    beta = weights[1]
    if layer.track_running_stats:
        mean = layer.running_mean
        var = layer.running_var
    else:
        mean = torch.tensor(np.ones(20))
        var = torch.tensor(np.zeros(20))

    scale = gamma / np.sqrt(var + epsilon)
    bias = beta - gamma * mean / np.sqrt(var + epsilon)

    return [scale, bias], ['s', 'b']


def _torch_layer(layer):
    return list(layer.parameters()), ['w', 'b']


def _torch_rnn(layer):
    return list(layer.parameters()), ['w_ih_l0', 'w_hh_l0', 'b_ih_l0', 'b_hh_l0']


torch_process_layer_map = defaultdict(
    lambda: _torch_layer,
    {
        'BatchNorm1d': _torch_batchnorm,
        'BatchNorm2d': _torch_batchnorm,
        'RNN': _torch_rnn,
        'LSTM': _torch_rnn,
        'GRU': _torch_rnn,
    },
)


class WeightsTorch:
    def __init__(self, model: torch.nn.Module, fmt: str = 'longform', plot: str = 'boxplot') -> None:
        self.model = model
        self.fmt = fmt
        self.plot = plot
        self.registered_layers = list()
        self._find_layers(self.model, self.model.__class__.__name__)

    def _find_layers(self, model, module_name):
        for name, module in model.named_children():
            if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
                self._find_layers(module, module_name + "." + name)
            elif isinstance(module, (torch.nn.Module)) and self._is_parameterized(module):
                if len(list(module.named_children())) != 0:
                    # custom nn.Module, continue search
                    self._find_layers(module, module_name + "." + name)
                else:
                    self._register_layer(module_name + "." + name)

    def _is_registered(self, name: str) -> bool:
        return name in self.registered_layers

    def _register_layer(self, name: str) -> None:
        if self._is_registered(name) is False:
            self.registered_layers.append(name)

    def _is_parameterized(self, module: torch.nn.Module) -> bool:
        return any(p.requires_grad for p in module.parameters())

    def _get_weights(self) -> pandas.DataFrame | list[dict]:
        if self.fmt == 'longform':
            data = {'x': [], 'layer': [], 'weight': []}
        elif self.fmt == 'summary':
            data = []
        for layer_name in self.registered_layers:
            layer = self._get_layer(layer_name, self.model)
            name = layer.__class__.__name__
            weights, suffix = torch_process_layer_map[layer.__class__.__name__](layer)
            for i, w in enumerate(weights):
                label = f'{name}/{suffix[i]}'
                w = weights[i].detach().numpy()
                w = w.flatten()
                w = abs(w[w != 0])
                n = len(w)
                if n == 0:
                    print(f'Weights for {name} are only zeros, ignoring.')
                    break
                if self.fmt == 'longform':
                    data['x'].extend(w.tolist())
                    data['layer'].extend([name] * n)
                    data['weight'].extend([label] * n)
                elif self.fmt == 'summary':
                    data.append(array_to_summary(w, fmt=self.plot))
                    data[-1]['layer'] = name
                    data[-1]['weight'] = label

        if self.fmt == 'longform':
            data = pandas.DataFrame(data)
        return data

    def get_weights(self) -> pandas.DataFrame | list[dict]:
        return self._get_weights()

    def get_layers(self) -> list[str]:
        return self.registered_layers

    def _get_layer(self, layer_name: str, module: torch.nn.Module) -> torch.nn.Module:
        for name in layer_name.split('.')[1:]:
            module = getattr(module, name)
        return module


def activations_torch(model, X, fmt='longform', plot='boxplot'):
    X = torch.Tensor(X)
    if fmt == 'longform':
        data = {'x': [], 'weight': []}
    elif fmt == 'summary':
        data = []

    partial_model = torch.nn.Sequential
    layers = []
    for layer in model.children():
        lname = layer.__class__.__name__
        layers.append(layer)
        pm = partial_model(*layers)
        print(f"   {lname}")
        y = pm(X).flatten().detach().numpy()
        y = abs(y[y != 0])
        if len(y) == 0:
            print(f'Activations for {lname} are only zeros, ignoring.')
            continue
        if fmt == 'longform':
            data['x'].extend(y.tolist())
            data['weight'].extend([lname for _ in range(len(y))])
        elif fmt == 'summary':
            data.append(array_to_summary(y, fmt=plot))
            data[-1]['weight'] = lname

    if fmt == 'longform':
        data = pandas.DataFrame(data)
    return data


def numerical(model=None, hls_model=None, X=None, plot='boxplot'):
    """Perform numerical profiling of a model.

    Args:
        model (optional): Keras of PyTorch model. Defaults to None.
        hls_model (ModelGraph, optional): The ModelGraph to profile. Defaults to None.
        X (ndarray, optional): Test data on which to evaluate the model to profile activations.
            Must be formatted suitably for the ``model.predict(X)``. Defaults to None.
        plot (str, optional): The type of plot to produce. Options are: 'boxplot' (default), 'violinplot', 'histogram',
            'FacetGrid'. Defaults to 'boxplot'.

    Returns:
        tuple: The quadruple of produced figures. First weights and biases
            for the pre- and post-optimization models respectively,
            then activations for the pre- and post-optimization models
            respectively. (Optimizations are applied to an ModelGraph by hls4ml,
            a post-optimization ModelGraph is a final model).
    """
    wp, wph, ap, aph = None, None, None, None

    hls_model_present = hls_model is not None and isinstance(hls_model, ModelGraph)
    model_present = model is not None

    if hls_model_present:
        before = " (before optimization)"
        after = " (final / after optimization)"
        hls_model_unoptimized, tmp_output_dir = get_unoptimized_hlsmodel(hls_model)
    else:
        before = ""
        after = ""
        hls_model_unoptimized, tmp_output_dir = None, None

    print("Profiling weights" + before)
    data = None

    if hls_model_present:
        data = weights_hlsmodel(hls_model_unoptimized, fmt='summary', plot=plot)
    elif model_present:
        if __keras_profiling_enabled__ and isinstance(model, keras.Model):
            data = weights_keras(model, fmt='summary', plot=plot)
        elif __torch_profiling_enabled__ and isinstance(model, torch.nn.Module):
            data = weights_torch(model, fmt='summary', plot=plot)

    if data is None:
        print("Only keras, PyTorch and ModelGraph models " + "can currently be profiled")

        if hls_model_present and os.path.exists(tmp_output_dir):
            shutil.rmtree(tmp_output_dir)

        return wp, wph, ap, aph

    wp = plots[plot](data, fmt='summary')  # weight plot

    if hls_model_present and plot in types_plots:
        t_data = types_hlsmodel(hls_model_unoptimized)
        types_plots[plot](t_data, fmt='summary')

    plt.title("Distribution of (non-zero) weights" + before)
    plt.tight_layout()

    if hls_model_present:
        print("Profiling weights" + after)

        data = weights_hlsmodel(hls_model, fmt='summary', plot=plot)
        wph = plots[plot](data, fmt='summary')  # weight plot

        if plot in types_plots:
            t_data = types_hlsmodel(hls_model)
            types_plots[plot](t_data, fmt='summary')

        plt.title("Distribution of (non-zero) weights" + after)
        plt.tight_layout()

    if X is not None:
        print("Profiling activations" + before)
        data = None
        if __keras_profiling_enabled__ and isinstance(model, keras.Model):
            data = activations_keras(model, X, fmt='summary', plot=plot)
        elif __torch_profiling_enabled__ and isinstance(model, torch.nn.Sequential):
            data = activations_torch(model, X, fmt='summary', plot=plot)

        if data is not None:
            ap = plots[plot](data, fmt='summary')  # activation plot
            if hls_model_present and plot in types_plots:
                t_data = activation_types_hlsmodel(hls_model_unoptimized)
                types_plots[plot](t_data, fmt='summary')
            plt.title("Distribution of (non-zero) activations" + before)
            plt.tight_layout()

        if hls_model_present:
            print("Profiling activations" + after)
            data = activations_hlsmodel(hls_model, X, fmt='summary', plot=plot)
            aph = plots[plot](data, fmt='summary')

            t_data = activation_types_hlsmodel(hls_model)
            types_plots[plot](t_data, fmt='summary')

            plt.title("Distribution of (non-zero) activations (final / after optimization)")
            plt.tight_layout()

    if hls_model_present and os.path.exists(tmp_output_dir):
        shutil.rmtree(tmp_output_dir)

    return wp, wph, ap, aph


#########
# COMPARE OUTPUT IMPLEMENTATION
#########
def _is_ignored_layer(layer):
    """Some layers need to be ingored during inference"""
    if isinstance(layer, (keras.layers.InputLayer, keras.layers.Dropout)):
        return True
    return False


def _get_outputs(layers, X, model_input):
    """Get outputs of intermediate layers"""
    partial_models = keras.models.Model(inputs=model_input, outputs=[layer.output for layer in layers])
    y = partial_models.predict(X)
    return y


def get_ymodel_keras(keras_model, X):
    """Calculate each layer's ouput and put them into a dictionary.

    Args:
        keras_model (_type_): A keras Model
        X (ndarray): Test data on which to evaluate the model to profile activations.
            Must be formatted suitably for the ``model.predict(X)``.

    Returns:
        dict: A dictionary in the form {"layer_name": ouput array of layer}.
    """
    ymodel = {}
    traced_layers = []
    layer_names = []
    for layer in keras_model.layers:
        if _is_ignored_layer(layer):
            continue
        # If the layer has activation integrated then separate them
        # Note that if the layer is a standalone activation layer then skip this
        name = layer.name
        if (
            hasattr(layer, 'activation')
            and layer.activation is not None
            and not isinstance(layer, tuple(__keras_activations))
            and layer.activation.__name__ != 'linear'
        ):
            tmp_activation = layer.activation
            layer.activation = None
            ymodel.update({layer.name: _get_outputs([layer], X, keras_model.input)})
            layer.activation = tmp_activation
            name = layer.name + f"_{tmp_activation.__name__}"
        traced_layers.append(layer)
        layer_names.append(name)
    outputs = _get_outputs(traced_layers, X, keras_model.input)
    for name, output in zip(layer_names, outputs):
        ymodel[name] = output
    print("Done taking outputs for Keras model.")
    return ymodel


def _norm_diff(ymodel, ysim):
    """Calculate the square root of the sum of the squares of the differences"""
    diff = {}

    for key in list(ysim.keys()):
        diff[key] = np.linalg.norm(ysim[key] - ymodel[key])

    # ---Bar Plot---
    f, ax = plt.subplots()
    plt.bar(list(diff.keys()), list(diff.values()))
    plt.title("layer-by-layer output differences")
    ax.set_ylabel('Norm of difference vector')
    plt.xticks(rotation=90)
    plt.tight_layout()
    return f


def _dist_diff(ymodel, ysim):
    """
    Calculate the normalized distribution of the differences of the elements
    of the output vectors.
    If difference >= original value then the normalized difference will be set to 1,
    meaning "very difference".
    If difference < original value then the normalized difference would be difference/original.
    """

    diff = {}

    for key in list(ysim.keys()):
        flattened_ysim = ysim[key].flatten()
        flattened_ymodel = np.array(ymodel[key]).flatten()

        diff[key] = np.absolute(flattened_ymodel - flattened_ysim) / np.linalg.norm(flattened_ymodel - flattened_ysim)
        diff_vector = np.absolute(flattened_ymodel - flattened_ysim)
        abs_ymodel = np.absolute(flattened_ymodel)

        normalized_diff = np.zeros(diff_vector.shape)
        normalized_diff[(diff_vector >= abs_ymodel) & (abs_ymodel > 0) & (diff_vector > 0)] = 1

        # Fill out the rest
        index = diff_vector < abs_ymodel
        normalized_diff[index] = diff_vector[index] / abs_ymodel[index]

        diff[key] = normalized_diff

    # ---Box Plot---
    f, ax = plt.subplots()
    pos = np.array(range(len(list(diff.values())))) + 1
    ax.boxplot(list(diff.values()), sym='k+', positions=pos)

    # --formatting
    plt.title("Layer-by-layer distribution of output differences")
    ax.set_xticklabels(list(diff.keys()))
    ax.set_ylabel('Normalized difference')
    ax.set_ylabel('Percent difference.')
    plt.xticks(rotation=90)
    plt.tight_layout()

    return f


def compare(keras_model, hls_model, X, plot_type="dist_diff"):
    """Compare each layer's output in keras and hls model. Note that the hls_model should not be compiled before using this.

    Args:
        keras_model: Original keras model.
        hls_model (ModelGraph): Converted ModelGraph, with "Trace:True" in the configuration file.
        X (ndarray): Input tensor for the model.
        plot_type (str, optional): Different methods to visualize the y_model and y_sim differences.
            Possible options include:
            - 'norm_diff':: square root of the sum of the squares of the differences between each output vectors.
            - 'dist_diff':: The normalized distribution of the differences of the elements between two output vectors.
            Defaults to "dist_diff".

    Returns:
        matplotlib figure: Plot object of the histogram depicting the difference in each layer's output.
    """

    # Take in output from both models
    # Note that each y is a dictionary with structure {"layer_name": flattened ouput array}
    ymodel = get_ymodel_keras(keras_model, X)
    _, ysim = hls_model.trace(X)

    print("Plotting difference...")
    f = plt.figure()
    if plot_type == "norm_diff":
        f = _norm_diff(ymodel, ysim)
    elif plot_type == "dist_diff":
        f = _dist_diff(ymodel, ysim)

    return f
