import numpy as np

from hls4ml.backends import Backend
from hls4ml.backends.template import FunctionCallTemplate
from hls4ml.model.layers import Layer
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer, UnaryLUT
from hls4ml.model.types import Source


def to_apfixed(k, b, i, RND, SAT):
    u = 'u' if k == 0 else ''
    return f'ap_{u}fixed<{b},{i},AP_{RND},AP_{SAT}>'


def to_acfixed(k, b, i, RND, SAT):
    k = 'false' if k == 0 else 'true'
    return f'ac_fixed<{b},{i},{k},AC_{RND},AC_{SAT}>'


def generate_mask_fn(
    name: str, shape: tuple[int, ...], k: np.ndarray, b: np.ndarray, i: np.ndarray, RND: str, SAT: str, backend: str
) -> str:
    """Generate heterogenous quantization mask function, ONLY works for IOType=io_parallel"""
    assert k.shape[0] == b.shape[0] == i.shape[0] == 1
    assert backend.lower() in ('quartus', 'vivado', 'vitis'), f'Backend {backend} not tested'
    Ks, Bs, Is = k[0], b[0], i[0]
    Ks, Bs, Is = np.broadcast_to(Ks, shape), np.broadcast_to(Bs, shape), np.broadcast_to(Is, shape)
    Ks, Bs, Is = Ks.ravel(), Bs.ravel(), Is.ravel()
    masks = []
    to_fixed = to_acfixed if backend.lower() == 'quartus' else to_apfixed
    for idx, (k, b, i) in enumerate(zip(Ks, Bs, Is)):
        if b == 0:
            fn = f'out[{idx}] = 0;'
        else:
            fn = f'out[{idx}] = {to_fixed(k, b, i, RND, SAT)}(inp[{idx}]);'
        masks.append(f'    {fn}')
    body = "\n".join(masks)
    mask_fn = f'''
template<typename input_t, typename output_t>
void {name}(input_t *inp, output_t *out) {{
    #pragma HLS INLINE

{body}
}}
'''
    return mask_fn


class ProcessFixedPointQuantizerLayer(OptimizerPass):
    def match(self, node: Layer):
        return isinstance(node, FixedPointQuantizer)

    def transform(self, model, node: FixedPointQuantizer):
        if node.fusible:
            model.remove_node(node, rewire=True)
            return True

        if model.config.config['IOType'] != 'io_parallel':
            raise NotImplementedError('Heterogenous quantization for activations is only supported with IOType=io_parallel')

        backend = model.config.config['Backend']

        name = node.name

        assert node.mask_kbi is not None
        k, b, i = node.mask_kbi
        RND = node.RND
        SAT = node.SAT
        mask_fn: str = generate_mask_fn(name, node.get_input_variable().shape, k, b, i, RND, SAT, backend)

        node.set_attr('mask_fn_codegen', Source(mask_fn))


class ProcessFixedPointQuantizerCall(FunctionCallTemplate):
    def __init__(self):
        super().__init__(FixedPointQuantizer, include_header=[])
        self.template = '{namespace}::{name}<{input_t}, {output_t}>({input}, {output});'

    def format(self, node):
        params = self._default_function_params(node)
        namespace = node.model.config.writer_config.get('Namespace', None) or 'nnet'
        params['namespace'] = namespace

        return self.template.format(**params)


class ProcessUnaryLUTCall(FunctionCallTemplate):
    def __init__(self):
        super().__init__(UnaryLUT, include_header=[])
        self.template = 'nnet::unary_lut<{input_t}, {output_t}, {config}>({input}, {output}, {table});'
        self.include_header = [
            'nnet_utils/nnet_activation.h',
            'nnet_utils/nnet_activation_stream.h',
        ]

    def format(self, node):
        params = self._default_function_params(node)
        node.attributes['result_t'].precision = node.attributes['table_t'].precision
        params['config'] = f'unary_lut_config{node.index}'
        params['table'] = node.get_weights('table').name

        return self.template.format(**params)


def register_hgq_proxy_model(backend: Backend):
    backend.register_pass('process_fixed_point_quantizer_layer', ProcessFixedPointQuantizerLayer)
    backend.register_template(ProcessFixedPointQuantizerCall)
    backend.register_template(ProcessUnaryLUTCall)
