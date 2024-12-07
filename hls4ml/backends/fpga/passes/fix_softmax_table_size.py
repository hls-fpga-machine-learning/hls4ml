import warnings

from hls4ml.model.layers import Layer, Softmax
from hls4ml.model.optimizer import OptimizerPass


class FixSoftmaxTableSize(OptimizerPass):
    def match(self, node):
        if not isinstance(node, Softmax):
            return False
        if 'inv_table_size' in node.attributes:
            return False  # handler generating inv_table_size sets it properly
        return True

    def transform(self, model, node: Layer):
        inp_layer = node.get_input_node()  # type: ignore
        if not isinstance(inp_layer, Layer):
            raise RuntimeError(f'Softmax layer {node.name} does not have an input layer')

        input_bw: int = inp_layer.get_attr('result_t').precision.width  # type: ignore
        table_bw: int = node.get_attr('inv_table_t').precision.width  # type: ignore
        table_size = int(node.get_attr('table_size'))  # type: ignore

        backend = model.config.config['Backend']

        # Somehow, Intel want one extra bits for the table.
        # I don't know why but if not simulation will crash with segmentation fault.
        backend_limitation = -1 if backend == 'Quartus' else 0

        if 2 ** (min(input_bw, table_bw) + backend_limitation) < table_size:
            # If table size is too large w.r.t. input bitwidth and table bitwidth,
            # reduce table size to avoid undefined behavior when cutting indices from,
            # fixed point number.
            node.set_attr('table_size', str(2 ** (min(input_bw, table_bw) + backend_limitation)))
            if 2**input_bw < table_size:
                # The warning message does not have to be looking like this, but you are asking
                # 125 characters long line.
                warnings.warn(
                    (
                        f"Softmax layer {node.name} table size is too large for input"
                        f"bitwidth {input_bw}. Setting table size to {2**input_bw}."
                        "To avoid this warning, please increase input bitwidth or"
                        "decrease table size."
                    ),
                    stacklevel=1,
                )
            if 2**table_bw < table_size:
                warnings.warn(
                    (
                        f"Softmax layer {node.name} table size is too large for input"
                        f"bitwidth {input_bw}. Setting table size to {2**input_bw}."
                        "To avoid this warning, please increase input bitwidth or"
                        "decrease table size."
                    ),
                    stacklevel=1,
                )
            if backend == 'Quartus':
                warnings.warn(
                    (
                        "Quartus backend's table size is half of 2^min(input_bw-1,table_bw-1)"
                        " instead of 2^min(input_bw,table_bw)."
                    ),
                    stacklevel=1,
                )
            return False


def register_softmax__table_size_fix(backend):
    backend.register_pass('fix_softmax_table_size', FixSoftmaxTableSize)
