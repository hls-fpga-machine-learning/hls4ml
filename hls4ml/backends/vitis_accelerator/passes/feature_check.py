from hls4ml.model.optimizer import OptimizerPass


class ValidateConvImplementation(OptimizerPass):
    def match(self, node):
        return 'Conv' in node.class_name

    def transform(self, model, node):
        if node.get_attr('implementation', 'linebuffer') == 'encoded':
            print(
                f'WARNING: "Encoded" implementation in "{node.name}" ({node.class_name}) is not supported in Vitis backend. '
                'Switching to "LineBuffer" implementation.'
            )
            node.set_attr('implementation', 'linebuffer')


class ValidateStrategy(OptimizerPass):
    _resource_layer_cls = ['Conv1D', 'Conv2D', 'Dense']

    def match(self, node):
        is_resource_layer = len([layer_cls for layer_cls in self._resource_layer_cls if layer_cls in node.class_name]) > 0
        is_resource_strategy = node.model.config.is_resource_strategy(node)

        return is_resource_layer and is_resource_strategy

    def transform(self, model, node):
        n_in, _ = model.config.backend.get_layer_mult_size(node)
        rf = node.get_attr('reuse_factor')
        if rf > n_in and rf % n_in > 0:
            print(
                f'WARNING: "Resource" strategy in "{node.name}" ({node.class_name}) may have suboptimal QoR in Vitis '
                'backend due to use of "urem" cores.\n'
                'Consider using a different ReuseFactor or switching to "Latency" strategy.'
            )
