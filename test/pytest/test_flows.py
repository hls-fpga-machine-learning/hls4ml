import pytest

import hls4ml

'''
Tests for model flows.
Construct some dummy optimizer passes and flows that do nothing.
Passes record their label to the model.
Tests check that the order of applied passes matches the expectations
'''


class DummyPass(hls4ml.model.optimizer.OptimizerPass):
    def __init__(self, label):
        self.label = label

    def match(self, node):
        return True

    def transform(self, model, node):
        if getattr(model, 'test_flow_passes', None) is None:
            model.test_flow_passes = []
        model.test_flow_passes.append(self.label)
        return False


class DummyPassA(DummyPass):
    def __init__(self):
        super().__init__('A')


class DummyPassB(DummyPass):
    def __init__(self):
        super().__init__('B')


class DummyPassC(DummyPass):
    def __init__(self):
        super().__init__('C')


hls4ml.model.optimizer.register_pass('A', DummyPassA)
hls4ml.model.optimizer.register_pass('B', DummyPassB)
hls4ml.model.optimizer.register_pass('C', DummyPassC)

DummyFlowA = hls4ml.model.flow.register_flow('A', ['A'])
DummyFlowB = hls4ml.model.flow.register_flow('B', ['B'])
DummyFlowC = hls4ml.model.flow.register_flow('C', ['C'])
DummyFlowAB = hls4ml.model.flow.register_flow('AB', ['A', 'B'])
DummyFlowBReqA = hls4ml.model.flow.register_flow('BReqA', ['B'], requires=[DummyFlowA])
DummyFlowCReqBReqA = hls4ml.model.flow.register_flow('CReqBReqA', ['C'], requires=[DummyFlowBReqA])


def dummy_flow_model():
    layers = [{'class_name': 'Input', 'name': 'layer0_input', 'input_shape': [1]}]
    config = {'HLSConfig': {'Model': {'Precision': 'ap_fixed<32,16>', 'ReuseFactor': 1}, 'Flows': []}}
    model = hls4ml.model.ModelGraph.from_layer_list(config, layers)
    return model


class FlowTester:
    index = 0

    def __init__(self, flows_to_apply, expected_pass_order, reapply):
        self.flows_to_apply = flows_to_apply
        self.expected_pass_order = expected_pass_order
        self.reapply = reapply
        self.index = FlowTester.index
        FlowTester.index += 1

    def run(self):
        model = dummy_flow_model()
        model.test_flow_passes = []
        for flow in self.flows_to_apply:
            model.apply_flow(flow, self.reapply)
        self.observed_pass_order = model.test_flow_passes
        return self.observed_pass_order == self.expected_pass_order


flow_tests = [
    FlowTester(['A', 'B', 'C'], ['A', 'B', 'C'], 'single'),  # independent flows in order
    FlowTester(['A', 'A'], ['A', 'A'], 'single'),  # same flow twice, single application
    FlowTester(['A', 'A'], ['A', 'A'], 'all'),  # same flow twice with reapply
    FlowTester(['A', 'A'], ['A'], 'none'),  # same flow twice with none
    FlowTester(['BReqA'], ['A', 'B'], 'single'),  # one flow with a dependency
    FlowTester(['CReqBReqA'], ['A', 'B', 'C'], 'single'),  # one flow with dependency chain
    FlowTester(['CReqBReqA', 'A'], ['A', 'B', 'C', 'A'], 'single'),  # one flow with dependency chain, repeat dependency
    FlowTester(['CReqBReqA', 'A'], ['A', 'B', 'C', 'A'], 'all'),  # one flow with dependency chain, repeat dependency
    FlowTester(['CReqBReqA', 'A'], ['A', 'B', 'C'], 'none'),  # one flow with dependency chain, repeat depencency
    FlowTester(['A', 'CReqBReqA'], ['A', 'B', 'C'], 'single'),  # one flow with dependency chain, repeat depencency
    FlowTester(['A', 'CReqBReqA'], ['A', 'A', 'B', 'C'], 'all'),  # one flow with dependency chain, repeat depencency
    FlowTester(['A', 'CReqBReqA'], ['A', 'B', 'C'], 'none'),  # one flow with dependency chain, repeat depencency
    FlowTester(['A', 'BReqA'], ['A', 'B'], 'single'),  # second flow dependency already run
    FlowTester(['A', 'BReqA'], ['A', 'A', 'B'], 'all'),  # second flow dependency reapply
    FlowTester(['A', 'BReqA'], ['A', 'B'], 'none'),  # second flow dependency no reapply
    FlowTester(['A', 'A', 'BReqA'], ['A', 'A', 'A', 'B'], 'all'),  # second flow dependency reapply
    FlowTester(['A', 'A', 'BReqA'], ['A', 'B'], 'none'),  # second flow dependency no reapply
    FlowTester(['A', 'A', 'BReqA'], ['A', 'A', 'B'], 'single'),  # second flow dependency skip requirements
    FlowTester(['A', 'BReqA', 'CReqBReqA'], ['A', 'B', 'C'], 'single'),  # two flows depending on earlier flows
    FlowTester(['A', 'BReqA', 'CReqBReqA'], ['A', 'B', 'C'], 'none'),  # two flows depending on earlier flows
    FlowTester(['A', 'BReqA', 'CReqBReqA'], ['A', 'A', 'B', 'A', 'B', 'C'], 'all'),  # three flows depending on earlier flows
    FlowTester(['CReqBReqA', 'BReqA', 'A'], ['A', 'B', 'C', 'B', 'A'], 'single'),  # three flows depending on earlier flows
    FlowTester(['CReqBReqA', 'BReqA', 'A'], ['A', 'B', 'C'], 'none'),  # three flows depending on earlier flows
    FlowTester(['CReqBReqA', 'BReqA', 'A'], ['A', 'B', 'C', 'A', 'B', 'A'], 'all'),  # three flows depending on earlier flows
    FlowTester(
        ['A', 'CReqBReqA', 'BReqA', 'A'], ['A', 'B', 'C', 'B', 'A'], 'single'
    ),  # three flows depending on earlier flows
    FlowTester(['A', 'CReqBReqA', 'BReqA', 'A'], ['A', 'B', 'C'], 'none'),  # three flows depending on earlier flows
    FlowTester(
        ['A', 'CReqBReqA', 'BReqA', 'A'], ['A', 'A', 'B', 'C', 'A', 'B', 'A'], 'all'
    ),  # three flows depending on earlier flows
]


@pytest.mark.parametrize('tester', flow_tests)
def test_flows(tester):
    success = tester.run()
    i = tester.index
    expected = tester.expected_pass_order
    observed = tester.observed_pass_order
    assert success, f'Tester {i} fails: expected ({expected}), observed ({observed})'
