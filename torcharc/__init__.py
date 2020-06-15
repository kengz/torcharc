from torcharc import module_builder
from torcharc.net_util import to_namedtuple
from torcharc.module import dag
from torch import nn


def build(arc: dict) -> nn.Module:
    '''Interface method to build a DAGNet or a simple nn module. See arf_ref.py for arc references.'''
    if 'dag_in_shape' in arc:
        return dag.DAGNet(arc)
    else:
        return module_builder.build_module(arc)
