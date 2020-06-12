# build DAG of nn modules
from torcharc import module_builder, net_util
from torch import nn
from typing import NamedTuple, Union
import pydash as ps
import torch


class DAGNet(nn.Module):
    '''
    DAG network that can flexibly build a DAG of modules, for example HydraNet (multi-head multi-tail network)
    TODO usage guide with info on arc and in_shapes
    NOTE arc must contain the key 'dag_in_shape', which will be popped when building the DAGNet
    '''

    def __init__(self, arc: dict) -> None:
        super().__init__()
        self.dag_in_shape = arc['dag_in_shape']
        self.arc = ps.omit(arc, 'dag_in_shape')

        # build module_dict by inferring in_shape and carry_forward
        xs = net_util.get_rand_tensor(self.dag_in_shape)
        self.module_dict = nn.ModuleDict()
        # set simple arc as dict for generality
        if 'type' in self.arc:
            self.arc = {self.arc['type']: self.arc}
        for name, m_arc in self.arc.items():
            module_builder.infer_in_shape(m_arc, xs)
            module = module_builder.build_module(m_arc)
            xs = module_builder.carry_forward(module, xs, m_arc.get('in_names'))
            self.module_dict.update({name: module})

    def forward(self, xs: Union[torch.Tensor, NamedTuple]) -> Union[torch.Tensor, NamedTuple]:
        # jit.trace will spread args on encountering a namedtuple, thus xs needs to be passed as dict then converted back into namedtuple
        if ps.is_dict(xs):  # guard to convert dict xs into namedtuple
            xs = net_util.to_namedtuple(xs)
        for name, module in self.module_dict.items():
            m_arc = self.arc[name]
            xs = module_builder.carry_forward(module, xs, m_arc.get('in_names'))
        return xs
