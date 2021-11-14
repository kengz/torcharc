# build neural networks modularly
from torch import nn
from torcharc import optim
from torcharc.module import fork, merge, sequential
from torcharc.module.perceiver_io import perceiver
from typing import Callable, List, Optional, Union
import inspect
import pydash as ps
import torch


# register custom classes
setattr(torch.nn, 'ReuseFork', fork.ReuseFork)
setattr(torch.nn, 'SplitFork', fork.SplitFork)
setattr(torch.nn, 'ConcatMerge', merge.ConcatMerge)
setattr(torch.nn, 'FiLMMerge', merge.FiLMMerge)
setattr(torch.nn, 'Perceiver', perceiver.Perceiver)

setattr(torch.optim, 'GlobalAdam', optim.GlobalAdam)
setattr(torch.optim, 'GlobalRMSprop', optim.GlobalRMSprop)
setattr(torch.optim, 'Lookahead', optim.Lookahead)
setattr(torch.optim, 'RAdam', optim.RAdam)


def get_init_fn(init: Union[str, dict], activation: Optional[str] = None) -> Callable:
    '''Get init function that can be called as `module.apply(init_fn)`. Initializes weights only. Internally this also takes care of gain and nonlinearity args. Ref: https://pytorch.org/docs/stable/nn.init.html'''
    def init_fn(module: nn.Module) -> None:
        if init is None:
            return
        elif isinstance(init, str):
            init_type = init
            init_kwargs = {}
        else:
            assert isinstance(init, dict)
            init_type = init['type']
            init_kwargs = ps.omit(init, 'type')
        fn = getattr(nn.init, init_type)
        args = inspect.getfullargspec(fn).args
        try:
            try:
                # first try with gain/activation args
                if 'gain' in args:
                    gain = nn.init.calculate_gain(activation)
                    ext_init_kwargs = {'gain': gain, **init_kwargs}
                    fn(module.weight, **ext_init_kwargs)
                elif 'nonlinearity' in args:
                    ext_init_kwargs = {'nonlinearity': activation, **init_kwargs}
                    fn(module.weight, **ext_init_kwargs)
                else:
                    fn(module.weight, **init_kwargs)
            except Exception:  # first fallback to plain init
                fn(module.weight, **init_kwargs)
        except Exception:  # second fallback: module weight cannot be initialized, ok
            pass
    return init_fn


def build_module(arc: dict) -> nn.Module:
    '''The core method to build nn module of any type given arc in which "type" is the case-insensitive class name and the remainder of arc is the kwargs'''
    if arc.get('layers'):  # if given layers, build as sequential
        module = sequential.build(arc)
    else:
        kwargs = ps.omit(arc, 'type', 'in_names', 'init')
        module = getattr(nn, arc['type'])(**kwargs)
    # initialize weights if 'init' is given
    if arc.get('init'):
        module.apply(get_init_fn(arc.get('init'), arc.get('activation')))
    return module


def infer_in_shape(arc: dict, xs: Union[torch.Tensor, dict]) -> None:
    '''Infer the input shape(s) for arc depending on its type and the input tensor. This updates the arc with the appropriate key.'''
    nn_type = arc['type']
    if nn_type == 'Linear':
        if isinstance(xs, dict):
            in_names = arc.get('in_names', list(xs)[:1])
            xs = xs[in_names[0]]
        assert isinstance(xs, torch.Tensor)
        assert len(xs.shape) == 2, f'xs shape {xs.shape} is not meant for {nn_type} layer'
        in_features = xs.shape[1]
        arc.update(in_features=in_features)
    elif nn_type.startswith('Conv') or nn_type == 'Perceiver':
        if isinstance(xs, dict):
            in_names = arc.get('in_names', list(xs)[:1])
            xs = xs[in_names[0]]
        assert isinstance(xs, torch.Tensor)
        assert len(xs.shape) >= 2, f'xs shape {xs.shape} is not meant for {nn_type} layer'
        in_shape = list(xs.shape)[1:]
        arc.update(in_shape=in_shape)
    elif nn_type == 'FiLMMerge':
        assert isinstance(xs, dict)
        assert len(arc['in_names']) == 2, 'FiLMMerge in_names should only specify 2 keys for feature and conditioner'
        shapes = {name: list(x.shape)[1:] for name, x in xs.items() if name in arc['in_names']}
        arc.update(shapes=shapes)
    else:
        pass


def carry_forward(module: nn.Module, xs: Union[torch.Tensor, dict], in_names: Optional[List[str]] = None) -> Union[torch.Tensor, dict]:
    '''
    Main method to call module.forward and handle tensor and dict input/output
    If xs and ys are tensors, forward as usual
    If xs or ys is dict, then arc.in_names must specify the inputs names to be used in forward, and any unused names will be carried with the output, which will be dict.
    '''
    if isinstance(xs, dict):
        if in_names is None:  # use the first by default
            in_names = list(xs)[:1]

        if len(in_names) == 1:  # single input is tensor
            m_xs = xs[in_names[0]]
        else:  # multi input is dict of tensors
            m_xs = {name: xs[name] for name in in_names}

        ys = module(m_xs)

        # any unused_xs must be carried with the output as dict
        d_xs = xs
        unused_d_xs = ps.omit(d_xs, in_names)
        if unused_d_xs:
            if isinstance(ys, dict):
                d_ys = {**ys, **unused_d_xs}
            else:  # when formed as dict, single output will use the first of in_names
                d_ys = {**{in_names[0]: ys}, **unused_d_xs}
            ys = d_ys
    else:
        ys = module(xs)
    return ys
