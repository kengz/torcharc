import torch.fx as fx
from torch import nn


def build_sequential(module_specs: list[dict]) -> nn.Sequential:
    '''
    Build Sequential module from list of module specs, e.g.
    - Linear:
        in_features: 128
        out_features: 64
    - ReLU:
    - Linear:
        in_features: 64
        out_features: 10
    '''
    module = nn.Sequential()
    for module_spec in module_specs:
        module_name = next(iter(module_spec))
        cls = getattr(nn, module_name)
        kwargs = module_spec[module_name] or {}
        layer = cls(**kwargs)
        module.append(layer)
    return module


def build_modules(module_specs: dict) -> dict[str, nn.Module]:
    '''
    Build modules from dict of module specs, e.g.
    head:
      Sequential:
        - Linear:
            in_features: 128
            out_features: 64
        - ReLU:
        - Linear:
            in_features: 64
            out_features: 10
    tail:
      Linear:
        in_features: 10
        out_features: 1
    '''
    modules = {}
    for name, module_spec in module_specs.items():
        # module is always single-key dict
        module_name = next(iter(module_spec))
        cls = getattr(nn, module_name)
        if cls == nn.Sequential:
            module = build_sequential(module_spec[module_name])
        else:
            kwargs = module_spec[module_name] or {}
            module = cls(**kwargs)
        modules[name] = module
    return modules


def build_graph(graph_spec: dict) -> fx.Graph:
    '''
    Build graph from a graph spec, e.g.
    input: [x_1, x_2]
    modules:
      head_1: [x_1]
      head_2: [x_2]
      concat_1_2: [head_1, head_2]
      tail: [concat_1_2]
    output: tail
    '''
    graph = fx.Graph()
    _nodes = {}
    # first, create inputs
    input_spec = graph_spec['input']
    if isinstance(input_spec, str):
        _nodes[input_spec] = graph.placeholder(input_spec)
    elif isinstance(input_spec, list):
        for name in input_spec:
            _nodes[name] = graph.placeholder(name)
    else:
        raise ValueError(f'Invalid input type: {type(input_spec)}')

    # next, create modules
    for name, in_nodes in graph_spec['modules'].items():
        # either list or dict
        if isinstance(in_nodes, list):
            args = tuple([_nodes[name] for name in in_nodes])
            kwargs = {}
        elif isinstance(in_nodes, dict):
            args = ()
            kwargs = {k: _nodes[name] for k, name in in_nodes.items()}
        else:
            raise ValueError(f'Invalid module type: {type(in_nodes)}')
        _nodes[name] = graph.call_module(name, args=args, kwargs=kwargs)

    # finally, create output
    output_spec = graph_spec['output']
    if isinstance(output_spec, str):
        output = _nodes[output_spec]
    elif isinstance(output_spec, list):
        output = tuple([_nodes[name] for name in output_spec])
    elif isinstance(output_spec, dict):
        output = {k: _nodes[name] for k, name in output_spec.items()}
    else:
        raise ValueError(f'Invalid output type: {type(output_spec)}')
    graph.output(output)
    return graph


def build(spec: dict) -> fx.GraphModule:
    '''
    Build a GraphModule from a spec, e.g.
    modules:
      mlp:
        Sequential:
          - Linear:
              in_features: 128
              out_features: 64
          - ReLU:
          - Linear:
              in_features: 64
              out_features: 10

    graph:
      input: x
      modules:
        mlp: [x]
      output: mlp
    '''
    modules = build_modules(spec['modules'])
    graph = build_graph(spec['graph'])
    graph.lint()
    return fx.GraphModule(modules, graph)


if __name__ == '__main__':
    import torch
    import yaml
    spec = yaml.safe_load(open('v2/specs/mlp.yaml'))
    gm = build(spec)
    print(gm.code)
    x = torch.randn(1, 128)
    y = gm(x)
    y
    spec = yaml.safe_load(open('v2/specs/transformer.yaml'))
    gm = build(spec)
    batch_size = 4
    seq_len = 10
    src_x = torch.randn(seq_len, batch_size, 20)
    tgt_x = torch.randn(seq_len, batch_size, 10)
    y = gm(src_x=src_x, tgt_x=tgt_x)
    y.shape

    # TODO attention and RNN has tuple as output. maybe do first with guard
    spec = yaml.safe_load(open('v2/specs/attention.yaml'))

