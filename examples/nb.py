import torch
import torch.fx as fx
# yaml to read nb.yaml
import yaml

spec = yaml.safe_load(open("examples/nb.yaml"))

# example MLP, using sequential
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 64)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, 10)
        self.model = torch.nn.Sequential(self.linear1, self.relu, self.linear2)

    def forward(self, x):
        return self.model(x)
    
# test
x = torch.randn(128)
model = MLP()
y = model(x)
y

# turn this into YAML that can be iterated to construct the model
yaml_str = """
- Linear:
    in_features: 128
    out_features: 64
- ReLU: {}
- Linear:
    in_features: 64
    out_features: 10
"""

# parse the yaml string
spec = yaml.safe_load(yaml_str)
# create a new model
model = torch.nn.Sequential()
# iterate over the yaml spec
for layer_spec in spec:
    # get the class name
    class_name = next(iter(layer_spec))
    # get the class
    cls = getattr(torch.nn, class_name)
    # create an instance of the class
    layer = cls(**layer_spec[class_name])
    # add the layer to the model
    model.add_module(str(len(model)), layer)

# test
x = torch.randn(128)
y = model(x)



import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node

# Define your modules
class ModuleA(nn.Module):
    def forward(self, x):
        return x * 2

class ModuleB(nn.Module):
    def forward(self, x):
        return x + 1

class ModuleC(nn.Module):
    def forward(self, x):
        return x - 1

# Create instances
A = ModuleA()
B = ModuleB()
C = ModuleC()

# Build a computation graph
graph = Graph()
input_node = graph.placeholder('x')  # Input node
a_output = graph.call_module('A', args=(input_node,))  # A(x)
b_output = graph.call_module('B', args=(a_output,))    # B(A(x))
c_output = graph.call_module('C', args=(a_output,))    # C(A(x))
graph.output((b_output, c_output))  # Output both B's and C's outputs

# Create a GraphModule
modules = {'A': A, 'B': B, 'C': C}
graph_module = GraphModule(modules, graph)
graph_module

# Test the GraphModule
x = torch.tensor([1.0])  # Input tensor
outputs = graph_module(x)
print(f"B's Output: {outputs[0]}")
print(f"C's Output: {outputs[1]}")


# use this to redo the MLP example above
linear1 = nn.Linear(128, 64)
relu = nn.ReLU()
linear2 = nn.Linear(64, 10)

graph = Graph()
input_node = graph.placeholder('x')
node = input_node
# node = graph.call_module('linear1', (node, linear1))
# node = graph.call_module('relu', (node, relu))
# node = graph.call_module('linear2', (node, linear2))
# programmatically determine the name
# ahh code construction determines the name
modules = {}
# use name with suffix by count if name already exists
for i, module in enumerate([linear1, relu, linear2]):
    name = module.__class__.__name__.lower()
    if name in modules:
        name = f"{name}{i}"
    modules[name] = module
    node = graph.call_module(name, (node,))

graph.output(node)
graph.lint()
graph.print_tabular()
gm = GraphModule(modules, graph)
gm

# test
x = torch.randn(128)
y = gm(x)
y
