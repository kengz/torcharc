from pathlib import Path

import torch
import yaml

import torcharc

# ================================================
# Example: build model from spec file

filepath = Path(".") / "torcharc" / "example" / "spec" / "basic" / "mlp.yaml"

# The following are equivalent:

# 1. build from YAML spec file
model = torcharc.build(filepath)

# 2. build from dictionary
with filepath.open("r") as f:
    spec_dict = yaml.safe_load(f)
model = torcharc.build(spec_dict)

# 3. use the underlying Pydantic validator to build the model
spec = torcharc.Spec(**spec_dict)
model = spec.build()


# ================================================
# Example: basic MLP
model = torcharc.build(torcharc.SPEC_DIR / "basic" / "mlp.yaml")
assert isinstance(model, torch.nn.Module)

# Run the model and check the output shape
x = torch.randn(4, 128)
y = model(x)
assert y.shape == (4, 10)

# Test compatibility with compile, script and trace
compiled_model = torch.compile(model)
assert compiled_model(x).shape == y.shape
scripted_model = torch.jit.script(model)
assert scripted_model(x).shape == y.shape
traced_model = torch.jit.trace(model, (x))
assert traced_model(x).shape == y.shape

model
# GraphModule(
#   (mlp): Sequential(
#     (0): Linear(in_features=128, out_features=64, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=64, out_features=10, bias=True)
#   )
# )

# ================================================
# Example: MLP (Lazy)
# PyTorch Lazy layers will infer the input size from the first forward pass. This is recommended as it greatly simplifies the model definition.
model = torcharc.build(torcharc.SPEC_DIR / "basic" / "mlp_lazy.yaml")
model  # shows LazyLinear before first forward pass
# GraphModule(
#   (mlp): Sequential(
#     (0): LazyLinear(in_features=0, out_features=64, bias=True)
#     (1): ReLU()
#     (2): LazyLinear(in_features=0, out_features=10, bias=True)
#   )
# )

# Run the model and check the output shape
x = torch.randn(4, 128)
y = model(x)
assert y.shape == (4, 10)

# Because it is lazy - wait till first forward pass to run compile, script or trace
compiled_model = torch.compile(model)

model  # shows Linear after first forward pass
# GraphModule(
#   (mlp): Sequential(
#     (0): Linear(in_features=128, out_features=64, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=64, out_features=10, bias=True)
#   )
# )


# ================================================
# Example: MNIST conv
model = torcharc.build(torcharc.SPEC_DIR / "mnist" / "conv.yaml")

# Run the model and check the output shape
x = torch.randn(4, 1, 28, 28)
y = model(x)
assert y.shape == (4, 10)

model


# ================================================
# Example: Reuse syntax: Stereo Conv
# To use the same module for many passes, specify with the reuse syntax in graph spec: <module>~<suffix>.
# E.g. stereoscopic model with `left` and `right` inputs over a shared `conv` module: `conv~left` and `conv~right`.
model = torcharc.build(torcharc.SPEC_DIR / "basic" / "stereo_conv_reuse.yaml")

# Run the model and check the output shape
left_image = right_image = torch.randn(4, 3, 32, 32)
left, right = model(left_image=left_image, right_image=right_image)
assert left.shape == (4, 10)
assert right.shape == (4, 10)

model


# ================================================
# Example: transformer
model = torcharc.build(torcharc.SPEC_DIR / "transformer" / "transformer.yaml")

# Run the model and check the output shape
src_x = torch.randn(4, 10, 64)
tgt_x = torch.randn(4, 20, 64)
y = model(src_x=src_x, tgt_x=tgt_x)
assert y.shape == (4, 20, 10)

model


# ================================================
# Example: Get modules (torcharc/module/get.py): Attention
# Some cases need "Get" module, e.g. get first output from MultiheadAttention
model = torcharc.build(torcharc.SPEC_DIR / "transformer" / "attn.yaml")

# Run the model and check the output shape
src_x = torch.rand(4, 10, 64)  # (batch_size, seq_len, embed_dim)
tgt_x = torch.rand(4, 20, 64)  # (batch_size, seq_len, embed_dim)
# Get module gets the first output from MultiheadAttention
y = model(src_x=src_x, tgt_x=tgt_x)
assert y.shape == (4, 10, 10)

model


# ================================================
# Example: Merge modules (torcharc/module/merge.py): DLRM
# For more complex models, merge modules can be used to combine multiple inputs into a single output.
model = torcharc.build(torcharc.SPEC_DIR / "advanced" / "dlrm_sum.yaml")

# Run the model and check the output shape
dense = torch.randn(4, 256)
cat_0 = torch.randint(0, 1000, (4,))
cat_1 = torch.randint(0, 1000, (4,))
cat_2 = torch.randint(0, 1000, (4,))
y = model(dense, cat_0, cat_1, cat_2)
assert y.shape == (4, 1)

model


# ================================================
# Example: Merge modules: FiLM
# Custom MergeFiLM module for Feature-wise Linear Modulation https://distill.pub/2018/feature-wise-transformations/
model = torcharc.build(torcharc.SPEC_DIR / "advanced" / "film_image_state.yaml")

# Run the model and check the output shape
image = torch.randn(4, 3, 32, 32)
state = torch.randn(4, 4)
y = model(image=image, state=state)
assert y.shape == (4, 10)

model


# ================================================
# Example: Fork modules (torcharc/module/fork.py): Chunk
# Fork module can be used to split the input into multiple outputs.
model = torcharc.build(torcharc.SPEC_DIR / "fork" / "chunk.yaml")

# Run the model and check the output shape
x = torch.randn(4, 32)
y = model(x)
assert len(y) == 2  # tail_0, tail_1

model


# ================================================
# Example: reduction op functional modules (torcharc/module/fn.py): Reduce
# Sometimes we need to reduce the input tensor, e.g. pooling token embeddings into a single sentence embedding
model = torcharc.build(torcharc.SPEC_DIR / "fn" / "reduce_mean.yaml")

# Run the model and check the output shape
text = torch.randint(0, 1000, (4, 10))
y = model(text)
assert y.shape == (4, 128)  # reduced embedding

model


# ================================================
# Example: general functional modules (torcharc/module/fn.py): TorchFn
# While TorchArc uses module-based design, not all torch functions are available as modules. Use TorchFn to wrap any torch function into a module - with the caveat that it does not work with JIT script (but does work with torch compile and JIT trace).
model = torcharc.build(torcharc.SPEC_DIR / "fn" / "fn_topk.yaml")

# Run the model and check the output shape
x = torch.randn(4, 128)
y = model(x)  # topk output with k=3
assert y.indices.shape == (4, 3)
assert y.values.shape == (4, 3)

model


# ================================================
# See more examples in spec files torcharc/example/spec/ and unit tests test/example/spec/
