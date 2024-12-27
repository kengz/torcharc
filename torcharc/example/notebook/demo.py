import torch
import yaml

import torcharc

spec = yaml.safe_load(open("torcharc/example/spec/mlp.yaml"))
model = torcharc.build(spec)
x = torch.randn(1, 128)
y = model(x)
assert y.shape == (1, 10)

# compat with JIT script, trace, compile
scripted_model = torch.jit.script(model)
assert scripted_model(x).shape == (1, 10)
traced_model = torch.jit.trace(model, (x,))
assert traced_model(x).shape == (1, 10)
compiled_model = torch.compile(model)
compiled_model(x)
assert compiled_model(x).shape == (1, 10)


spec = yaml.safe_load(open("torcharc/example/spec/conv.yaml"))
model = torcharc.build(spec)
x = torch.randn(1, 3, 32, 32)
y = model(x)
assert y.shape == (1, 10)


spec = yaml.safe_load(open("torcharc/example/spec/rnn.yaml"))
model = torcharc.build(spec)
x = torch.randn(4, 10, 7)
y = model(x)
y.shape
assert y.shape == (1, 10)


spec = yaml.safe_load(open("torcharc/example/spec/fn/fn_topk.yaml"))
model = torcharc.build(spec)
x = torch.randn(1, 128)
y = model(x)
compiled_model = torch.compile(model)
compiled_model(x)

spec = yaml.safe_load(open("torcharc/example/spec/merge/concat.yaml"))
model = torcharc.build(spec)
print(model.code)
x = torch.randn(1, 32)
y = model(x, x)
y

spec = yaml.safe_load(open("torcharc/example/spec/fork/chunk.yaml"))
spec = yaml.safe_load(open("torcharc/example/spec/fork/split.yaml"))
model = torcharc.build(spec)
print(model.code)
x = torch.randn(1, 32)
y = model(x)
y

spec = yaml.safe_load(open("torcharc/example/spec/transformer.yaml"))
spec = yaml.safe_load(open("torcharc/example/spec/graph_format/modules_list_multi.yaml"))
model = torcharc.build(spec)
batch_size = 4
seq_len = 10
src_x = torch.randn(seq_len, batch_size, 20)
tgt_x = torch.randn(seq_len, batch_size, 10)
y = model(src_x=src_x, tgt_x=tgt_x)
y.shape

spec = yaml.safe_load(open("torcharc/example/spec/attention.yaml"))
model = torcharc.build(spec)
batch_size = 4
seq_len = 10
src_x = torch.randn(seq_len, batch_size, 20)
tgt_x = torch.randn(seq_len, batch_size, 10)
y = model(src_x=src_x, tgt_x=tgt_x)
y.shape


spec = yaml.safe_load(open("torcharc/example/spec/film_image_text.yaml"))
model = torcharc.build(spec)
print(model.code)
batch_size = 4
im = torch.randn(batch_size, 3, 64, 64)
# use random int - assume tokenized and ready to pass to Embedding
text = torch.randint(0, 1000, (batch_size, 10))
y = model(image=im, text=text)
y.shape

model.graph.print_tabular()
model.print_readable()
cgm = torch.compile(model)
y = cgm(image=im, text=text)
y.shape


spec = yaml.safe_load(open("torcharc/example/spec/stereo_conv.yaml"))
model = torcharc.build(spec)
batch_size = 4
im = torch.randn(batch_size, 3, 64, 64)
y = model(left_image=im, right_image=im)
y.shape


spec = yaml.safe_load(open("torcharc/example/spec/transformer/transformer.yaml"))
model = torcharc.build(spec)
print(model.code)
x = torch.randn(1, 32)
y = model(x, x)
y.shape

spec = yaml.safe_load(open("torcharc/example/spec/transformer/text_summarization.yaml"))
model = torcharc.build(spec)
print(model.code)
vocab_size, embed_dim, num_heads = 10000, 256, 8
source = torch.randint(0, vocab_size, (1, 100))  # 100 source tokens
target = torch.randint(0, vocab_size, (1, 20))   # 20 target tokens

y = model(source, target)
y.shape


spec = yaml.safe_load(open("torcharc/example/spec/transformer/text_classifier.yaml"))
model = torcharc.build(spec)
print(model.code)
vocab_size, embed_dim, num_heads = 10000, 256, 8
source = torch.randint(0, vocab_size, (1, 100))  # 100 source tokens

y = model(source)
y.shape


spec = yaml.safe_load(open("torcharc/example/spec/advanced/dlrm_sum.yaml"))
spec = yaml.safe_load(open("torcharc/example/spec/advanced/dlrm_attn.yaml"))
model = torcharc.build(spec)
print(model.code)

batch_size = 4
dense = torch.randn(batch_size, 256)
cat_0 = torch.randint(0, 1000, (batch_size,))
cat_1 = torch.randint(0, 1000, (batch_size,))
cat_2 = torch.randint(0, 1000, (batch_size,))
cat_2.shape
y = model(dense, cat_0, cat_1, cat_2)
y.shape


spec = yaml.safe_load(open("torcharc/example/spec/advanced/film_image_state.yaml"))
model = torcharc.build(spec)
batch_size = 4
image = torch.randn(batch_size, 3, 64, 64)
state = torch.randn(batch_size, 4)
y = model(image, state)

sgm = torch.jit.script(model)
result = sgm(image, state)
