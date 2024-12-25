import torch
import yaml

import torcharc

spec = yaml.safe_load(open("torcharc/example/spec/mlp.yaml"))
spec = yaml.safe_load(open("torcharc/example/spec/mlp_lazy.yaml"))
gm = torcharc.build(spec)
print(gm.code)
x = torch.randn(1, 128)
y = gm(x)
y


spec = yaml.safe_load(open("torcharc/example/spec/merge/concat.yaml"))
gm = torcharc.build(spec)
print(gm.code)
x = torch.randn(1, 32)
y = gm(x, x)
y

spec = yaml.safe_load(open("torcharc/example/spec/fork/chunk.yaml"))
spec = yaml.safe_load(open("torcharc/example/spec/fork/split.yaml"))
gm = torcharc.build(spec)
print(gm.code)
x = torch.randn(1, 32)
y = gm(x)
y

spec = yaml.safe_load(open("torcharc/example/spec/transformer.yaml"))
spec = yaml.safe_load(open("torcharc/example/spec/graph_format/modules_list_multi.yaml"))
gm = torcharc.build(spec)
batch_size = 4
seq_len = 10
src_x = torch.randn(seq_len, batch_size, 20)
tgt_x = torch.randn(seq_len, batch_size, 10)
y = gm(src_x=src_x, tgt_x=tgt_x)
y.shape

spec = yaml.safe_load(open("torcharc/example/spec/attention.yaml"))
gm = torcharc.build(spec)
batch_size = 4
seq_len = 10
src_x = torch.randn(seq_len, batch_size, 20)
tgt_x = torch.randn(seq_len, batch_size, 10)
y = gm(src_x=src_x, tgt_x=tgt_x)
y.shape


spec = yaml.safe_load(open("torcharc/example/spec/film_image_text.yaml"))
gm = torcharc.build(spec)
batch_size = 4
im = torch.randn(batch_size, 3, 64, 64)
# use random int - assume tokenized and ready to pass to Embedding
text = torch.randint(0, 1000, (batch_size, 10))
y = gm(image=im, text=text)
y.shape

gm.graph.print_tabular()
gm.print_readable()
cgm = torch.compile(gm)
y = cgm(image=im, text=text)
y.shape


spec = yaml.safe_load(open("torcharc/example/spec/stereo_conv.yaml"))
gm = torcharc.build(spec)
batch_size = 4
im = torch.randn(batch_size, 3, 64, 64)
y = gm(left_image=im, right_image=im)
y.shape
