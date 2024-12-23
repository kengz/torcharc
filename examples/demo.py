import torch
import yaml
import torcharc

spec = yaml.safe_load(open("torcharc/specs/mlp.yaml"))
gm = torcharc.build(spec)
print(gm.code)
x = torch.randn(1, 128)
y = gm(x)
y

spec = yaml.safe_load(open("torcharc/specs/mlp_lazy.yaml"))
spec = yaml.safe_load(open("torcharc/specs/fork/chunk.yaml"))
spec = yaml.safe_load(open("torcharc/specs/fork/split.yaml"))
gm = torcharc.build(spec)
print(gm.code)
x = torch.randn(1, 32)
y = gm(x)
y
spec = yaml.safe_load(open("torcharc/specs/transformer.yaml"))
gm = torcharc.build(spec)
batch_size = 4
seq_len = 10
src_x = torch.randn(seq_len, batch_size, 20)
tgt_x = torch.randn(seq_len, batch_size, 10)
y = gm(src_x=src_x, tgt_x=tgt_x)
y.shape

spec = yaml.safe_load(open("torcharc/specs/attention.yaml"))
gm = torcharc.build(spec)
batch_size = 4
seq_len = 10
src_x = torch.randn(seq_len, batch_size, 20)
tgt_x = torch.randn(seq_len, batch_size, 10)
y = gm(src_x=src_x, tgt_x=tgt_x)
y.shape
