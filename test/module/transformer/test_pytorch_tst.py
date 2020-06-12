from torcharc import arc_ref, module_builder, net_util
import pytest
import torch


@pytest.mark.parametrize('seq_len', [16, 32])
@pytest.mark.parametrize('in_channels', [1, 5])
@pytest.mark.parametrize('out_channels', [1, 5])
def test_pytorch_tst(seq_len, in_channels, out_channels):
    arc = arc_ref.REF_ARCS['pytorch_tstransformer']
    in_shapes = [seq_len, in_channels]
    xs = net_util.get_rand_tensor(in_shapes)
    model = module_builder.build_module({**arc, **{'in_channels': in_channels, 'out_channels': out_channels}})
    ys = model(xs)
    assert isinstance(ys, torch.Tensor)
    assert ys.shape == torch.Size([xs.shape[0], seq_len, out_channels])
