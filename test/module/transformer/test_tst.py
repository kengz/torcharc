from torcharc import arc_ref, module_builder, net_util
import pytest
import torch


@pytest.mark.parametrize('seq_len', [16, 32])
@pytest.mark.parametrize('in_channels', [1, 5])
@pytest.mark.parametrize('out_channels', [1, 5])
def test_tst(seq_len, in_channels, out_channels):
    arc = arc_ref.REF_ARCS['tstransformer']
    arc = {**arc, **{'in_channels': in_channels, 'out_channels': out_channels}}
    in_shapes = [seq_len, in_channels]
    xs = net_util.get_rand_tensor(in_shapes)
    model = module_builder.build_module(arc)
    ys = model(xs)
    assert isinstance(ys, torch.Tensor)
    assert ys.shape == torch.Size([xs.shape[0], seq_len, out_channels])
