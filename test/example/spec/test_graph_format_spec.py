import torch

import torcharc

B = 4  # batch size


def test_input_list():
    # list of inputs
    spec_file = torcharc.SPEC_DIR / "graph_format" / "input_list.yaml"
    model = torcharc.build(spec_file)
    x_0 = x_1 = torch.rand(B, 16)
    model(x_0=x_0, x_1=x_1)


def test_input_str():
    # single input
    spec_file = torcharc.SPEC_DIR / "graph_format" / "input_str.yaml"
    model = torcharc.build(spec_file)
    x = torch.rand(B, 16)
    model(x)


def test_modules_dict():
    # transformer module uses dict: src, tgt
    spec_file = torcharc.SPEC_DIR / "graph_format" / "modules_dict.yaml"
    model = torcharc.build(spec_file)
    # shape (batch_size, seq_len, embed_dim)
    src_x = tgt_x = torch.rand(B, 10, 64)
    model(src_x=src_x, tgt_x=tgt_x)


def test_modules_list_multi():
    # transformer module can use list [src, tgt] too
    spec_file = torcharc.SPEC_DIR / "graph_format" / "modules_list_multi.yaml"
    model = torcharc.build(spec_file)
    # shape (batch_size, seq_len, embed_dim)
    src_x = tgt_x = torch.rand(B, 10, 64)
    model(src_x=src_x, tgt_x=tgt_x)


def test_modules_list_single():
    spec_file = torcharc.SPEC_DIR / "graph_format" / "modules_list_single.yaml"
    model = torcharc.build(spec_file)
    x = torch.rand(B, 16)
    model(x)


def test_modules_nested_list():
    # merge uses list of tensors as inputs, so args = ([head_0, head_1],)
    spec_file = torcharc.SPEC_DIR / "graph_format" / "modules_nested_list.yaml"
    model = torcharc.build(spec_file)
    x_0 = x_1 = torch.rand(B, 16)
    model(x_0=x_0, x_1=x_1)


def test_modules_reuse():
    # reuse syntax: <module>~<suffix>. conv is shared for left_image and right_image
    spec_file = torcharc.SPEC_DIR / "graph_format" / "modules_reuse.yaml"
    model = torcharc.build(spec_file)
    left_image = right_image = torch.randn(B, 3, 32, 32)
    model(left_image=left_image, right_image=right_image)


def test_output_dict():
    # use dict for named multi-output
    spec_file = torcharc.SPEC_DIR / "graph_format" / "output_dict.yaml"
    model = torcharc.build(spec_file)
    x = torch.rand(B, 16)
    output = model(x)
    assert isinstance(output, dict)
    assert isinstance(output["y_0"], torch.Tensor)
    assert isinstance(output["y_1"], torch.Tensor)


def test_output_list():
    # use list for multi-output
    spec_file = torcharc.SPEC_DIR / "graph_format" / "output_list.yaml"
    model = torcharc.build(spec_file)
    x = torch.rand(B, 16)
    output = model(x)
    assert isinstance(output, tuple)  # return type is tuple
    assert isinstance(output[0], torch.Tensor)
    assert isinstance(output[1], torch.Tensor)


def test_output_str():
    # use str for single output
    spec_file = torcharc.SPEC_DIR / "graph_format" / "output_str.yaml"
    model = torcharc.build(spec_file)
    x = torch.rand(B, 16)
    output = model(x)
    assert isinstance(output, torch.Tensor)
