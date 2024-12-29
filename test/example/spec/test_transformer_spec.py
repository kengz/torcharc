import torch

import torcharc

B = 4  # batch size


def test_attn():
    # attention module with q=src, k=tgt, v=tgt
    spec_file = torcharc.SPEC_DIR / "transformer" / "attn.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    src_x = torch.rand(B, 10, 64)  # (batch_size, seq_len, embed_dim)
    tgt_x = torch.rand(B, 20, 64)  # (batch_size, seq_len, embed_dim)
    y = model(src_x=src_x, tgt_x=tgt_x)
    assert y.shape == (B, 10, 10)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(src_x, tgt_x).shape == y.shape
    traced_model = torch.jit.trace(model, (src_x, tgt_x))
    assert traced_model(src_x, tgt_x).shape == y.shape


def test_text_classifier():
    # attention for text classifier example
    spec_file = torcharc.SPEC_DIR / "transformer" / "text_classifier.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    vocab_size = 10000
    tokens = torch.randint(0, vocab_size, (B, 10))  # 10 src tokens
    y = model(tokens)
    assert y.shape == (B, 10)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(tokens).shape == y.shape
    traced_model = torch.jit.trace(model, (tokens))
    assert traced_model(tokens).shape == y.shape


def test_text_summarizer():
    # attention for text summarizer example
    spec_file = torcharc.SPEC_DIR / "transformer" / "text_summarizer.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    vocab_size = 10000
    src = torch.randint(0, vocab_size, (B, 10))  # 10 src tokens
    tgt = torch.randint(0, vocab_size, (B, 20))  # 20 tgt tokens
    y = model(src=src, tgt=tgt)
    assert y.shape == (B, 20, vocab_size)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(src, tgt).shape == y.shape
    traced_model = torch.jit.trace(model, (src, tgt))
    assert traced_model(src, tgt).shape == y.shape


def test_transformer():
    # transformer with src, tgt
    spec_file = torcharc.SPEC_DIR / "transformer" / "transformer.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    src_x = torch.rand(B, 10, 64)  # (batch_size, seq_len, embed_dim)
    tgt_x = torch.rand(B, 20, 64)  # (batch_size, seq_len, embed_dim)
    y = model(src_x=src_x, tgt_x=tgt_x)
    assert y.shape == (B, 20, 10)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(src_x, tgt_x).shape == y.shape
    traced_model = torch.jit.trace(model, (src_x, tgt_x))
    assert traced_model(src_x, tgt_x).shape == y.shape
