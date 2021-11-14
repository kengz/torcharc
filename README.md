# TorchArc ![CI](https://github.com/kengz/torcharc/workflows/CI/badge.svg)
Build PyTorch networks by specifying architectures.

## Installation

```bash
pip install torcharc
```

Bring your own PyTorch: this allows you to use the version of PyTorch as needed - GPU/CPU, nightly etc. For example:

```bash
conda install pytorch -c pytorch
```

## Usage

Given just the architecture, `torcharc` can build generic DAG (directed acyclic graph) of nn modules, which consists of:
- single-input-output modules: `Conv1d, Conv2d, Conv3d, Linear, Perceiver` or any other valid nn.Module
- fork modules: `ReuseFork, SplitFork`
- merge modules: `ConcatMerge, FiLMMerge`

The custom modules are defined in [`torcharc/module`](https://github.com/kengz/torcharc/tree/master/torcharc/module), registered in [`torcharc/module_builder.py`](https://github.com/kengz/torcharc/blob/master/torcharc/module_builder.py).


The full examples of architecture references are in [`torcharc/arc_ref.py`](https://github.com/kengz/torcharc/blob/master/torcharc/arc_ref.py), and full functional examples are in [`test/module/`](https://github.com/kengz/torcharc/tree/master/test/module). Below we walk through some main examples.


### ConvNet

```python
import torcharc


arc = {
    'type': 'Conv2d',
    'in_shape': [3, 20, 20],
    'layers': [
        [16, 4, 2, 0, 1],
        [16, 4, 1, 0, 1]
    ],
    'batch_norm': True,
    'activation': 'ReLU',
    'dropout': 0.2,
    'init': 'kaiming_uniform_',
}
model = torcharc.build(arc)

batch_size = 16
x = torch.rand([batch_size, *arc['in_shape']])
y = model(x)
```

<details><summary>model</summary>
<p>

```
Sequential(
  (0): Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2))
  (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
  (3): Dropout2d(p=0.2, inplace=False)
  (4): Conv2d(16, 16, kernel_size=(4, 4), stride=(1, 1))
  (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): ReLU()
  (7): Dropout2d(p=0.2, inplace=False)
)
```

</p>
</details>


### MLP

```python
arc = {
    'type': 'Linear',
    'in_features': 8,
    'layers': [64, 32],
    'batch_norm': True,
    'activation': 'ReLU',
    'dropout': 0.2,
    'init': {
      'type': 'normal_',
      'std': 0.01,
    },
}
model = torcharc.build(arc)

batch_size = 16
x = torch.rand([batch_size, arc['in_features']])
y = model(x)
```

<details><summary>model</summary>
<p>

```
Sequential(
  (0): Linear(in_features=8, out_features=64, bias=True)
  (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
  (3): Dropout(p=0.2, inplace=False)
  (4): Linear(in_features=64, out_features=32, bias=True)
  (5): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): ReLU()
  (7): Dropout(p=0.2, inplace=False)
)
```

</p>
</details>


### Perceiver

```python
arc = {
    'type': 'Perceiver',
    'in_shape': [64, 64, 3],
    'arc': {
        'preprocessor': {
            'type': 'FourierPreprocessor',
            'num_freq_bands': 32,
            'cat_pos': True,
        },
        'encoder': {
            'type': 'PerceiverEncoder',
            'latent_shape': [4, 11],
            'head_dim': 32,
            'v_head_dim': None,
            'cross_attn_num_heads': 1,
            'cross_attn_widening_factor': 1,
            'num_self_attn_blocks': 8,
            'num_self_attn_per_block': 6,
            'self_attn_num_heads': 8,
            'self_attn_widening_factor': 1,
            'dropout_p': 0.0,
        },
        'decoder': {
            'type': 'PerceiverDecoder',
            'out_shape': [1, 16],
            'head_dim': 32,
            'v_head_dim': None,
            'cross_attn_num_heads': 1,
            'cross_attn_widening_factor': 1,
            'dropout_p': 0.0,
        },
        'postprocessor': {
            'type': 'ClassificationPostprocessor',
            'out_dim': 10,
        }
    }
}
model = torcharc.build(arc)

seq_len = 32
x = torch.rand([seq_len, *arc['in_shape']])
```

<details><summary>model</summary>
<p>

```
Perceiver(
  (module): SpreadSequential(
    (0): FourierPreprocessor()
    (1): PerceiverEncoder(
      (encoder_processor): SpreadSequential(
        (0): SpreadSequential(
          (0): Residual(
            (module): CrossAttention(
              (attn): Attention(
                (to_flat_q): Linear(in_features=11, out_features=32, bias=True)
                (to_flat_k): Linear(in_features=133, out_features=32, bias=True)
                (to_flat_v): Linear(in_features=133, out_features=32, bias=True)
                (to_z): Linear(in_features=32, out_features=11, bias=True)
              )
              (embed_layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              (context_layer_norm): LayerNorm((133,), eps=1e-05, elementwise_affine=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (1): Residual(
            (module): TransformerMLP(
              (module): Sequential(
                (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                (1): Linear(in_features=11, out_features=11, bias=True)
                (2): GELU()
                (3): Linear(in_features=11, out_features=11, bias=True)
              )
            )
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
        (1): Sequential(
          (0): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (2): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (3): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (4): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (5): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (2): Sequential(
          (0): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (2): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (3): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (4): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (5): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (3): Sequential(
          (0): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (2): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (3): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (4): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (5): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (4): Sequential(
          (0): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (2): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (3): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (4): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (5): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (5): Sequential(
          (0): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (2): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (3): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (4): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (5): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (6): Sequential(
          (0): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (2): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (3): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (4): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (5): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (7): Sequential(
          (0): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (2): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (3): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (4): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (5): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (8): Sequential(
          (0): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (1): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (2): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (3): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (4): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (5): Sequential(
            (0): Residual(
              (module): SelfAttention(
                (attn): Attention(
                  (to_flat_q): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_k): Linear(in_features=11, out_features=256, bias=True)
                  (to_flat_v): Linear(in_features=11, out_features=256, bias=True)
                  (to_z): Linear(in_features=256, out_features=11, bias=True)
                )
                (layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (1): Residual(
              (module): TransformerMLP(
                (module): Sequential(
                  (0): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
                  (1): Linear(in_features=11, out_features=11, bias=True)
                  (2): GELU()
                  (3): Linear(in_features=11, out_features=11, bias=True)
                )
              )
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    (2): PerceiverDecoder(
      (decoder): SpreadSequential(
        (0): Residual(
          (module): CrossAttention(
            (attn): Attention(
              (to_flat_q): Linear(in_features=16, out_features=32, bias=True)
              (to_flat_k): Linear(in_features=11, out_features=32, bias=True)
              (to_flat_v): Linear(in_features=11, out_features=32, bias=True)
              (to_z): Linear(in_features=32, out_features=16, bias=True)
            )
            (embed_layer_norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
            (context_layer_norm): LayerNorm((11,), eps=1e-05, elementwise_affine=True)
          )
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (1): Residual(
          (module): TransformerMLP(
            (module): Sequential(
              (0): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
              (1): Linear(in_features=16, out_features=16, bias=True)
              (2): GELU()
              (3): Linear(in_features=16, out_features=16, bias=True)
            )
          )
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (3): ClassificationPostprocessor(
      (linear): Linear(in_features=16, out_features=10, bias=True)
    )
  )
)
```

</p>
</details>


### DAG: Hydra

Ultimately, we can build a generic DAG network using the modules linked by the fork and merge modules. The example below shows HydraNet - a network with multiple inputs and multiple outputs.

```python
arc = {
    'dag_in_shape': {'image': [3, 20, 20], 'vector': [8]},
    'image': {
        'type': 'Conv2d',
        'in_names': ['image'],
        'layers': [
            [16, 4, 2, 0, 1],
            [16, 4, 1, 0, 1]
        ],
        'batch_norm': True,
        'activation': 'ReLU',
        'dropout': 0.2,
        'init': 'kaiming_uniform_',
    },
    'merge': {
        'type': 'FiLMMerge',
        'in_names': ['image', 'vector'],
        'names': {'feature': 'image', 'conditioner': 'vector'},
    },
    'Flatten': {
        'type': 'Flatten'
    },
    'Linear': {
        'type': 'Linear',
        'layers': [64, 32],
        'batch_norm': True,
        'activation': 'ReLU',
        'dropout': 0.2,
        'init': 'kaiming_uniform_',
    },
    'out': {
        'type': 'Linear',
        'out_features': 8,
    },
    'fork': {
        'type': 'SplitFork',
        'shapes': {'mean': [4], 'std': [4]},
    }
}
model = torcharc.build(arc)

batch_size = 16
dag_in_shape = arc['dag_in_shape']
xs = {'image': torch.rand([batch_size, *dag_in_shape['image']]), 'vector': torch.rand([batch_size, *dag_in_shape['vector']])}
# returns dict if output is multi-model, Tensor otherwise
ys = model(xs)
```

<details><summary>model</summary>
<p>

```
DAGNet(
  (module_dict): ModuleDict(
    (image): Sequential(
      (0): Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout2d(p=0.2, inplace=False)
      (4): Conv2d(16, 16, kernel_size=(4, 4), stride=(1, 1))
      (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU()
      (7): Dropout2d(p=0.2, inplace=False)
    )
    (merge): FiLMMerge(
      (conditioner_scale): Linear(in_features=8, out_features=16, bias=True)
      (conditioner_shift): Linear(in_features=8, out_features=16, bias=True)
    )
    (Flatten): Flatten()
    (Linear): Sequential(
      (0): Linear(in_features=576, out_features=64, bias=True)
      (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.2, inplace=False)
      (4): Linear(in_features=64, out_features=32, bias=True)
      (5): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU()
      (7): Dropout(p=0.2, inplace=False)
    )
    (out): Linear(in_features=32, out_features=8, bias=True)
    (fork): SplitFork()
  )
)
```

</p>
</details>

DAG module accepts a `dict` (example below) as input, and the module selects its input by matching its own name in the arc and the `in_name`, then carry forward the output together with any unconsumed inputs.

For example, the input `xs` with keys `image, vector` passes through the first `image` module, and the output becomes `{'image': image_module(xs.image), 'vector': xs.vector}`. This is then passed through the remainder of the modules in the arc as declared.

## Development

### Setup

```bash
# install the dev dependencies
bin/setup
# activate Conda environment
conda activate torcharc
# install PyTorch
conda install pytorch -c pytorch
```

### Unit Tests

```bash
python setup.py test
```
