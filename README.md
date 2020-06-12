# TorchArc
TorchArc: Build PyTorch networks by specifying architectures.

## Installation

```bash
pip install torcharc
# to use the time-series transformer model
pip install git+https://github.com/kengz/transformer.git
```

Bring your own PyTorch: this allows you to use the version of PyTorch as needed - GPU/CPU, nightly etc. For example:

```bash
conda install pytorch -c pytorch
```

## Usage

TODO show example arcs from ref and tests here

```bash
import torcharc

torcharc.build(arc)
```

## Development

### Setup

```bash
# install the dev dependencies
bin/setup
# activate Conda environment
conda activate torcharc
# install PyTorch
conda install pytorch -c pytorch
# to use the time-series transformer model
pip install git+https://github.com/kengz/transformer.git
```

### Unit Tests

```bash
python setup.py test
```
