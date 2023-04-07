# MGKN-jax

`jax` implementation of [(MGKN) Multipole Graph Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2006.09535). [Original implementation in pytorch](https://github.com/neuraloperator/graph-pde)

## Download Dataset

```bash
./scripts/download_data.sh
```

## Install

```bash
pip install -e .
```

## Run

To run training:
```bash
python main.py
```

The configuration is stored in `MGKN_jax/config.py`. To run training with custom config, add config flags, e.g.:
```bash
python main.py --config.train_cfg.data_cfg.static_grids=False
```
