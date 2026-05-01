# Remaining Useful Life Prediction on NASA C-MAPSS FD001

This repository contains the code, checkpoints, plots, and report assets for the 24-788 mini-project on remaining useful life (RUL) prediction using the NASA C-MAPSS FD001 dataset.

## Project setup

- **Dataset:** NASA C-MAPSS FD001
- **Task:** Predict engine remaining useful life from multivariate sensor windows
- **Baseline:** LSTM
- **Variant:** Temporal Convolutional Network (TCN)

## Included results from the final run

| Model | Validation RMSE | Validation PHM | Test RMSE | Test PHM |
|---|---:|---:|---:|---:|
| LSTM | 13.62 | 15574.43 | 14.89 | 369.19 |
| TCN | 16.87 | 24417.29 | 17.46 | 542.45 |

Lower is better for both RMSE and PHM score.

## Repository contents

- `train.py` - training script for either model
- `reproduce_results.py` - reloads a saved checkpoint and regenerates metrics and figures
- `requirements.txt` - Python dependencies
- `runs/lstm/` - final LSTM checkpoint, metrics, and plots
- `runs/tcn/` - final TCN checkpoint, metrics, and plots
- `cmapss_colab_starter.ipynb` - Colab notebook version

## Environment

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset download

The scripts automatically download the official NASA C-MAPSS archive and extract the FD001 files into `data/CMAPSS/` the first time you run them.

## Training commands

Run the baseline:

```bash
python train.py --model lstm --window_size 30 --epochs 30 --batch_size 128 --output_dir runs/lstm
```

Run the variant:

```bash
python train.py --model tcn --window_size 30 --epochs 30 --batch_size 128 --output_dir runs/tcn
```

## Reproducing the paper figures and metrics without retraining

```bash
python reproduce_results.py --run_dir runs/lstm
python reproduce_results.py --run_dir runs/tcn
```

This will regenerate:

- `learning_curve_reproduced.png`
- `test_scatter_reproduced.png`
- `reproduced_metrics.json`

## Modeling choices

### Preprocessing
- Computed train RUL as `max_cycle - cycle`
- Clipped RUL at 125
- Removed near-constant channels
- Standardized features using only the training split
- Split validation by engine unit rather than by windows
- Used a window size of 30 cycles

### LSTM baseline
- 2-layer LSTM
- Hidden size 64
- Dropout 0.2
- MLP readout from the last hidden state

### TCN variant
- 3 residual temporal convolution blocks
- Channel sizes `(64, 64, 64)`
- Kernel size 3
- Dilations `(1, 2, 4)`
- MLP readout from the last time step

## Notes

- The final run in this package was completed on CPU. The code also supports GPU automatically if CUDA is available.
- The `train.py` download logic was patched to search the extracted archive recursively, because the NASA zip layout can vary slightly across mirrors.

## AI use disclosure

AI tools were used to help reframe code structure for clarity (eg. adding user friendly comments), debug Colab download issues, and polish repository documentation. Experiment design choices, interpretation of the results, and final written analysis were completed by the author.
