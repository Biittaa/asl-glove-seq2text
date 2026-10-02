# asl-glove-seq2text

Sensor-to-text Transformer for sign-language recognition and translation.

The model is an encoder-decoder Transformer with two outputs:

- **Encoder + CTC head**: predicts the gesture-order gloss sequence (`label_no_order`).
- **Decoder + cross-entropy head**: predicts the spoken-order sentence (`label_order`) autoregressively.

Total loss: `L = lambda_ctc * L_CTC + lambda_ce * L_CE`.

Optional modules (all switchable from the config): conv subsampling, two separate hand branches with cross-attention fusion, and a graph neural network (GNN) over the sensor nodes.

---

## Repository structure

```
configs/config.yaml        # all paths and hyperparameters
scripts/build_splits.py    # builds train/validation/test manifests
train.py                   # training + final evaluation + predictions dump
infer.py                   # inference on a single recording CSV
src/                       # dataset, model, losses, metrics, vocab, utils
notebooks/colab_train.ipynb
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset layout

`data.dataset_root` in the config must point to the folder that directly contains:

```
data/            # sequence CSV files
manifests/       # all.jsonl (+ generated split folders)
metadata.json
```

---

## Step 1: Build the data splits

Every split type is written to its **own subfolder** (`manifests/<mode>/`), so building one split never overwrites another.

```bash
python scripts/build_splits.py --config configs/config.yaml --split <mode>
```

| Mode | What it tests |
|------|---------------|
| `subject` | Subject-disjoint split taken from the `split` field in `all.jsonl` (subject1 -> train, subject2 -> validation, subject3 -> test). Generalization to an **unseen signer**. |
| `random` | Random split over all pooled recordings, stratified by `sentence_id`. Generalization to a **new recording of a known sentence**. |
| `compositional` | A set of sentences (default 15%) is held out entirely: ALL recordings of those sentences go to test and none appear in train. Tests generalization to **sentences never seen in training**, built from words that do appear in other training sentences. Sentences are only held out if no word/gloss would be missing from the training vocabulary. |
| `compositional_s1` | Same as `compositional`, but uses only the recordings of `subject1`. |
| `paper` | Paper-style user-independent split: train on all other subjects, validation/test come from one held-out subject (`data.split.paper.held_out_subject`). |


## Step 2: Train

```bash
python train.py --config configs/config.yaml --split_mode random --epochs 100 --batch_size 16
```

| Argument | Description |
|----------|-------------|
| `--config` | Path to the YAML config (default: `configs/config.yaml`). |
| `--dataset_root` | Overrides `data.dataset_root` (useful on Colab / Drive). |
| `--split_mode` | Overrides `data.split.mode` (`subject`, `random`, `compositional`, `compositional_s1`, `paper`). |
| `--epochs` | Overrides `train.epochs`. |
| `--batch_size` | Overrides `train.batch_size`. |
| `--resume` | Resume from `last.pt` in the checkpoint folder. |

Everything produced by a run is stored under a folder named after the split mode, so runs with different splits do not collide:

```
artifacts/vocab/<mode>/             # gloss and word vocabularies (built from train only)
artifacts/checkpoints/<mode>/       # best.pt, last.pt
artifacts/predictions/<mode>/predictions.csv
<dataset_root>/artifacts/<mode>/feature_stats.json
```

After training, the best checkpoint (by `train.best_metric`, default `val_wer`) is loaded and evaluated on the test set, and predictions for train/validation/test are written to `predictions.csv`.

---

## Important config options (`configs/config.yaml`)

### Data

| Key | Meaning |
|-----|---------|
| `data.dataset_root` | Dataset folder (see layout above). |
| `data.split.mode` | Default split mode. |
| `data.split.compositional.*` | `unseen_sentence_fraction`, `train_recording_ratio`, `seed`. |
| `data.split.compositional_s1.subjects` | Which subjects are used for `compositional_s1`. |
| `data.split.paper.held_out_subject` | Subject used for validation/test in the `paper` split. |
| `data.norm_mode` | Feature normalization: `recording` (z-score per recording), `recording_center` (per-recording mean removal, global std), or any other value for global train mean/std. |

### Model front-end

| Key | Meaning |
|-----|---------|
| `model.use_conv_subsampling` | Stride-2 conv stack that shortens the frame axis before attention. Lowers attention cost roughly 16x with the defaults. `false` = plain linear projection. |
| `model.conv_kernel_size / conv_stride / conv_num_layers` | Conv subsampling settings (T shrinks by `stride ** num_layers`). |
| `model.two_hand` | `true` = right hand and left hand are processed in two separate branches and then fused. `false` = single branch over all features. |
| `model.hand_feature_dim` | Number of features per hand (19). The first 19 columns are the right hand, the rest the left hand. |
| `model.hand_d_model` | Hidden size of each hand branch. |
| `model.fusion` | How the two hands are fused: `concat` or `cross_attention`. |
| `model.fusion_nhead` | Number of heads for the cross-attention fusion. |
| `model.hand_encoder_layers` | Optional Transformer layers inside each hand branch (0 = none). |

### Graph (GNN) option

Each sensor is a node (nodes are derived from the column names) and edges follow the skeleton (arm - forearm - hand - fingers). The GNN runs **before** the Transformer encoder (spatial first, then temporal).

| Key | Meaning |
|-----|---------|
| `model.use_gnn` | Turns the graph front-end on/off. |
| `model.gnn_hidden` | Hidden size per node. |
| `model.gnn_layers` | Number of graph layers. |
| `model.gnn_adaptive` | Adds a learnable adjacency residual on top of the skeleton adjacency. |
| `model.gnn_cross_hand` | Connects the two hands with an edge. Only used when `two_hand: false`. |

How the options combine:

- `two_hand: true`, `use_gnn: false` -> two conv branches + fusion (baseline two-hand model).
- `two_hand: true`, `use_gnn: true` -> one GNN per hand, then the fusion (`fusion`).
- `two_hand: false`, `use_gnn: true` -> one joint graph over both hands.
- `two_hand: false`, `use_gnn: false` -> single conv branch (baseline).

### Transformer and loss

| Key | Meaning |
|-----|---------|
| `model.d_model`, `nhead`, `dim_feedforward`, `dropout` | Transformer size. |
| `model.encoder_layers`, `decoder_layers` | Depth of the encoder / decoder. |
| `loss.lambda_ctc`, `loss.lambda_ce` | Weights of the CTC and cross-entropy losses. |
| `loss.label_smoothing` | Label smoothing for the decoder loss only. |

### Training

| Key | Meaning |
|-----|---------|
| `train.epochs`, `batch_size`, `lr`, `weight_decay` | Basic optimization settings. |
| `train.warmup_steps` | Warmup of the inverse-sqrt LR schedule. Reduce it for small datasets. |
| `train.amp` | Mixed precision (CUDA only). |
| `train.best_metric` | Metric used to select the best checkpoint (lower is better). |
| `train.early_stop_patience` | Epochs without improvement before stopping (0 = disabled). |
| `runtime.device` | `cuda` (falls back to CPU if unavailable). |
