# Privacy-Preserving GNN Inference with Homomorphic Encryption

Encrypted inference on Graph Neural Network embeddings using **Paillier** and **CKKS** homomorphic encryption. The project demonstrates that a GNN classifier can make predictions on **encrypted feature vectors** — the server never sees the plaintext input — while matching the accuracy of ordinary plaintext inference, and quantifies the runtime cost of doing so.

Two tasks are covered:

- **Cora** — node classification (7-class, multiclass) on the Planetoid citation graph.
- **MUTAG** — graph classification (binary) on the TUDataset molecular benchmark.

---

## The core idea

Homomorphic Encryption (HE) lets you compute on ciphertexts: `Dec(Enc(a) ⊕ Enc(b)) = a + b`. But HE only supports a limited algebra efficiently (additions and scalar multiplications), so a full non-linear GNN cannot be run under encryption directly. This project uses a **split (hybrid) architecture** that keeps the heavy non-linear work in plaintext and reserves encryption for the one linear step where privacy matters:

```
┌─────────────────────── plaintext (training / feature extraction) ──────────────────────┐
│  Graph  ──►  GNN encoder  ──►  node/graph embeddings  ──►  Logistic Regression (W, b)   │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                                                     │  extract W, b
                                                                     ▼
┌──────────────────────────── ENCRYPTED inference (the private step) ─────────────────────┐
│   Enc(x)  ──►   logits_c = Σ_j Enc(x_j)·W_cj  +  b_c   (homomorphic)  ──►  Dec  ──► softmax │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

Because the classifier head is **linear**, the logit computation `W·x + b` is exactly an HE-friendly sequence of scalar-multiply-and-add operations. The non-linear `softmax` is applied **after decryption**, in plaintext, so it costs nothing cryptographically. Accuracy is therefore identical to the plaintext baseline (up to CKKS approximation noise); the only price is latency.

---

## Technical details (the important part)

### 1. GNN encoders (plaintext feature extractors)

Both notebooks implement several interchangeable encoders so their representations can be compared under the same HE pipeline:

| Encoder | Key mechanics |
|---------|---------------|
| **GIN** | `GINConv` with 2–3 MLP layers; MUTAG version concatenates `global_mean_pool + global_max_pool` for graph-level readout |
| **GCN** | 3-layer `GCNConv`, `BatchNorm1d`/`LayerNorm`, **GELU**, residual connections, Xavier init, gradient clipping |
| **GAT** | Multi-head `GATConv` (`heads=4–8`), ELU, attention dropout; graph-level version pools mean+max+sum |
| **GraphSAGE** | 3× `SAGEConv` + batchnorm; **mean+max+sum** pooling on MUTAG; followed by **PCA** on Cora to shrink the HE input |
| **Hybrid GAT+GIN++** | Dual parallel branches (GAT ‖ GIN) with **gated fusion** (`g·h₂ + (1−g)·h₁`, `g=σ(θ)` a learnable scalar), residual projections, `LayerNorm`, **DropEdge**, and label smoothing |

Training uses AdamW, `CosineAnnealingLR` / `CosineAnnealingWarmRestarts` / `ReduceLROnPlateau`, early stopping on validation loss, and `StratifiedKFold` CV with Gaussian-noise feature augmentation.

### 2. Paillier inference — exact, additively homomorphic

Implemented with the [`phe`](https://github.com/data61/python-paillier) library. Paillier is **partially homomorphic**: it supports ciphertext+ciphertext addition and ciphertext×plaintext multiplication — exactly what a linear layer needs.

- **Fixed-point quantization:** floats are mapped to integers with an integer `scale` (e.g. 1000–5000). Inputs and weights are each scaled by `scale`, so their product carries `scale²`; the bias is therefore pre-scaled by `scale²` and the decrypted integer is divided back by `scale²`:
  ```python
  W_int = round(W * scale)
  b_int = round(b * scale**2)        # bias lives in the scale² domain
  acc   = Enc(b_int)
  for enc_feat, w in zip(enc_x, W_int):
      acc += enc_feat * w            # ciphertext × plaintext int
  score = Dec(acc) / scale**2        # decode back to float
  ```
- **Key size:** `PAILLIER_KEY` = 1024 bits for demo speed, 2048 for stronger security.
- **Properties:** results are **exact** (integer arithmetic, no approximation), but it is the **slowest** path — one ciphertext is created per feature per sample, and multiclass needs one encrypted dot product **per class** (One-vs-Rest).

### 3. CKKS inference — approximate, SIMD-batched (leveled FHE)

Implemented with [`TenSEAL`](https://github.com/OpenMined/TenSEAL) (Microsoft SEAL backend). CKKS operates natively on **vectors of reals** and supports packed (SIMD) operations, with these context parameters exposed as knobs:

- `poly_modulus_degree` = **4096** (cheaper) or **8192** (more precision / larger multiplicative budget)
- `coeff_mod_bit_sizes` = `[40,20,40]` or `[60,40,40,60]` — the modulus chain that sets the noise budget
- `global_scale` = `2**20` or `2**40` — the fixed-point precision of the encoded reals
- `generate_galois_keys()` — enables the rotations needed for packed dot-products

Two execution strategies are benchmarked:
- **CKKS single** — encrypt each sample as one `ckks_vector`, then `enc_x.dot(enc_w)` per class.
- **CKKS batched** — pack a batch so one SIMD ciphertext carries many values, amortizing per-op overhead. This is the **fastest** HE variant here.

CKKS is **approximate** (small numerical noise), which is harmless because the argmax/softmax is robust to it.

### 4. Multiclass under HE (One-vs-Rest)

Cora is 7-class. Since HE handles only the linear map, multiclass logits are produced by running the encrypted dot product **once per class row** of `W` (shape `[n_classes, n_features]`) and assembling the logit vector before a plaintext softmax. MUTAG is binary, so a single positive-class weight vector suffices.

### 5. PCA as an HE cost-reducer

The dominant HE cost scales with the feature dimension (Paillier encrypts per-feature; CKKS slot usage grows with length). On Cora, embeddings are passed through **PCA** (e.g. 256 → 64 dims) *before* encryption, cutting the number of homomorphic operations dramatically with negligible accuracy loss.

### 6. Evaluation

Every fold compares **four inference paths** — Plaintext, Paillier, CKKS-single, CKKS-batched — on identical weights, reporting accuracy, macro precision/recall/F1, ROC-AUC (OvR), log loss, and per-scheme wall-clock runtime, aggregated across folds as `mean ± std`.

---

## Requirements

```bash
pip install torch torch_geometric scikit-learn numpy pandas tqdm
pip install phe tenseal        # Paillier + CKKS
```

- `phe` — Paillier (partially homomorphic encryption)
- `tenseal` — CKKS (leveled fully homomorphic encryption, SEAL backend)
- `torch_geometric` — GNN layers and the Planetoid / TUDataset loaders

## Usage

```bash
jupyter notebook "GNN_Homomorphic_Encryption_on_Cora_Dataset.ipynb"     # node classification
jupyter notebook "GNN_Homomorphic_Encryption_on_MUTAG_Dataset.ipynb"    # graph classification
```

Both were written for Google Colab and download their datasets automatically. Each notebook is organized as independent cells, one per GNN encoder, and prints a plaintext-vs-encrypted comparison plus timing. Tune the `USER KNOBS` block at the top of each cell (key size, CKKS poly degree, scale, PCA dim, folds, epochs) to trade security/precision against runtime.

> ⚠️ **Research/demo code.** 1024-bit Paillier keys and small CKKS parameters are chosen for speed, not production security. Choose parameters per a real threat model before any real-world use.

## Project Structure

```
homomorphic-gnn-inference/
├── notebooks/
│   ├── GNN_Homomorphic_Encryption_on_Cora_Dataset.ipynb
│   └── GNN_Homomorphic_Encryption_on_MUTAG_Dataset.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

## How the two notebooks differ

| | Cora | MUTAG |
|---|------|-------|
| Task | Node classification | Graph classification |
| Classes | 7 (multiclass, OvR under HE) | 2 (binary) |
| Loader | `Planetoid` | `TUDataset` |
| Readout | per-node embeddings | global mean+max+sum pooling |
| HE input reduction | PCA before encryption | direct (small graphs) |

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Built with PyTorch Geometric, [python-paillier (`phe`)](https://github.com/data61/python-paillier), and [TenSEAL](https://github.com/OpenMined/TenSEAL). Datasets: [Planetoid/Cora](https://github.com/kimiyoung/planetoid) and [TUDataset/MUTAG](https://chrsmrrs.github.io/datasets/).
