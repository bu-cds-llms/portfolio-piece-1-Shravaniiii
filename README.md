# From Words to Attention: Building a Text Representation Pipeline from Scratch

> **DS587 — Shravani Maskar | Portfolio Piece**

---

## What This Project Is

This project builds the core machinery behind modern language models  **from raw text to self-attention**  entirely from scratch in PyTorch, trained on real IMDb movie reviews.

The same mathematical operations that power BERT and GPT are implemented here step by step: tokenization, word embeddings, scaled dot-product attention, and multi-head attention. Rather than treating these as black boxes, every component is written from first principles with detailed explanations of *why* each design choice was made.

---

## The Pipeline

```
Raw Text  →  Tokenizer  →  Word Embeddings  →  Q/K/V Projections  →  Self-Attention  →  Output
```

| Part | What It Does |
|---|---|
| **1. Tokenization** | Splits text into words, builds a ~15,000-word vocabulary from IMDb reviews, maps words to integer indices. Handles unknown words with `<UNK>` and variable-length sequences with `<PAD>`. |
| **2. Word Embeddings** | Trains a PyTorch classifier on 1,000 IMDb reviews to learn 16-dimensional word vectors. Embeddings are a byproduct of the classification task — the model learns to represent words in ways useful for predicting sentiment. |
| **3. Scaled Dot-Product Attention** | Implements `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V` from scratch. Every step — dot products, scaling, softmax, weighted sum — is written explicitly with no external attention libraries. |
| **4. Full Pipeline** | Connects embeddings to attention via learned linear projections W_Q, W_K, W_V. Produces attention heatmaps and runs experiments on how d_k affects attention sharpness. |
| **5. Multi-Head Attention** | Runs 4 parallel attention heads simultaneously, each with independent projections. Compares head diversity and visualizes positive vs. negative sentences across all heads. |
| **6. Critical Analysis** | Interprets what the model actually learned, why attention patterns look the way they do, and what the gap is between this implementation and production systems like BERT. |

---

## Key Findings

**Embeddings learned a clean sentiment axis.**
PCA of the trained word vectors shows positive words (*great*, *brilliant*, *wonderful*) clustering to the left and negative words (*terrible*, *awful*, *worst*) to the right along PC1, with near-zero variance on PC2. The model converged on a single linear dimension for sentiment — a direct consequence of the binary training objective. When the task only asks "positive or negative?", the optimizer finds exactly one useful direction and ignores the other 15 dimensions.

**Attention patterns are near-uniform with random projections — and that's expected.**
The Q/K/V projection matrices are randomly initialized and never trained, so attention weights are close to uniform across all words. This is the correct baseline, not a failure. Meaningful patterns (like BERT's coreference resolution) emerge only after millions of gradient updates on tasks that *require* understanding word relationships.

**Higher d_k → sharper attention distributions.**
Across d_k values of 2, 4, 8, and 32, larger key/query dimensions produce measurably lower-entropy attention. This validates the intuition behind the √d_k scaling factor — without it, high-dimensional dot products grow too large and collapse the softmax into near-zero gradients.

**Multi-head heads diverge even from random initialization.**
Four heads on the same sentence produce visibly different attention patterns. This explains why multi-head attention consistently outperforms single-head — parallel heads explore different subspaces of the embedding simultaneously rather than committing to one projection.

---

## How to Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd portfolio-piece

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook notebooks/main_analysis.ipynb
```

Run all cells from top to bottom. The IMDb dataset (~80MB) downloads automatically on first run — internet connection required. All figures save to `outputs/` automatically.

> **Important**: Always run cells in order from the top. Each cell depends on variables defined in previous cells.

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.6.0
datasets>=2.0.0
jupyter>=1.0.0
```

---

## Repository Structure

```
portfolio-piece/
├── README.md
├── requirements.txt
└── notebooks/
    └── main_analysis.ipynb
```

---

## What Would Make This Better

This implementation is intentionally minimal to make the mechanics legible. The gap between this and production systems is not the architecture — the math is identical — it's the training:

- **Pre-trained embeddings** (GloVe-100, fastText-300) would produce richer geometric structure than 1,000 reviews can
- **Trained Q/K/V projections** via end-to-end training would produce interpretable attention patterns instead of near-uniform ones
- **Positional encoding** would let the model distinguish word order (currently order-blind)
- **Residual connections** (`x = x + Attention(x)`) would make deeper stacking stable
- **More training data** — BERT used 3 billion words; we used ~100,000

---

## References

- Vaswani et al. (2017) — *Attention Is All You Need*
- Devlin et al. (2018) — *BERT: Pre-training of Deep Bidirectional Transformers*
- Kaplan et al. (2020) — *Scaling Laws for Neural Language Models*
