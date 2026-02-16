# Word Embeddings

Word Embeddings transform discrete token IDs into continuous vector representations, enabling neural networks to learn semantic relationships between words.

## Mathematical Foundation

The embedding lookup is a simple matrix multiplication with a one-hot vector:

```markdown
E(x) = W[x] ⋅sqrt(d_model)
```

Where `W ∈ R^{V × d}` is the embedding matrix, `V` is vocabulary size, and `d` is embedding dimension.

## Scaling Factor

The `sqrt(d_model)` scaling serves a critical purpose:

- **Variance Control**: Embeddings are typically initialized with small values (variance ~1/d). Scaling restores appropriate magnitude.
- **Balance with Positional Encoding**: Positional encodings have fixed magnitude. Scaling ensures embeddings aren't dominated by positions.

## Weight Tying

The Transformer paper shares weights between:

1. Input embedding matrix
2. Output embedding matrix
3. Pre-softmax linear transformation

This reduces parameters by ~30% and improves generalization by forcing consistent token representations.

## Learned vs. Fixed

- **Learned Embeddings**: Randomly initialized, trained end-to-end with the model.
- **Pre-trained (Word2Vec, GloVe)**: Transfer learning from large corpora, can be frozen or fine-tuned.

Modern transformers almost always use learned embeddings trained jointly with the model architecture.