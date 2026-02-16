# Scaled Dot-Product Attention

Scaled Dot-Product Attention is the core mechanism that allows Transformers to dynamically focus on relevant parts of the input sequence.

## The Attention Formula:

Attention(Q, K, V) = softmax(Q * (K^T) / √d_k)V

This computes a weighted sum of Values, where weights are determined by Query-Key compatibility.

### **Core Formula**

The attention computation is:

Attention(Q, K, V) = **softmax( (Q · Kᵀ) / √d_k ) · V**

**Where:**

Q = queries → shape (batch, seq_len_q, d_k)

K = keys → shape (batch, seq_len_k, d_k)

V = values → shape (batch, seq_len_k, d_v)

## Query, Key, Value Intuition:

- **Query (Q)**: "What am I looking for?" - The current position's information needs  
- **Key (K)**: "What do I contain?" - Each position's retrievable identifier  
- **Value (V)**: "What information do I provide?" - The actual content to aggregate  

## Why Scale by √d_k?

For large √d_k, the dot product QK^T grows in magnitude. If Q and K have components with mean 0 and variance 1:

Var(Q ⋅ K) = d_k

Large values push softmax into regions with extremely small gradients (saturation). Dividing by √d_k stabilizes the variance to 1.

## Attention as Soft Dictionary Lookup:

Think of attention as a differentiable hash table:

- Keys are addresses  
- Query is the lookup key  
- Instead of exact match, we get soft similarity-weighted retrieval  

## Masking:

For autoregressive (decoder) models, we mask future positions by setting their scores to −∞ before softmax, ensuring the model can only attend to past tokens.