## **What is Positional Encoding?**
**Positional Encoding** injects sequence order information into the Transformer, which otherwise has no inherent notion of position due to its permutation-invariant attention mechanism.

Positional Encoding is a method used in Transformers to add word order (position information) into the input embeddings.

## The Position Problem:

Unlike RNNs that process tokens sequentially, Transformers process all positions in parallel. Without positional information, "the cat sat on the mat" and "mat the on sat cat the" would be indistinguishable.

## Sinusoidal Encoding:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each dimension uses a different frequency, creating a unique "fingerprint" for each position.

## Why Sinusoids?

- **Bounded Values:** Always between -1 and 1, stable for any sequence length
- **Relative Position:** `PE(pos+k)` can be expressed as a linear function of `PE(pos)`, enabling the model to learn relative distances
- **Extrapolation:** Works for sequences longer than seen during training

## Frequency Intuition:

- Low dimensions (small `i`): High frequency, rapidly oscillating, encode fine-grained position differences
- High dimensions (large `i`): Low frequency, slowly varying, encode coarse position information

## Learned vs. Fixed:

The original paper found learned positional embeddings perform similarly to sinusoidal. However:

- **Sinusoidal:** No additional parameters, extrapolates to longer sequences
- **Learned:** More flexible, but limited to training sequence length

BERT and GPT use learned positional embeddings, while the original Transformer uses sinusoidal.

## Why do we need it

Transformers process all tokens in parallel using self-attention.

So without positional encoding:

- The model only sees words, not their order

The sentences:

> "the cat sat on the mat"  
> and  
> "mat the on sat cat the"  

would look identical to the model.

That is a big problem because word order carries meaning.

## What positional encoding does

It adds a position vector to each token embedding.

So final input becomes:

Final Input = Token Embedding + Positional Encoding

This tells the model:

- what the word is
- where the word is in the sentence

## How it is represented

For each position `pos`, we create a vector of size `d_model`. Each dimension uses sin and cos functions with different frequencies, so each position gets a unique pattern.

### Why sin and cos

The sinusoidal design gives important properties:

1. Unique encoding
   - Each position has a distinct vector
2. Relative position understanding
   - The model can learn relationships like:
     - next word
     - distance between words
3. Generalization
   - It works for longer sequences than seen during training

## Intuition

Think of positional encoding like:

- giving every word a GPS coordinate

so the model knows where it is located in the sentence.

## Simple Example

Sentence:

"I love AI"

Tokens:

- I → embedding
- love → embedding
- AI → embedding

Positions:

0, 1, 2

We generate positional vectors:

PE(0), PE(1), PE(2)

Final inputs:

- I + PE(0)
- love + PE(1)
- AI + PE(2)

Now the model knows both word meaning + position.

## Final One-Line Answer

Positional encoding is a technique used in Transformers to inject word order information into token embeddings so the model can understand sequence structure.