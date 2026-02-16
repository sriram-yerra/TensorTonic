import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    embedding = nn.Embedding(vocab_size, d_model)

    nn.init.normal_(embedding.weight, mean=0, std=(1/math.sqrt(d_model)))

    return embedding

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    emb = embedding(tokens)
    print(emb)

    emb = emb * math.sqrt(d_model)
    print(emb)

    return emb
    
embed = create_embedding_layer(10, 10)
tokens = torch.tensor([2,3,5])
print(embed_tokens(embed, tokens, 4))