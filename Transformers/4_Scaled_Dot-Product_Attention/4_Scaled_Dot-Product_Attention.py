import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # compute dot product QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # scale by sqrt(d_k)
    d_k = Q.size(-1)
    scores = scores / math.sqrt(d_k)

    # apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # multiply by V to get final output
    output = torch.matmul(attention_weights, V)

    return output

'''
Each word asks:

"Which other words in the sentence are important for me?"

Q = what I am looking for
K = what each word offers
V = the information each word provides

Then attention combines them.
'''