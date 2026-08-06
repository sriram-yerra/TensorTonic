import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_length, d_model), dtype=np.float64)

    # Positions: [[0], [1], [2], ...]
    position = np.arange(seq_length).reshape(seq_length, 1)

    # Denominator terms for each pair of dimensions
    div_term = np.power(10000, np.arange(0, d_model, 2) / d_model)

    # Angles for all positions and even dimensions
    angles = position / div_term

    # Even dimensions: sin
    pe[:, 0::2] = np.sin(angles)

    # Odd dimensions: cos
    pe[:, 1::2] = np.cos(angles)

    return pe