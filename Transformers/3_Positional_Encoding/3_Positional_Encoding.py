import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Create position indices
    # [:, np.newaxis] → reshapes it into a column vector
    positions = np.arange(seq_length)[:, np.newaxis]

    # Create dimension indices
    # [np.newaxis, :] → converts it into a row vector
    dims = np.arange(d_model)[np.newaxis, :]

    # Compute Frequency Term
    angle_rates = 1/np.power(10000, (2*(dims//2))/d_model)

    # Multiply Positions with frequency
    angle_rads = positions * angle_rates

    # Initialize output matrix
    pos_encoding = np.zeros((seq_length, d_model))

    # Fill even indices with sin
    pos_encoding[:, 0::2] = np.cos(angle_rads[:, 0::2])

    # Fill odd indices with cos
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Return result
    return pos_encoding

print(positional_encoding(4, 4))
