"""Utility functions."""

def dot(vector1, vector2):
    """Dot product of two vectors."""
    if len(vector1) != len(vector2):
        raise TypeError("cannot take dot product of unequal-sized vectors")
    return sum(vector1[ii] * vector2[ii] for ii in range(len(vector1)))
