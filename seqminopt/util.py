"""Utility functions."""

def dot(vector1, vector2):
    """Dot product of two vectors."""
    if len(vector1) != len(vector2):
        raise TypeError("cannot take dot product of unequal-sized vectors")
    return sum(vector1[ii] * vector2[ii] for ii in range(len(vector1)))

def scalar_multiply(vector, scalar):
    """Multiply a vector by a scalar."""
    return [vector[ii] * scalar for ii in range(len(vector))]

def vector_add(vector1, vector2):
    """Elementwise vector addition."""
    if len(vector1) != len(vector2):
        raise TypeError("cannot add unequal-sized vectors")
    return [vector1[ii] + vector2[ii] for ii in range(len(vector1))]
