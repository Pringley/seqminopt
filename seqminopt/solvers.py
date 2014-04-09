"""Engines for solving individual SVM problems."""

# TODO:
# - check if solution actually works
# - precompute dot products using O(size*size) space
# - replace dot with custom kernel function

import random

from .util import dot, scalar_multiply, vector_add

class SimpleSolver:
    """Solver based on Andrew Ng's Simplified SMO.

    Simplified SMO: http://cs229.stanford.edu/materials/smo.pdf

    :param points: array of training point vectors
    :param values: corresponding array of training values (scalars)
    :param tradeoff: tradeoff constant
    :param tol: (optional) numerical tolerance, default 1e-10
    :param max_iter: (optional) stop after this many iterations without change
    """

    def __init__(self, points, values, tradeoff, tol=1e-10, max_iter=None):

        self.size = len(points)
        if len(values) != self.size:
            raise TypeError("points and values should be equal length")

        self.points = points
        self.values = values
        self.tradeoff = tradeoff
        self.tol = tol if tol is not None else 1e-10
        self.max_iter = max_iter if max_iter is not None else self.size * self.size

        # Initialize variables to optimize
        self.alphas = [0. for _ in range(self.size)]
        self.offset = 0.

    def solve(self):
        """Solve the SVM problem.

        :return: a tuple of (weights, offset)
        """
        unchanged_iter = 0
        while unchanged_iter < self.max_iter:

            changes = 0

            # Choose ii by looking for KKT violations.
            for ii in range(self.size):
                if self.has_violation(ii):

                    # Choose corresponding jj randomly.
                    jj = ii
                    while jj == ii:
                        jj = random.randint(0, self.size - 1)

                    # Update alpha_ii, alpha_jj, and offset
                    if self.update(ii, jj):
                        changes += 1

            if changes:
                unchanged_iter = 0
            else:
                unchanged_iter += 1

        return self.weight_vector(), self.offset

    def weight_vector(self):
        """Compute the weight vector for the primal problem."""
        total = [0. for ii in range(len(self.points[0]))]
        for ii in range(self.size):
            total = vector_add(total,
                    scalar_multiply(self.points[ii],
                        self.alphas[ii] * self.values[ii]))
        return total

    def classify(self, point):
        """Return the current classification of a point."""
        return sum(self.alphas[ii] * self.values[ii]
                * dot(self.points[ii], point) + self.offset
                for ii in range(self.size))

    def error(self, index):
        """Return the current classification error of a point."""
        return self.classify(self.points[index]) - self.values[index]

    def has_violation(self, ii):
        """Check if the point at ii violates any KKT conditions."""
        value_ii = self.values[ii]
        alpha_ii = self.alphas[ii]
        error_ii = self.error(ii)

        viol1 = value_ii * error_ii < -self.tol and alpha_ii < self.tradeoff
        viol2 = value_ii * error_ii > self.tol and alpha_ii > 0
        return viol1 or viol2

    def update(self, ii, jj):
        """Update variables for the given pair."""

        # Convenience constants
        pt_ii = self.points[ii]
        pt_jj = self.points[jj]
        val_ii = self.values[ii]
        val_jj = self.values[jj]
        alpha_ii = self.alphas[ii]
        alpha_jj = self.alphas[jj]
        error_ii = self.error(ii)
        error_jj = self.error(jj)

        # Compute bounds for new alpha_jj such that 0 <= alpha_jj <= tradeoff
        if val_ii == val_jj:
            lower = max(0, alpha_ii + alpha_jj - self.tradeoff)
            upper = min(self.tradeoff, alpha_ii + alpha_jj)
        else:
            lower = max(0, alpha_jj - alpha_ii)
            upper = min(self.tradeoff, self.tradeoff + alpha_jj - alpha_ii)

        # If the bounds are equal, skip this update
        if lower == upper:
            return False

        # Compute new alpha_jj to optimize the objective
        eta = 2. * dot(pt_ii, pt_jj) - dot(pt_ii, pt_ii) - dot(pt_jj, pt_jj)
        if eta == 0:
            return False
        new_alpha_jj = alpha_jj - val_jj * (error_ii - error_jj) / eta

        # Clip new alpha_jj to fall within bounds
        new_alpha_jj = max(new_alpha_jj, lower)
        new_alpha_jj = min(new_alpha_jj, upper)

        # If alpha_jj has not changed (within tolerance), skip the rest
        if abs(alpha_jj - new_alpha_jj) < self.tol:
            return False

        new_alpha_ii = alpha_ii + val_ii * val_jj * (alpha_jj - new_alpha_jj)

        # Compute new offset to satisfy 0 < alpha < tradeoff
        offset1 = (self.offset - error_ii
                - val_ii * (new_alpha_ii - alpha_ii) * dot(pt_ii, pt_ii)
                - val_jj * (new_alpha_jj - alpha_jj) * dot(pt_ii, pt_jj))
        offset2 = (self.offset - error_jj
                - val_ii * (new_alpha_ii - alpha_ii) * dot(pt_ii, pt_jj)
                - val_jj * (new_alpha_jj - alpha_jj) * dot(pt_jj, pt_jj))
        if alpha_ii < self.tol or alpha_ii < self.tradeoff - self.tol:
            new_offset = offset1
        elif alpha_jj < self.tol or alpha_jj < self.tradeoff - self.tol:
            new_offset = offset2
        else:
            new_offset = (offset1 + offset2) / 2.

        # Commit updates
        self.alphas[ii] = new_alpha_ii
        self.alphas[jj] = new_alpha_jj
        self.offset = new_offset

        return True
