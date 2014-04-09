import unittest

import seqminopt
from seqminopt.solvers import SimpleSolver
from seqminopt.util import dot

PARAMS = {
    'points': [ [3., 2.], [2., 1.], [1., 3.], [5., 7.], [5., 9.], [4., 1.] ],
    'values': [1., 1., -1., -1., -1., 1.],
    'tradeoff': 1.
}

class SimpleSolverTestCase(unittest.TestCase):

    def test_solve(self):
        solver = SimpleSolver(**PARAMS)
        (weights, offset) = solver.solve()

        # test values obtained with scikit-learn's LinearSVC
        self.assertAlmostEqual(weights[0], 0.66, places=1)
        self.assertAlmostEqual(weights[1], -0.66, places=1)
        self.assertAlmostEqual(offset, 0.33, places=1)

class UtilTestCase(unittest.TestCase):

    def test_dot(self):
        vector1 = [1, 2, 3]
        vector2 = [2, 2, 7]
        product = dot(vector1, vector2)
        self.assertEqual(product, 27)

if __name__ == '__main__':
    unittest.main()
