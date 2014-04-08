import unittest
import seqminopt

from seqminopt.solvers import SimpleSolver
from seqminopt.util import dot

PARAMS = {
    'points': [ [3, 2], [2, 1], [1, 3], [5, 7], [5, 9], [4, 1] ],
    'values': [1, 1, -1, -1, -1, 1],
    'tradeoff': 10.0
}

class SimpleSolverTestCase(unittest.TestCase):

    def test_solve(self):
        solver = SimpleSolver(**PARAMS)
        (alphas, offset) = solver.solve()
        print(alphas, offset)

class UtilTestCase(unittest.TestCase):

    def test_dot(self):
        vector1 = [1, 2, 3]
        vector2 = [2, 2, 7]
        product = dot(vector1, vector2)
        self.assertEqual(product, 27)

if __name__ == '__main__':
    unittest.main()
