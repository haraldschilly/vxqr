from __future__ import print_function, absolute_import
import unittest

import numpy as np
import logging
from vxqr1 import VXQR1
from vxqr1.vxqr1 import Result
from vxqr1.utils import create_logger


class TestVXQR1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.log = create_logger("testing")

    def test_problem_1(self):
        n = 100  # dimension
        p = 2  # norm
        e = 1  # exponent
        A = np.random.rand(n, n) - .5
        x0 = np.random.rand(n) - .5
        x = A.dot(x0)
        b = A.sum(axis=1)

        func = lambda _: np.linalg.norm(A.dot(_) - b, p) ** e

        starting_point = 2 * np.random.rand(n) - 1
        upper_bounds = 1e7 * np.ones(n)
        lower_bounds = -upper_bounds
        nf_max = 200 * n
        #f_target = 0.01 * func(x)
        nb_solves = 5

        vx1config = VXQR1.Config(
            iscale=np.max([np.linalg.norm(lower_bounds, np.inf),
                           np.linalg.norm(upper_bounds, np.inf)]),
            stop_nf_target=nf_max, # np.array([nf_max, 0]),
            stop_f_target=np.inf # np.array([np.inf, f_target])
        )

        vx1 = VXQR1(vx1config, log_level=logging.INFO)

        results = []
        for i in range(nb_solves):
            self.log.info("solve loop: i=%d" % i)
            results.append(vx1.solve(func,
                                     starting_point,
                                     lower_bounds=lower_bounds,
                                     upper_bounds=upper_bounds
                                     ))
        solution = sorted(results)[0]

        assert solution is not None
        assert isinstance(solution, Result)
        assert isinstance(solution.xbest, np.ndarray)
        print("solution: f=%s\n@x=%s" % (solution.fbest, solution.xbest))
