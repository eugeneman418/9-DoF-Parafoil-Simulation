from unittest import TestCase
import numpy as np
import SixDegreeModel


class Test(TestCase):
    def test_t_ib_func(self):
        euler_angle = np.array([1, 2, 3])
        T_IB = SixDegreeModel.T_IB_func(euler_angle)

        self.assertGreaterEqual(1e-5, np.sum(T_IB.T - np.linalg.inv(T_IB)))

