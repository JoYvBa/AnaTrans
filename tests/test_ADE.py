import sys

sys.path.insert(0, "/Users/jorrit/anatrans/anatrans")

import ADE_eq as ade
import numpy as np
import pytest


class TestClassOneDim:
    """Testing class for the cxt_1D class."""

    def setup_method(self):
        self.x_min = 0
        self.x_max = 200
        self.x_nstep = 51
        self.t_min = 0
        self.t_max = 1500
        self.t_nstep = 51

        self.x = np.linspace(self.x_min, self.x_max, self.x_nstep)
        self.t = np.linspace(self.t_min, self.t_max, self.t_nstep)

    # def test_inf_flow_pulse(self) -> None:

    #     # results = D1.transport(method = method, x = self.x, t = self.t, D_eff = self.D_eff, c0 = self.c0)
    #     # assert all(results[1:,0] == self.c0)
    #     # print(results[50,25])
    #     # assert results[50,25] == pytest.approx(5.90636)

    def test_value_type(self) -> None:
        """Test to make sure that wrong input data types are recognized."""
        adv = ade.cxt_1D(self.x, self.t)
        v = "This should not work"
        with pytest.raises(ValueError, match=f"The data type of v must be 'float' or 'int', not {type(v)}"):
            adv.ade(v=v)

    def test_value_range(self) -> None:
        """Test to make sure that negative input data is not accepted."""
        adv = ade.cxt1D(self.x, self.t)
        v = -0.8
        with pytest.raises(ValueError, match=f"The value of v is {v}, but must be larger than or equal to 0"):
            adv.ade(v=v)

