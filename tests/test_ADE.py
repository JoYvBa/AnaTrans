import sys

sys.path.insert(0, "/Users/jorrit/anatrans")

import numpy as np
import pytest

import anatrans.ADE_eq as ade
from tests.output_testing import ADE_results


class TestClassOneDim:
    """Testing class for the cxt_1D class."""

    def setup_method(self):
        self.x_min = 0
        self.x_max = 20
        self.x_nstep = 11
        self.t_min = 0
        self.t_max = 10
        self.t_nstep = 11

        self.x = np.linspace(self.x_min, self.x_max, self.x_nstep)
        self.t = np.linspace(self.t_min, self.t_max, self.t_nstep)
        self.adv = ade.Cxt1D(self.x, self.t)

    def test_value_type(self) -> None:
        """Test to make sure that wrong input data types are recognized."""
        v = "This should not work"
        with pytest.raises(TypeError, match=f"The data type of v must be 'float' or 'int', not {type(v)}"):
            self.adv.ade(v=v)

    def test_value_range(self) -> None:
        """Test to make sure that negative input data is not accepted."""
        v = -0.8
        with pytest.raises(ValueError, match=f"The value of v is {v}, but must be larger than or equal to 0"):
            self.adv.ade(v=v)

    def test_input_type(self) -> None:
        """Test to make sure that only numpy arrays are accepted as input."""
        self.x = 10
        with pytest.raises(TypeError, match=f"The data type of x and t must both be 'numpy.ndarray'.\
 Types were x : {type(self.x)}, t : {type(self.t)} instead."):
            ade.Cxt1D(self.x, self.t)

    def test_tile(self) -> None:
        """Test to make sure distance, time and concentration tiles are correctly generated."""
        x = np.tile(self.x, (len(self.t), 1))
        t = np.tile(self.t, (len(self.x), 1)).T
        cxt = np.zeros((len(self.t), len(self.x)))
        out = ade.Cxt1D(self.x, self.t)

        np.testing.assert_allclose(x, out.x)
        np.testing.assert_allclose(t, out.t)
        np.testing.assert_allclose(cxt, out.cxt)

    def test_ade_noadv(self) -> None:
        """Test to make sure variables are correctly made when calculating the ADE without advection."""
        self.adv.ade(advection=False)
        np.testing.assert_allclose(self.adv.adv, self.adv.x)

    def test_ade_advdispdec(self) -> None:
        """Test to make sure ADE is correctly calculated with advection, dispersion and decay."""
        self.adv.ade(advection=True, dispersion=True, decay=True)
        np.testing.assert_allclose(self.adv.cxt[-1, :], ADE_results["adv_disp_decay_pulse"])

    def test_ade_advdispdecret(self) -> None:
        """Test to make sure ADE is correctly calculated with advection, dispersion, decay and retardation."""
        self.adv.ade(advection=True, dispersion=True, decay=True, retardation=True)
        np.testing.assert_allclose(self.adv.cxt[-1, :], ADE_results["adv_disp_decay_retard_pulse"])
