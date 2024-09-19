"""Author: Jorrit Bakker, Alraune Zech.

Script with analytical solutions for various boundary conditions and initial conditions of the ADE.
"""

import copy

import numpy as np

DEF_SET = {
    "v": 1.,             # flow velocity
    "c0": 1.,            # initial concentration
    "por": 1.,           # porosity
    "m0": 1,            # initial mass
    "deff": 1e-9,        # effective diffusion
    "alphaL": 0.1,       # dispersivity
    "alphaT": 0.01,      # dispersivity
    "alphaV": 0.001,     # dispersivity
    "R": 1.,             # retardation rate
    "mu": 0,             # decay rate
    "t0": 0,             # initial time
    "t1": 1,             # first time sill (e.g. end of block injection)
    "t2": 2,             # first time sill
    "sigma2": 1,         # broadness of gaussian input for advection
    "h_y": 1.,           # thickness of source strip in y-direction
    "h_z": 1.,           # thickness of source strip in z-direction
    }


class cxt_1D():
    """Calculate concentration profile over length and time in 1D using the advection-disperison equation."""

    def __init__(
        self,
        x: np.ndarray,
        t: np.ndarray,
        **settings: float,
        ) -> None:

        allowed_type = [np.ndarray, float, int]
        if ((type(x) or type(t)) not in allowed_type):
            msg = f"The data type of x and t must both be 'numpy.ndarray', 'float' or 'int'. Types were x : {type(x)},\
 t : {type(t)} instead."
            raise Exception(msg)

        self.settings=copy.copy(DEF_SET)
        self.settings.update(settings)

        self.check_settings()

        self.x = np.tile(x, (len(t), 1))
        self.t = np.tile(t, (len(x), 1)).T

        self.cxt = np.zeros((len(t), len(x)), dtype=float)

    def ade(
        self,
        *,
        advection: bool = True,
        dispersion: bool = False,
        decay: bool = False,
        retardation: bool = False,
        **settings: float,
        ) -> np.ndarray:
        """Calculate advection dispersion equation based on set parameters.

        Returns :
            Two-dimensional array with concentration values over distance (innner dimension) and time (outer dimension).
        """
        self.settings.update(settings)

        if retardation:
            t = self.t * self.settings["R"]
            m0 = self.settings["m0"] / self.settings["R"]
        else:
            t = self.t
            m0 = self.settings["m0"]

        if advection and dispersion:
            d = self.settings["deff"] + self.settings["alphaL"] * self.settings["v"]
        else:
            d = self.settings["deff"]

        adv = self.x - self.settings["v"] * t if advection else self.x

        spread = d * t

        print(np.exp(-((adv)**2 / (4 * spread))))
        self.cxt = m0 / (self.settings["por"] * np.sqrt(4 * np.pi * spread)) * np.exp(-((adv)**2 / (4 * spread)))

        if decay:
            self.cxt *= np.exp(-self.settings["mu"] * t)

        return self.cxt

    def check_settings(self) -> None:
        for key, value in self.settings.items():
            if type(value) not in {float, int}:
                msg = f"The data type of {key} must be 'float' or 'int', not {type(value)}"
                raise Exception(msg)

            if value < 0.:
                msg = f"The value of {key} is {value}, but must be larger than or equal to 0"
                raise Exception(msg)