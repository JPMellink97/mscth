import numpy as np
import pysindy as ps
from pysindy import SINDy


class SINDy_model(SINDy):
    def __init__(
        self,
        x_data = None,
        x_test = None,
        u_data = None,
        u_test = None,
        data_split = False, # percentage of samples for test set
        optimizer=None,
        feature_library=None,
        differentiation_method=None,
        feature_names=None,
        t_default=1,
        discrete_time=False,
    ):
        if optimizer is None:
            optimizer = ps.STLSQ()
        self.optimizer = optimizer
        if feature_library is None:
            feature_library = ps.PolynomialLibrary()
        self.feature_library = feature_library
        if differentiation_method is None:
            differentiation_method = ps.FiniteDifference(axis=-2)
        self.differentiation_method = differentiation_method
        if not isinstance(t_default, float) and not isinstance(t_default, int):
            raise ValueError("t_default must be a positive number")
        elif t_default <= 0:
            raise ValueError("t_default must be a positive number")
        else:
            self.t_default = t_default
        self.feature_names = feature_names
        self.discrete_time = discrete_time

        if isinstance(data_split, float) and (x_test is None):
            # SPLIT is specified
            Nx, Nu = x_data.shape[0], u_data.shape[0]
            if Nx != Nu:
                raise ValueError("x_data has more or less samples than u_data")
            N_test = round(Nx*data_split)

            self.x_data = x_data[:-N_test]
            self.u_data = u_data[:-N_test]

            self.x_test = x_data[-N_test:]
            self.u_test = u_data[-N_test:]
        else:
            # no split 
            self.x_data = x_data
            self.u_data = u_data

            self.x_test = x_test
            self.u_test = u_test

        self.data_split = data_split
        

    def predict_s(self, train=False):
        if train:
            return self.predict(self.x_data, u=self.u_data)
        else:
            return self.predict(self.x_test, u=self.u_test)
        

    def simulate_s(
        self,
        train=False,
        integrator="solve_ivp",
        stop_condition=None,
        interpolator=None,
        integrator_kws={"method": "LSODA", "rtol": 1e-12, "atol": 1e-12},
        interpolator_kws={},
    ):
        # simulate full test sequence
        if train:
            x = self.x_data
            u = self.u_data
        else:
            x = self.x_test
            u = self.u_test

        if self.discrete_time:
            t_sim = u.shape[0]
        else:
            t_sim = np.linspace(0, self.t_default*x.shape[0], x.shape[0])
        
        return self.simulate(np.atleast_1d(x[0]), 
                             u=u, 
                             t=t_sim,
                             integrator="solve_ivp",
                             stop_condition=None,
                             interpolator=None,
                             integrator_kws={"method": "LSODA", "rtol": 1e-12, "atol": 1e-12},
                             interpolator_kws={})
    

    def fit_s(self):
        return self.fit(self.x_data, u=self.u_data)


def NRMS(y_pred, y_true):
    RMS = np.sqrt(np.mean((y_pred-y_true)**2))
    return RMS/np.std(y_true)


def center(y):
    return y-np.mean(y)