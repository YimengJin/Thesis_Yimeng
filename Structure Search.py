import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.cross_validation import gen_loo_cv_folds
import math

# Adapted from Milica's file
# Here we load AMBER data from a file and build an GPR model.
def load_model(filename):
    """Recreates a GPR model from saved parameters and data. """

    # load saved data
    data = np.load(filename)
    dim = data["X"].shape[1]

    # create kernel and mean functions
    kernel = GPy.kern.StdPeriodic(input_dim=dim, ARD1=True, ARD2=True)
    mean_func = GPy.mappings.Constant(dim, 1)

    # create model
    model = GPy.models.GPRegression(
        data["X"], data["Y"], kernel=kernel, mean_function=mean_func
    )

    # set model params
    model[:] = data["params"]
    model.fix()
    model.parameters_changed()

    return model

# Here, we define the AMBER emulator for a 2D structure search in d1-d4

!wget https://gitlab.com/joalof/bigmax_boss_tutorials/-/raw/319e6ba6dbbf37ac456abc6dac7f4de5ef097312/data/model_2D_E0.npz
AMBER_emulator = load_model("model_2D_E0.npz")

# Here, we define the utility function that retrieves data from the AMBER emulator.
def f(X):
    return AMBER_emulator.predict(np.atleast_2d(X))[0]