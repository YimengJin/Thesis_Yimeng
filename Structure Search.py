import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
import torch
from botorch.models import SingleTaskGP, ModelListGP, MultiTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.cross_validation import gen_loo_cv_folds
import math
import GPy
import wget


# Model data

dim = data["X"].shape[1]
# dim[X] = 2

# print(data['X'][:,0])
X1 = torch.from_numpy(data['X'][:,0])
X2 = torch.from_numpy(data['X'][:,1])
i1, i2 = torch.zeros(len(X1)), torch.ones(len(X2))
train_X = torch.stack([
    torch.cat([X1, i1], -1), torch.cat([X2, i2], -1),
])
train_Y = data['Y']
best_init_y = train_Y.max().item()
bounds = torch.tensor([[0.], [10.]])

def generate_next_sample(train_X, train_Y, best_init_y, bounds, n):
    model = MultiTaskGP(train_X, train_Y, task_feature=-1)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    EI = qExpectedImprovement(
        model=model,
        best_f=best_init_y)
    next_sample, _ = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=n,
        num_restarts=200,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200})
    return next_sample

n_runs = 10 # no. of BO runs
for i in range(n_runs):
    print(f"No. of optimisation runs:{i}")
    next_sample = generate_next_sample(train_X, train_Y, best_init_y, bounds, n=1)
    new_result = f(next_sample).unsqueeze(-1)

    print(f"The next sample is: {next_sample}")

    train_X = torch.cat([train_X, next_sample])
    train_Y = torch.cat([train_Y, new_result])
    best_init_y = init_y.max().item()
    print(f"Best point performed: {best_init_y}")





