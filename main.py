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

# Define function value
def f(X):
    y = []
    for x in X:
        y.append(np.sin(x) + 1.5*np.exp(-(x - 4.3)**2))
    return torch.tensor(np.array(y))

bounds = np.array([[0., 7.]])
fig, ax = plt.subplots()
x = np.linspace(bounds[0, 0], bounds[0, 1], 100)
y_true = np.array([f(np.atleast_2d(xi)) for xi in x])
ax.plot(x, y_true)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('1D function')
plt.show()

# Data conditioning
# train_x = torch.rand(10,1)
# exact_objective = f(train_x).unsqueeze(-1)
# best_observation = exact_objective.max().item()

# Get initial data via looping
def generate_init_data(n):
    train_x = torch.rand(n,1)
    exact_objective = f(train_x).unsqueeze(-1)
    best_observation = exact_objective.max().item()
    return train_x, exact_objective, best_observation

# print (generate_init_data(20))

# Build model
init_x, init_y, best_init_y = generate_init_data(20)
bounds = torch.tensor([[0.], [10.]])
# one_param = SingleTaskGP(init_x,init_y)
# mll = ExactMarginalLogLikelihood(one_param.likelihood, one_param)
#
# # Fit the model
# fit_gpytorch_model(mll)
#
# # Acquisition function
# EI = qExpectedImprovement(
#     model = one_param,
#     best_f = best_init_y)
#
# next_sample, _ = optimize_acqf(
#     acq_function = EI,
#     bounds = bounds,
#     q = 1,
#     num_restarts = 200,
#     raw_samples = 512,
#     options = {"batch_limit":5, "maxiter": 200}
# )
# print (next_sample)

# looping
def generate_next_sample(init_x, init_y, best_init_y, bounds, n):
    one_param = SingleTaskGP(init_x, init_y)
    mll = ExactMarginalLogLikelihood(one_param.likelihood, one_param)
    fit_gpytorch_model(mll)
    EI = qExpectedImprovement(
        model=one_param,
        best_f=best_init_y)
    next_sample, _ = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=n,
        num_restarts=200,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200})
    return next_sample


n_runs = 10
for i in range (n_runs):
    print(f"No. of optimisation runs:{i}" )
    next_sample = generate_next_sample(init_x, init_y, best_init_y, bounds, n=1)
    new_result = f(next_sample).unsqueeze (-1)

    print(f"The next sample is: {next_sample}")

    init_x = torch.cat([init_x,next_sample])
    init_y = torch.cat([init_y,new_result])
    best_init_y = init_y.max().item()
    print (f"Best point performed: {best_init_y}")









