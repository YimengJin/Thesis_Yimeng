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

# Set up GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


# Define function value
def f(X):
    y = []
    for x in X:
        y.append(np.sin(x) + 1.5 * np.exp(-(x - 4.3) ** 2))
    return torch.tensor(np.array(y))


bounds = np.array([[0., 10.]])
fig, ax = plt.subplots()
x = np.linspace(bounds[0, 0], bounds[0, 1], 100)
y_true = np.array([f(np.atleast_2d(xi)) for xi in x])
ax.plot(x, y_true)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('1D function')
# plt.show()


# Data conditioning
# train_x = torch.rand(10,1)
# exact_objective = f(train_x).unsqueeze(-1)
# best_observation = exact_objective.max().item()

# Get initial data via looping
def generate_init_data(n):
    train_x = torch.rand(n, 1)
    exact_objective = f(train_x).unsqueeze(-1)
    best_observation = exact_objective.max().item()
    return train_x, exact_objective, best_observation


# print (generate_init_data(20))

# Build model
init_data_no = 5
init_x, init_y, best_init_y = generate_init_data(init_data_no)
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
    model = SingleTaskGP(init_x, init_y)
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


n_runs = 2 # no. of BO runs
for i in range(n_runs):
    print(f"No. of optimisation runs:{i}")
    next_sample = generate_next_sample(init_x, init_y, best_init_y, bounds, n=1)
    new_result = f(next_sample).unsqueeze(-1)

    print(f"The next sample is: {next_sample}")

    init_x = torch.cat([init_x, next_sample])
    init_y = torch.cat([init_y, new_result])
    best_init_y = init_y.max().item()
    print(f"Best point performed: {best_init_y}")

# Plot results
model = SingleTaskGP(init_x, init_y)
model.eval()

# Initialize plot
f, ax = plt.subplots(1, 1, figsize=(6, 4))
# test model on 101 regular spaced points on the interval [0, 10]
test_x = torch.linspace(0, 10, 101, dtype=dtype, device=device)
# no need for gradients
with torch.no_grad():
    # compute posterior
    posterior = model.posterior(test_x)
    # Get upper and lower confidence bounds (2 standard deviations from the mean)
    lower, upper = posterior.mvn.confidence_region()
    # Plot training points as black stars
    ax.plot(init_x.cpu().numpy(), init_y.cpu().numpy(), 'k*')
    # Plot posterior means as blue line
    ax.plot(test_x.cpu().numpy(), posterior.mean.cpu().numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)
ax.legend(['Observed Data', 'Mean', 'Confidence'])
plt.tight_layout()
plt.show()

# Cross Validate
# Initialise CV dataset
# n = 20 #initial points
# sigma = math.sqrt(0.2)
# train_x = torch.rand(n,1)
# train_y_noiseless = f(train_x).view(-1)
# train_Y = train_y_noiseless + sigma * torch.randn_like(train_y_noiseless)
# train_y_var = torch.full_like(train_Y, 0.2)
#
# cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y, train_Yvar=train_y_var)

