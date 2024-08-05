import numpy as np
import torch
from geomloss import SamplesLoss
import time
import plotly.graph_objects as go
import random

# Check for CUDA availability
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Load and normalize your tabular data for 10 features
def load_tabular_data_10_features():
    X_i = np.random.rand(1000, 10)  # Source data with 10 features
    Y_j = np.random.rand(1000, 10)  # Target data with 10 features
    return torch.from_numpy(X_i).type(dtype), torch.from_numpy(Y_j).type(dtype)

X_i, Y_j = load_tabular_data_10_features()

def gradient_descent_10d(loss, lr=1, Nsteps=11):
    x_i, y_j = X_i.clone(), Y_j.clone()
    x_i.requires_grad = True

    iterations = {0: (x_i.clone(), y_j.clone())}

    for i in range(Nsteps):  # Euler scheme
        L_αβ = loss(x_i, y_j)
        [g] = torch.autograd.grad(L_αβ, [x_i])
        x_i.data -= lr * len(x_i) * g
        if i == 0 or i == 1 or i == Nsteps - 1:
            iterations[i+1] = (x_i.clone(), y_j.clone())

    return iterations

# Define the loss function
loss = SamplesLoss("sinkhorn", p=2, blur=0.01)

# Perform gradient descent and capture iterations
iterations = gradient_descent_10d(loss)

# Randomly select three features to plot
random_features = random.sample(range(10), 3)
print(f"Randomly selected features: {random_features}")

def create_plot(iter_num, x_i, y_j, features):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_i[:, features[0]].detach().cpu().numpy(),
        y=x_i[:, features[1]].detach().cpu().numpy(),
        z=x_i[:, features[2]].detach().cpu().numpy(),
        mode='markers',
        marker=dict(size=3, color='red'),
        name='Source'
    ))

    fig.add_trace(go.Scatter3d(
        x=y_j[:, features[0]].detach().cpu().numpy(),
        y=y_j[:, features[1]].detach().cpu().numpy(),
        z=y_j[:, features[2]].detach().cpu().numpy(),
        mode='markers',
        marker=dict(size=3, color='blue'),
        name='Target'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title=f'Feature {features[0]+1}',
            yaxis_title=f'Feature {features[1]+1}',
            zaxis_title=f'Feature {features[2]+1}'
        ),
        title=f'3D Optimal Transport Map - Iteration {iter_num}',
        width=800,
        height=800
    )

    return fig

# Generate plots for the specified iterations
for iter_num in [1, 2, 11]:
    fig = create_plot(iter_num, iterations[iter_num][0], iterations[iter_num][1], random_features)
    fig.show()
