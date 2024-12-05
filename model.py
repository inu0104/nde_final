import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

class BayesianNeuralSDE(nn.Module):
    def __init__(self, dim, n_layers, vNetWidth, activation="tanh", activation_output="softplus"):
        super(BayesianNeuralSDE, self).__init__()
        self.dim = dim
        self.activation = getattr(F, activation)
        self.activation_output = getattr(F, activation_output)

        # Define the drift and diffusion neural networks
        self.drift = self._build_net(dim, 40, n_layers, vNetWidth, activation)
        self.diffusion = self._build_net(dim, 1, n_layers, vNetWidth, activation_output)

    def _build_net(self, input_dim, output_dim, n_layers, width, activation):
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, output_dim))
        return nn.ModuleList(layers)

    def forward(self, x):
        drift_output = self._forward_net(self.drift, x, self.activation)  # Reshape to match target_data dimensions
        diffusion_output = self._forward_net(self.diffusion, x, self.activation_output)
        return drift_output, diffusion_output

    def _forward_net(self, net, x, activation_fn):
        for layer in net[:-1]:
            x = activation_fn(layer(x))
        x = net[-1](x)
        return x

# Function for training the Bayesian Neural SDE
# Laplace Approximation for Bayesian Calibration
def laplace_approximation(model, prior_mean=0, prior_var=1):
    log_posterior = 0
    for param in model.parameters():
        if param.requires_grad:
            log_posterior -= (param - prior_mean).pow(2).sum() / (2 * prior_var)
    # Ensure log_posterior is not negative
    log_posterior = torch.clamp(log_posterior, min=0)
    return log_posterior

def train_bayesian_nsde(model, optimizer, config, device="cuda"):
    n_epochs = config["n_epochs"]
    batch_size = config["batch_size"]
    timegrid = config["timegrid"]
    n_steps = config["n_steps"]
    target_data = torch.tensor(config["target_data"], dtype=torch.float32).to(device)
    target_data = target_data.unsqueeze(0).repeat(batch_size, 1, 1)
    eta = 0.02  # Noise level for Bayesian SGD

    model.to(device)
    model.train()

    posterior_samples = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        batch_z = torch.randn(batch_size, model.dim, device=device)

        # Forward pass
        drift, diffusion = model(batch_z)
        loss = torch.mean((drift.view(batch_size, 4, 10) - target_data) ** 2)
        # Add Laplace approximation term for Bayesian calibration
        laplace_term = laplace_approximation(model)
        print(loss)
        print(laplace_term)
        loss += torch.abs(laplace_term)

        # Backward pass with noise for Bayesian calibration
        loss.backward()
        add_noise_to_gradients(model, eta)
        optimizer.step()

        # Sample model parameters to reflect posterior uncertainty
        if epoch % 10 == 0:
            sampled_params = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad}
            posterior_samples.append(sampled_params)

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

    # Evaluate uncertainty by sampling from posterior
    evaluate_model_uncertainty(model, posterior_samples, config)

    # Plotting the model predictions vs true prices
    prediction, _ = model(torch.tensor(config["target_data"], dtype=torch.float32).view(-1, model.dim).to(next(model.parameters()).device))
    pred_cpu = prediction.detach().cpu().numpy()
    plt.plot(config["target_data"][0,:], color='dodgerblue', label='True price maturity 10', linestyle='-.')
    plt.plot(pred_cpu[0,:], color='dodgerblue', label='Model price maturity 10')
    plt.plot(config["target_data"][1,:], color='crimson', label='True price maturity 30', linestyle='-.')
    plt.plot(pred_cpu[1,:], color='crimson', label='Model price maturity 30')
    plt.plot(config["target_data"][2,:], color='orange', label='True price maturity 60', linestyle='-.')
    plt.plot(pred_cpu[2,:], color='orange', label='Model price maturity 60')
    plt.plot(config["target_data"][3,:], color='darkblue', label='True price maturity 91', linestyle='-.')
    plt.plot(pred_cpu[3,:], color='darkblue', label='Model price maturity 91')
    plt.title('Model prices versus true prices (dashed line)')
    plt.legend()
    import os
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save the plot in the results directory with unique filename based on hyperparameters
    plot_filename = f'results/result_n_layers_{config["n_layers"]}_vNetWidth_{config["vNetWidth"]}_lr_{config["learning_rate"]}_batch_{config["batch_size"]}.png'
    plt.savefig(plot_filename)

# Add Gaussian noise for Bayesian calibration
def add_noise_to_gradients(model, eta):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.normal(mean=torch.zeros_like(param.grad), std=eta)
                param.grad += noise

# Function to evaluate model uncertainty
def evaluate_model_uncertainty(model, posterior_samples, config):
    results = []  # Store results from different parameter samples
    for i, sampled_params in enumerate(posterior_samples):
        print(f"Evaluating model with sampled parameters set {i + 1}")
        with torch.no_grad():
            # Perform evaluation with the current set of sampled parameters
            prediction, _ = model(torch.tensor(config["target_data"], dtype=torch.float32).view(-1, model.dim).to(next(model.parameters()).device))
            results.append(prediction.cpu().numpy())
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in sampled_params:
                    param.copy_(sampled_params[name])
            # Calculate the mean and variance of the predictions
    results = np.array(results)
    mean_prediction = np.mean(results, axis=0)
    variance_prediction = np.var(results, axis=0)

    #print("Mean Prediction:", mean_prediction)
    #print("Prediction Variance:", variance_prediction)
