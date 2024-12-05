from model import train_bayesian_nsde
import argparse
from model import BayesianNeuralSDE
from dataset import get_config_data
import torch
import matplotlib.pyplot as plt

def main(args):
    maturities = [0.175, 0.425, 0.695, 0.940]
    data_options, ivol, strikes = get_config_data()

    model = BayesianNeuralSDE(
        dim=len(maturities), 
        n_layers=args.n_layers, 
        vNetWidth=args.vNetWidth, 
        activation=args.activation, 
        activation_output=args.activation_output
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    CONFIG = {
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "timegrid": torch.tensor(maturities, dtype=torch.float32).to(args.device),
        "n_steps": len(maturities),
        "n_layers": args.n_layers,
        "vNetWidth": args.vNetWidth,
        "learning_rate": args.learning_rate,
        "target_data": data_options,
        "ivol": ivol,
        "strikes": strikes,
        "target_data": data_options
    }

    train_bayesian_nsde(model, optimizer, CONFIG, device=args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bayesian Neural SDE Model")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers in the neural network")
    parser.add_argument("--vNetWidth", type=int, default=64, help="Width of the neural network")
    parser.add_argument("--activation", type=str, default="tanh", help="Activation function")
    parser.add_argument("--activation_output", type=str, default="softplus", help="Activation function for output layer")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train the model on")
    args = parser.parse_args()
    main(args)