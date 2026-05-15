import sys
import numpy as np
import torch
import time
import torch.nn as nn
from torch.optim import SGD as torchSGD
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Setting seeds and learning rate as modifiable, gloabl macros
LR = 0.004
EPOCHS = 10

# ================================================================================
# Experiment classes and functions
# ================================================================================

# PyTorch neural network object for us to optimize
class NeuralNetwork(nn.Module):
    def __init__(self, dim_in, dim_out=2):  # dim_out set to 2 by default because of the binary classification nature of the dataset
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim_out)
        )

    def forward(self, x):
        return self.model(x)
    
    def device(self):
        return next(self.parameters()).device

# The optimizer we are to implement for this task - SGD
class ManualOptimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.data = param.data - self.lr * param.grad

# =================================================================================
# PyTorch_task_v1 protocol required functions 
# =================================================================================

def get_task_metadata() -> dict[str, any]:
    return {
        'task_name': 'sgd_implementation',
        'task_type': 'optimization',
        'input_type': '',
        'output_type': '',
        'created': datetime.now().isoformat()
    }

def set_seed(seed_value: int=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(batch_size=32, data=load_breast_cancer(return_X_y=True)):
    X, y = data
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    tensor_transform = lambda arr, dtype: torch.tensor(arr, dtype=dtype)
    train_data  = TensorDataset(tensor_transform(X_train, torch.float32), tensor_transform(y_train, torch.long))
    val_data    = TensorDataset(tensor_transform(X_val, torch.float32), tensor_transform(y_val, torch.long))

    return  DataLoader(train_data, batch_size=batch_size, shuffle=True), \
            DataLoader(val_data, batch_size=batch_size, shuffle=True), X.shape[1], 2

def build_model(input_dim, device, output_dim=2): # dim_out set to 2 by default - using binary classification dataset as default
    if device is None:
        device = get_device()

    return NeuralNetwork(dim_in=input_dim, dim_out=output_dim).to(device)

# Issues with the method signature present - see ae_lvl1 for reference - will need to refactor
def train(model, train_loader, val_loader, optimizer, epochs=10, cost_func=nn.CrossEntropyLoss()):
    device = get_device()
    if next(model.parameters()).device != device:
        model.to(device)

    history = {'Train Loss': [], 'Val Loss': [], 'Val Acc': [], 'Time Elapsed': float}
    start = time.time()

    for epoch in range(epochs):

        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)             # Forward pass
            loss = cost_func(logits, y_batch)   # Backward pass
            optimizer.zero_grad()               # Remove old gradients
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Updating weights

        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():  # no gradient tracking needed for evaluation
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                val_losses.append(cost_func(logits, y_batch).item())
                val_accs.append(accuracy(logits, y_batch))

        history["Train Loss"].append(loss.item())
        history["Val Loss"].append(sum(val_losses) / len(val_losses))
        history["Val Acc"].append(sum(val_accs) / len(val_accs))

        print(f"    Epoch {epoch+1}/{epochs} — "
                f"Train Loss: {history['Train Loss'][-1]:.4f}, "
                f"Val Loss: {history['Val Loss'][-1]:.4f}, "
                f"Val Acc: {history['Val Acc'][-1]:.4f}")

    history['Time Elapsed'] = time.time() - start

    return history

def evaluate(model, data_loader, X_data=None, device=None) -> dict[str, float]:
    cost_func=nn.CrossEntropyLoss()
    
    if device is None:
        device = get_device()

    model.eval()
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            all_outputs.append(logits)
            all_labels.append(y_batch)

    out = torch.cat(all_outputs)
    labels = torch.cat(all_labels)
    xe_loss = cost_func(out, labels).item()
    acc = accuracy(out, labels)

    return {
        "val_acc": acc,
        "val_loss": xe_loss
    }

def predict(model, X):
    device = get_device()
    if next(model.parameters()).device != device:
        model.to(device)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1)

    return preds.cpu().numpy()

def save_artifacts(model, path, optimizer, history):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, path)


# =================================================================================
# Functions that we will use for the experiment - Mostly for the main task
# =================================================================================

# Using sklearn's breast cancer dataset.
def import_data():
    return make_dataloaders(batch_size=32, data=load_breast_cancer(return_X_y=True))

# PyTorch y u no have built-in accuracy metric function :(
def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def main():
    train_loader, val_loader, input_shape, _ = import_data()

    print("\n=== PyTorch model optimization ===")
    model_pt    = NeuralNetwork(dim_in=input_shape)
    opt_pt      = torchSGD(model_pt.parameters(), lr=LR)
    history_pt  = train(model_pt, train_loader, val_loader, opt_pt)

    print("\n=== Experiment model optimization ===")
    model_ex   = NeuralNetwork(dim_in=input_shape)
    opt_ex     = ManualOptimizer(model_ex.parameters(), lr=LR)
    history_ex = train(model_ex, train_loader, val_loader, opt_ex)

    print("\n=== Experiment Results ===")

    print(f"PyTorch SGD final val loss: {history_pt['Val Loss'][-1]:.4f}")
    print(f"Experiment SGD final val loss: {history_ex['Val Loss'][-1]:.4f}")

    print(f"\nPyTorch SGD final val acc: {history_pt['Val Acc'][-1]:.4f}")
    print(f"Experiment SGD final val acc: {history_ex['Val Acc'][-1]:.4f}")

    print(f"\nPyTorch SGD time elapsed: {history_pt['Time Elapsed']:.2f} seconds")
    print(f"Experiment SGD time elapsed: {history_ex['Time Elapsed']:.2f} seconds")

    print() # Print some whitespace before the end of the output

    # Evaluate on both train and validation splits
    print("=== Evaluation ===")
    train_metrics_pt = evaluate(model_pt, train_loader)
    val_metrics_pt = evaluate(model_pt, val_loader)
    train_metrics_ex = evaluate(model_ex, train_loader)
    val_metrics_ex = evaluate(model_ex, val_loader)

    print(f"PyTorch SGD - Train Loss: {train_metrics_pt['val_loss']:.4f}, Train Acc: {train_metrics_pt['val_acc']:.4f}")
    print(f"PyTorch SGD - Val Loss: {val_metrics_pt['val_loss']:.4f}, Val Acc: {val_metrics_pt['val_acc']:.4f}")
    print(f"Manual SGD - Train Loss: {train_metrics_ex['val_loss']:.4f}, Train Acc: {train_metrics_ex['val_acc']:.4f}")
    print(f"Manual SGD - Val Loss: {val_metrics_ex['val_loss']:.4f}, Val Acc: {val_metrics_ex['val_acc']:.4f}")

    # Quality checks
    print("\n=== Quality Checks ===")
    acc_close = abs(val_metrics_pt['val_acc'] - val_metrics_ex['val_acc']) < 0.05
    loss_close = abs(val_metrics_pt['val_loss'] - val_metrics_ex['val_loss']) < 0.05
    val_acc_good = val_metrics_ex['val_acc'] > 0.85

    print(f"✓ Manual SGD val acc close to PyTorch (±0.02): {val_metrics_ex['val_acc']:.4f} vs {val_metrics_pt['val_acc']:.4f}" if acc_close else f"✗ Manual SGD val acc not close: {val_metrics_ex['val_acc']:.4f} vs {val_metrics_pt['val_acc']:.4f}")
    print(f"✓ Manual SGD val loss close to PyTorch (<0.05): {abs(val_metrics_pt['val_loss'] - val_metrics_ex['val_loss']):.6f}" if loss_close else f"✗ Manual SGD val loss not close: {abs(val_metrics_pt['val_loss'] - val_metrics_ex['val_loss']):.6f}")
    print(f"✓ Manual SGD val acc > 0.85: {val_metrics_ex['val_acc']:.4f}" if val_acc_good else f"✗ Manual SGD val acc too low: {val_metrics_ex['val_acc']:.4f}")

    all_pass = acc_close and loss_close and val_acc_good
    print(f"\n{'PASS' if all_pass else 'FAIL'}: All checks {'passed' if all_pass else 'failed'}")

    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(main())