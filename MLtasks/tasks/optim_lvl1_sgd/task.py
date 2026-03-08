import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD as torchSGD
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Setting seeds and learning rate as modifiable, gloabl macros
seed = 42
np.random.seed(42)
torch.manual_seed(42)
learning_rate = 0.004
epochs = 10

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

# The optimizer we are to implement for this task - SGD
class ManualOptimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p.data = p.data - self.lr * p.grad

def get_task_metadata() -> dict[str, any]:
    return {
        'task_name': 'sgd_implementation',
        'task_type': 'optimization',
        'input_type': '',
        'output_type': '',
        'created': datetime.now().isoformat()
    }

# Using sklearn's breast cancer dataset.
def import_data():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Don't need to batch the data, but helps add some noise and is the more applicable approach when working iwth large datasets
    tensor_transform = lambda arr, dtype: torch.tensor(arr, dtype=dtype)
    train_data  = TensorDataset(tensor_transform(X_train, torch.float32), tensor_transform(y_train, torch.long))
    val_data    = TensorDataset(tensor_transform(X_val, torch.float32), tensor_transform(y_val, torch.long))

    return  DataLoader(train_data, batch_size=32, shuffle=True), \
            DataLoader(val_data, batch_size=32, shuffle=False), X.shape[1], 2

# PyTorch y u no have built-in accuracy metric function :(
def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def experiment(optimizer, train, val, model):
    cost_func = nn.CrossEntropyLoss()
    history = {'Train Loss': [], 'Val Loss': [], 'Val Acc': []}

    for epoch in range(epochs):

        model.train()
        for x_batch, y_batch in train:
            logits = model(x_batch)             # Forward pass
            loss = cost_func(logits, y_batch)   # Backward pass
            optimizer.zero_grad()               # Remove old gradients
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Updating weights

        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():  # no gradient tracking needed for evaluation
            for X_batch, y_batch in val:
                logits = model(X_batch)
                val_losses.append(cost_func(logits, y_batch).item())
                val_accs.append(accuracy(logits, y_batch))

        history["Train Loss"].append(loss.item())
        history["Val Loss"].append(sum(val_losses) / len(val_losses))
        history["Val Acc"].append(sum(val_accs) / len(val_accs))

        print(f"  Epoch {epoch+1}/{epochs} — "
                f"Train Loss: {history['Train Loss'][-1]:.4f}, "
                f"Val Loss: {history['Val Loss'][-1]:.4f}, "
                f"Val Acc: {history['Val Acc'][-1]:.4f}")

    return history

def main():
    train_loader, val_loader, input_shape, _ = import_data()

    print("=== PyTorch model optimization ===")
    model_pt    = NeuralNetwork(dim_in=input_shape)
    opt_pt      = torchSGD(model_pt.parameters(), lr=learning_rate)
    history_pt  = experiment(optimizer=opt_pt, train=train_loader, val=val_loader, model=model_pt)

    print("=== Experiment model optimization ===")
    model_ex   = NeuralNetwork(dim_in=input_shape)
    opt_ex     = ManualOptimizer(model_ex.parameters(), lr=learning_rate)
    history_ex = experiment(optimizer=opt_ex, train=train_loader, val=val_loader, model=model_ex)

    print(f"PyTorch SGD final val loss: {history_pt['Val Loss'][-1]:.4f}")
    print(f"Experiment SGD final val loss: {history_ex['Val Loss'][-1]:.4f}")

    print(f"PyTorch SGD final val acc: {history_pt['Val Acc'][-1]:.4f}")
    print(f"Experiment SGD final val acc: {history_ex['Val Acc'][-1]:.4f}")

    return 0

if __name__ == '__main__':
    sys.exit(main())