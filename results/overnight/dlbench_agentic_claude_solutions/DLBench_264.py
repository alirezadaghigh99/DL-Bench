from collections import UserDict
from collections.abc import MutableMapping
from typing import Any, List
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

def toy_regression_dataset_1d(sigma, n_train=150, n_test=500, batch_size=150):
    torch.manual_seed(711)
    X_train = (torch.rand(n_train) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.linspace(-5, 13, n_test).unsqueeze(-1)
    return (X_train, y_train, train_loader, X_test)

def toy_multivariate_regression_dataset(sigma, d_input, n_train=150, n_test=500, batch_size=150):
    torch.manual_seed(711)
    X_train = torch.rand(n_train, d_input) * 8
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.rand(n_test, d_input) * 10
    return (X_train, y_train, train_loader, X_test)

def toy_classification_dataset(n_train=150, n_test=500, batch_size=150, in_dim=3, out_dim=2):
    torch.manual_seed(711)
    X_train = torch.randn(n_train, in_dim)
    y_train = torch.randint(out_dim, (n_train,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.randn(n_test, in_dim)
    return (X_train, y_train, train_loader, X_test)

def toy_model(train_loader: DataLoader, n_epochs=500, fit=True, in_dim=1, out_dim=1, regression=True):
    """Write a Python function named `toy_model` that builds and optionally trains a simple neural network using PyTorch. The function receives a `DataLoader` for the training data, the number of epochs (`n_epochs`), input and output dimensions (`in_dim`, `out_dim`), and flags for whether to fit the model (`fit`) and whether it's a regression task (`regression`). If `fit` is `True`, the model is trained using either Mean Squared Error loss for regression or Cross-Entropy loss for classification, with an Adam optimizer. The function should return the trained `nn.Sequential` model. Inputs include `train_loader` (PyTorch DataLoader), `n_epochs` (default 500), `fit` (default True), `in_dim` and `out_dim` (default 1), and `regression` (default True), and the output is the trained PyTorch model."""
    torch.manual_seed(711)
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 50), torch.nn.Tanh(), torch.nn.Linear(50, out_dim)
    )
    if fit:
        criterion = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(n_epochs):
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
    return model

def get_psd_matrix(dim):
    X = torch.randn(dim, dim * 3)
    return X @ X.T / (dim * 3)

def get_diag_psd_matrix(dim):
    return torch.randn(dim) ** 2

def grad(model):
    return torch.cat([p.grad.data.flatten() for p in model.parameters()]).detach()

def jacobians_naive(model, data):
    model.zero_grad()
    f = model(data)
    Jacs = list()
    for i in range(f.shape[0]):
        if len(f.shape) > 1:
            jacs = list()
            for j in range(f.shape[1]):
                rg = i != f.shape[0] - 1 or j != f.shape[1] - 1
                f[i, j].backward(retain_graph=rg)
                Jij = grad(model)
                jacs.append(Jij)
                model.zero_grad()
            jacs = torch.stack(jacs).t()
        else:
            rg = i != f.shape[0] - 1
            f[i].backward(retain_graph=rg)
            jacs = grad(model)
            model.zero_grad()
        Jacs.append(jacs)
    Jacs = torch.stack(Jacs).transpose(1, 2)
    return (Jacs.detach(), f.detach())

class ListDataset(Dataset):

    def __init__(self, data: List[Any]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def dict_data_collator(batch: List[MutableMapping]) -> UserDict:
    ret = UserDict({})
    for k in batch[0].keys():
        vals = torch.stack([v[k] for v in batch]).squeeze(-1)
        ret[k] = vals
    return ret
