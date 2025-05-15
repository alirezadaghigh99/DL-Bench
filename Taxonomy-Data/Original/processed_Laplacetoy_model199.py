def toy_model(
    train_loader: DataLoader,
    n_epochs=500,
    fit=True,
    in_dim=1,
    out_dim=1,
    regression=True,
):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 50), torch.nn.Tanh(), torch.nn.Linear(50, out_dim)
    )
    if fit:
        if regression:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=1e-2)
        for i in range(n_epochs):
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
    return model