import torch
import torch.nn as nn
import torch.optim as optim

def toy_model(train_loader, n_epochs=500, fit=True, in_dim=1, out_dim=1, regression=True):
    # Define a simple neural network model
    model = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim)
    )
    
    # Choose the appropriate loss function
    if regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model if fit is True
    if fit:
        model.train()  # Set the model to training mode
        for epoch in range(n_epochs):
            for inputs, targets in train_loader:
                # Forward pass
                outputs = model(inputs)
                
                # Compute the loss
                if regression:
                    loss = criterion(outputs, targets)
                else:
                    # For classification, targets should be long type
                    loss = criterion(outputs, targets.long())
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Optionally print the loss every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    return model

