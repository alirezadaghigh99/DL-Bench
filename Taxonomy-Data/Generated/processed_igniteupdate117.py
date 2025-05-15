import torch
from typing import Any, Callable, Optional, Union

def _prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (
        x.to(device=device, non_blocking=non_blocking),
        y.to(device=device, non_blocking=non_blocking),
    )

def supervised_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable[[Any, Any], torch.Tensor], torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
    gradient_accumulation_steps: int = 1,
    model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
) -> Callable:
    """Factory function for supervised training."""
    
    def update(engine, batch):
        model.train()
        
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        
        # Forward pass
        y_pred = model_fn(model, x)
        y_pred = model_transform(y_pred)
        
        # Compute loss
        loss = loss_fn(y_pred, y) / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights if necessary
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        return output_transform(x, y, y_pred, loss)
    
    return update