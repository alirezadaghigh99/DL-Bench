    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        output = model_fn(model, x)
        y_pred = model_transform(output)
        loss = loss_fn(y_pred, y)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()
        return output_transform(x, y, y_pred, loss * gradient_accumulation_steps)