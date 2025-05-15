import torch
from torch.nn import Module

class TorchLayer(Module):
    def __init__(self, qnode, qnode_weights, input_arg):
        super(TorchLayer, self).__init__()
        self.qnode = qnode
        self.qnode_weights = qnode_weights
        self.input_arg = input_arg

    def _evaluate_qnode(self, x):
        kwargs = {
            **{self.input_arg: x},
            **{arg: weight.to(x) for arg, weight in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)

        if isinstance(res, torch.Tensor):
            return res.type(x.dtype)

        def _combine_dimensions(_res):
            if len(x.shape) > 1:
                _res = [torch.reshape(r, (x.shape[0], -1)) for r in _res]
            return torch.hstack(_res).type(x.dtype)

        if isinstance(res, tuple) and len(res) > 1:
            if all(isinstance(r, torch.Tensor) for r in res):
                return tuple(_combine_dimensions([r]) for r in res)  # pragma: no cover
            return tuple(_combine_dimensions(r) for r in res)

        return _combine_dimensions(res)

    def forward(self, x):
        # Evaluate the QNode with the input data
        result = self._evaluate_qnode(x)

        # If the result is a tuple, concatenate the results along the last dimension
        if isinstance(result, tuple):
            # Assuming all elements in the tuple have the same batch size
            result = torch.cat(result, dim=-1)

        # Ensure the result is a tensor
        if not isinstance(result, torch.Tensor):
            raise ValueError("The result of the QNode evaluation must be a tensor or a tuple of tensors.")

        return result