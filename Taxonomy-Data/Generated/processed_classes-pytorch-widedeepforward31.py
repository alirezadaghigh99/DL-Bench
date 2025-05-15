import torch
from torch.autograd import Function

class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input

        """
        # Step 1: Sort input in descending order
        sorted_input, _ = torch.sort(input, descending=True, dim=dim)

        # Step 2: Calculate cumulative sum of sorted input
        cumsum_sorted_input = torch.cumsum(sorted_input, dim=dim)

        # Step 3: Create a range tensor for the dimension
        range_tensor = torch.arange(1, input.size(dim) + 1, device=input.device, dtype=input.dtype).view(
            [1] * (input.dim() - 1) + [-1]
        )

        # Step 4: Calculate the threshold
        threshold = (cumsum_sorted_input - 1) / range_tensor

        # Step 5: Find the support (k)
        support = (sorted_input > threshold).to(input.dtype)

        # Step 6: Calculate the number of elements in the support
        k = torch.sum(support, dim=dim, keepdim=True)

        # Step 7: Calculate the tau (threshold value)
        tau = (torch.sum(support * sorted_input, dim=dim, keepdim=True) - 1) / k

        # Step 8: Calculate the output
        output = torch.clamp(input - tau, min=0)

        # Save for backward pass
        ctx.save_for_backward(output, k)

        return output