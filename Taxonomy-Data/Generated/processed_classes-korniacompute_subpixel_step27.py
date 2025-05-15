import torch
from torch import Tensor

class DepthWarper(Module):
    # ... (other parts of the class)

    def compute_subpixel_step(self) -> Tensor:
        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError("Please, call compute_projection_matrix.")

        # Define a small sub-pixel step in image space
        subpixel_step = 0.5  # This can be adjusted based on desired accuracy

        # Create a grid of pixel coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, device=self.grid.device, dtype=self.grid.dtype),
            torch.arange(self.width, device=self.grid.device, dtype=self.grid.dtype)
        )

        # Flatten the grid to iterate over each pixel
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()

        # Initialize a tensor to store the inverse depth steps
        inv_depth_steps = torch.zeros_like(x_coords, dtype=self.grid.dtype)

        # Iterate over each pixel to compute the inverse depth step
        for i in range(x_coords.size(0)):
            x = x_coords[i].item()
            y = y_coords[i].item()

            # Compute the projection at the current inverse depth
            proj_current = self._compute_projection(x, y, 1.0)

            # Compute the projection at a slightly different inverse depth
            proj_next = self._compute_projection(x, y, 1.0 + self.eps)

            # Calculate the change in projected coordinates
            delta_proj = torch.norm(proj_next - proj_current, p=2)

            # Calculate the required inverse depth step for sub-pixel accuracy
            inv_depth_steps[i] = subpixel_step / (delta_proj + self.eps)

        # Reshape the result to match the image dimensions
        inv_depth_steps = inv_depth_steps.view(self.height, self.width)

        return inv_depth_steps